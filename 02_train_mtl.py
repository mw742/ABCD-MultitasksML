#!/usr/bin/env python3
# 02_train_mtl.py
#
# Optuna-tuned multi-task MLP (wide+deep) with masked loss.
# ALSO trains LR + ElasticNet baselines on the same split/mask,
# writes VAL+TEST predictions for MTL and baselines,
# and learns per-task EN<->MTL blending weights on VAL, then evaluates on TEST.
#
# Outputs in --outdir:
#   - run_manifest.json
#   - dropped_all_nan_features.txt
#   - optuna.db (if --storage sqlite)
#   - optuna_trials.tsv
#   - best_params.json
#   - best_model.pt
#   - predictions_val.tsv / predictions_test.tsv               (MTL p_mlp_*)
#   - enet_predictions_val.tsv / enet_predictions_test.tsv     (p_enet_*)
#   - lr_predictions_val.tsv / lr_predictions_test.tsv         (p_lr_*)
#   - metrics.json
#   - best_model_metrics.json
#
# Usage example:
#   python 02_train_mtl.py \
#     --data results/mtl_dataset_full.tsv \
#     --outdir results/mtl_optuna_h72_seed12345_t200_wideobj \
#     --seed 12345 \
#     --n_trials 200 \
#     --hpo_max_epochs 30 \
#     --final_max_epochs 120 \
#     --patience 8 \
#     --storage "sqlite:///results/mtl_optuna_h72_seed12345_t200_wideobj/optuna.db" \
#     --study_name "mtl_optuna_h72_seed12345_t200_wideobj"

import os, json, argparse, random, math, time
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import optuna


# -------------------------
# Repro
# -------------------------
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Metrics helpers
# -------------------------
def safe_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.size == 0:
        return float("nan")
    if np.unique(y_true[~np.isnan(y_true)]).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def safe_ap(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.size == 0:
        return float("nan")
    if np.unique(y_true[~np.isnan(y_true)]).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def bootstrap_ci(metric_fn, y, p, n_boot=500, seed=123):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        yy = y[b]
        if np.unique(yy).size < 2:
            continue
        try:
            vals.append(metric_fn(yy, p[b]))
        except Exception:
            vals.append(np.nan)
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 50:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (float(lo), float(hi))


def to_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(int(x))
    s = str(x).strip().lower()
    return s in {"true", "1", "yes", "y", "t"}


# -------------------------
# Model
# -------------------------
class SharedTrunk(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...], dropout: float, use_bn: bool = False):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x):
        return self.net(x)


class MTLNet(nn.Module):
    """
    Wide + Deep multi-task net.

    deep: shared trunk + (either shared multi-head or per-task adapter + head)
    wide: linear layer from input -> logits (like multitask logistic regression)

    output: logits shape (N, T)
    """
    def __init__(
        self,
        input_dim: int,
        n_tasks: int,
        hidden_dims: Tuple[int, ...] = (256, 128),
        dropout: float = 0.2,
        use_adapter: bool = False,
        adapter_dim: int = 64,
        use_wide: bool = True,
        use_bn: bool = False,
    ):
        super().__init__()
        self.n_tasks = n_tasks
        self.use_adapter = use_adapter
        self.use_wide = use_wide

        self.shared = SharedTrunk(input_dim, hidden_dims, dropout, use_bn=use_bn)
        sd = self.shared.out_dim

        self.wide = nn.Linear(input_dim, n_tasks) if use_wide else None

        if not use_adapter:
            self.head = nn.Linear(sd, n_tasks)
            self.adapters = None
            self.task_heads = None
        else:
            self.adapters = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(sd, adapter_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity()
                ) for _ in range(n_tasks)
            ])
            self.task_heads = nn.ModuleList([nn.Linear(adapter_dim, 1) for _ in range(n_tasks)])
            self.head = None

    def forward(self, x):
        h = self.shared(x)
        if not self.use_adapter:
            deep_logits = self.head(h)
        else:
            outs = []
            for t in range(self.n_tasks):
                a = self.adapters[t](h)
                outs.append(self.task_heads[t](a))
            deep_logits = torch.cat(outs, dim=1)

        if self.wide is not None:
            return deep_logits + self.wide(x)
        return deep_logits


def masked_weighted_bce_with_logits(
    logits: torch.Tensor,
    y: torch.Tensor,
    m: torch.Tensor,
    task_weights: Optional[torch.Tensor] = None,
    pos_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
):
    """
    logits,y,m shape: (N,T), m in {0,1}

    task_weights: (T,) multiply loss per task
    pos_weights:  (T,) passed to BCEWithLogitsLoss(pos_weight=...)
    label_smoothing: small smoothing in [0, 0.1] can help calibration/regularization.
    """
    if label_smoothing > 0:
        # only smooth observed labels; masked labels already 0 and masked out
        y = y * (1.0 - label_smoothing) + 0.5 * label_smoothing

    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, y, reduction="none", pos_weight=pos_weights
    )  # (N,T)
    loss = loss * m

    if task_weights is not None:
        loss = loss * task_weights.view(1, -1)
        denom = torch.clamp((m * task_weights.view(1, -1)).sum(), min=1.0)
    else:
        denom = torch.clamp(m.sum(), min=1.0)

    return loss.sum() / denom


# -------------------------
# Data prep
# -------------------------
@dataclass
class PreparedData:
    df: pd.DataFrame
    label_cols: List[str]
    mask_cols: List[str]
    feature_cols: List[str]
    key_cols: List[str]
    dropped_all_nan_cols: List[str]
    ix_tr: np.ndarray
    ix_va: np.ndarray
    ix_te: np.ndarray
    X_tr_s: np.ndarray
    X_va_s: np.ndarray
    X_te_s: np.ndarray
    Y_tr: np.ndarray
    Y_va: np.ndarray
    Y_te: np.ndarray
    M_tr: np.ndarray
    M_va: np.ndarray
    M_te: np.ndarray
    scaler: StandardScaler


def load_and_prepare(data_path: str) -> PreparedData:
    df = pd.read_csv(data_path, sep="\t")

    if "split" not in df.columns:
        raise ValueError("Dataset must contain 'split' column with train/val/test.")

    df["split"] = df["split"].astype(str).str.strip().str.lower()
    df["split"] = df["split"].replace({
        "training": "train", "tr": "train",
        "validation": "val", "valid": "val", "va": "val", "dev": "val",
        "te": "test",
    })

    label_cols = sorted([c for c in df.columns if c.startswith("y_")])
    if not label_cols:
        raise ValueError("No y_* label columns found.")

    # masks
    mask_cols_existing = [c for c in df.columns if c.startswith("m_")]
    if not mask_cols_existing:
        for yc in label_cols:
            mc = yc.replace("y_", "m_", 1)
            df[mc] = (~df[yc].isna()).astype(np.float32)
        mask_cols_existing = [c for c in df.columns if c.startswith("m_")]

    mask_cols = [yc.replace("y_", "m_", 1) for yc in label_cols]
    missing_masks = [mc for mc in mask_cols if mc not in df.columns]
    if missing_masks:
        raise ValueError(f"Missing mask columns: {missing_masks[:10]}")

    key_cols = [c for c in ["FID", "IID", "split"] if c in df.columns]
    feature_cols = [c for c in df.columns if c not in set(key_cols + label_cols + mask_cols)]

    # Masks to {0,1}
    M_df = df[mask_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)
    M_df = (M_df > 0).astype(np.float32)

    # Labels numeric; fill NaN with 0 then mask away
    Y_df = df[label_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32).fillna(0.0)
    Y = (Y_df.values * M_df.values).astype(np.float32)

    splits = df["split"].values
    ix_tr = np.where(splits == "train")[0]
    ix_va = np.where(splits == "val")[0]
    ix_te = np.where(splits == "test")[0]
    if len(ix_tr) == 0 or len(ix_va) == 0 or len(ix_te) == 0:
        raise ValueError("Need non-empty train/val/test splits.")

    # Features: coerce numeric
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    X = X.apply(pd.to_numeric, errors="coerce")

    # drop all-NaN columns (dataset has ~55 of these)
    all_nan_cols = X.columns[X.isna().all()].tolist()
    dropped_all_nan_cols = list(all_nan_cols)
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)
        feature_cols = [c for c in feature_cols if c not in set(all_nan_cols)]

    if len(feature_cols) == 0:
        raise ValueError("No feature columns remain after dropping all-NaN.")

    # median impute
    med = X.median(numeric_only=True)
    X = X.fillna(med).fillna(0.0)

    X_np = X.values.astype(np.float32)

    # split arrays
    X_tr, X_va, X_te = X_np[ix_tr], X_np[ix_va], X_np[ix_te]
    Y_tr, Y_va, Y_te = Y[ix_tr], Y[ix_va], Y[ix_te]
    M_tr, M_va, M_te = M_df.values[ix_tr], M_df.values[ix_va], M_df.values[ix_te]

    # standardize (train only)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    def assert_finite(name, arr):
        bad = ~np.isfinite(arr)
        if bad.any():
            ex = np.argwhere(bad)[:5].tolist()
            raise ValueError(f"{name} contains NaN/Inf. Example indices: {ex}")

    assert_finite("X_tr_s", X_tr_s)
    assert_finite("X_va_s", X_va_s)
    assert_finite("X_te_s", X_te_s)

    return PreparedData(
        df=df,
        label_cols=label_cols,
        mask_cols=mask_cols,
        feature_cols=feature_cols,
        key_cols=key_cols,
        dropped_all_nan_cols=dropped_all_nan_cols,
        ix_tr=ix_tr, ix_va=ix_va, ix_te=ix_te,
        X_tr_s=X_tr_s, X_va_s=X_va_s, X_te_s=X_te_s,
        Y_tr=Y_tr, Y_va=Y_va, Y_te=Y_te,
        M_tr=M_tr, M_va=M_va, M_te=M_te,
        scaler=scaler
    )


# -------------------------
# Baselines: LR + ElasticNet + predictions
# -------------------------
def fit_baselines_and_predict(
    data: PreparedData,
    min_obs_train: int = 50,
    enet_l1_ratio: float = 0.5,
    enet_C: float = 1.0,
    lr_C: float = 1.0,
) -> Dict[str, Any]:
    """
    Fits per-task LR (L2) and ElasticNet (saga) on TRAIN observed labels,
    then returns val/test metrics and val/test probabilities for each task.
    """
    X_tr_s, X_va_s, X_te_s = data.X_tr_s, data.X_va_s, data.X_te_s
    Y_tr, Y_va, Y_te = data.Y_tr, data.Y_va, data.Y_te
    M_tr, M_va, M_te = data.M_tr, data.M_va, data.M_te
    label_cols = data.label_cols

    out: Dict[str, Any] = {
        "baseline_lr": {},
        "baseline_elasticnet": {},
        "preds": {
            "baseline_lr": {"val": {}, "test": {}},
            "baseline_elasticnet": {"val": {}, "test": {}},
        }
    }

    def train_and_predict(model_kind: str):
        for t, lab in enumerate(label_cols):
            obs_tr = (M_tr[:, t] == 1.0)
            obs_va = (M_va[:, t] == 1.0)
            obs_te = (M_te[:, t] == 1.0)

            # init placeholders
            out["preds"][model_kind]["val"][lab] = np.full((X_va_s.shape[0],), np.nan, dtype=float)
            out["preds"][model_kind]["test"][lab] = np.full((X_te_s.shape[0],), np.nan, dtype=float)

            if obs_tr.sum() < min_obs_train or np.unique(Y_tr[obs_tr, t]).size < 2:
                out[model_kind][lab] = {"note": "insufficient train labels"}
                continue

            if model_kind == "baseline_lr":
                clf = LogisticRegression(
                    solver="liblinear", penalty="l2", C=lr_C, max_iter=300
                )
            else:
                clf = LogisticRegression(
                    solver="saga", penalty="elasticnet", l1_ratio=enet_l1_ratio,
                    C=enet_C, max_iter=8000
                )

            clf.fit(X_tr_s[obs_tr], Y_tr[obs_tr, t].astype(int))

            def eval_and_store(split: str, Xs: np.ndarray, Ys: np.ndarray, obsm: np.ndarray):
                if obsm.sum() == 0:
                    return {"n_obs": 0, "prev": float("nan"), "auc": float("nan"), "pr_auc": float("nan"), "brier": float("nan")}
                y = Ys[obsm, t].astype(int)
                p = clf.predict_proba(Xs[obsm])[:, 1].astype(float)
                return {
                    "n_obs": int(obsm.sum()),
                    "prev": float(y.mean()) if len(y) else float("nan"),
                    "auc": safe_auc(y, p),
                    "pr_auc": safe_ap(y, p),
                    "brier": float(brier_score_loss(y, p)) if len(y) else float("nan")
                }

            # predictions (store only for observed rows; keep NaN for masked-out)
            if obs_va.sum() > 0:
                out["preds"][model_kind]["val"][lab][obs_va] = clf.predict_proba(X_va_s[obs_va])[:, 1]
            if obs_te.sum() > 0:
                out["preds"][model_kind]["test"][lab][obs_te] = clf.predict_proba(X_te_s[obs_te])[:, 1]

            out[model_kind][lab] = {
                "val": eval_and_store("val", X_va_s, Y_va, obs_va),
                "test": eval_and_store("test", X_te_s, Y_te, obs_te),
            }

    train_and_predict("baseline_lr")
    train_and_predict("baseline_elasticnet")
    return out


# -------------------------
# Objective + evaluation
# -------------------------
def compute_val_objective(val_metrics_by_label: Dict[str, Dict[str, float]], mode: str = "wideobj") -> float:
    """
    Returns a SINGLE scalar to maximize during HPO.

    mode="wideobj" (recommended):
      per task score = 0.5*AUC + 0.5*PR
      aggregate with weights emphasizing common tasks but still counting rare tasks:
        alcohol 0.30, any 0.30, nicotine 0.20, cannabis 0.20

    mode="auc_only":
      weighted AUC only (can hurt rare tasks)
    """
    W = {"y_alcohol": 0.30, "y_any": 0.30, "y_nicotine": 0.20, "y_cannabis": 0.20}

    def clean(x):
        try:
            x = float(x)
        except Exception:
            return 0.0
        return 0.0 if (math.isnan(x) or not math.isfinite(x)) else x

    total = 0.0
    for lab, w in W.items():
        d = val_metrics_by_label.get(lab, {})
        auc = clean(d.get("auc", float("nan")))
        pr  = clean(d.get("pr_auc", float("nan")))
        if mode == "auc_only":
            total += w * auc
        else:
            total += w * (0.5 * auc + 0.5 * pr)
    return float(total)


def evaluate_probs(P: np.ndarray, Y: np.ndarray, M: np.ndarray, label_cols: List[str], seed_for_ci: int = 12345, do_ci: bool = False):
    out: Dict[str, Any] = {}
    for t, lab in enumerate(label_cols):
        obs = (M[:, t] == 1.0)
        y = Y[obs, t].astype(int)
        p = P[obs, t].astype(float)

        auc = safe_auc(y, p)
        apv = safe_ap(y, p)
        brier = float(brier_score_loss(y, p)) if len(y) else float("nan")

        row: Dict[str, Any] = {
            "n_obs": int(obs.sum()),
            "prev": float(y.mean()) if len(y) else float("nan"),
            "auc": float(auc),
            "pr_auc": float(apv),
            "brier": brier,
        }

        if do_ci and np.isfinite(auc):
            lo, hi = bootstrap_ci(lambda yy, pp: roc_auc_score(yy, pp), y, p, n_boot=500, seed=seed_for_ci + 7 + t)
            row["auc_ci95"] = [lo, hi]
        if do_ci and np.isfinite(apv):
            lo, hi = bootstrap_ci(lambda yy, pp: average_precision_score(yy, pp), y, p, n_boot=500, seed=seed_for_ci + 11 + t)
            row["pr_auc_ci95"] = [lo, hi]

        out[lab] = row
    return out


def choose_blend_alpha_on_val(
    p_en_val: np.ndarray,
    p_mtl_val: np.ndarray,
    Y_val: np.ndarray,
    M_val: np.ndarray,
    label_cols: List[str],
    grid: int = 101,
    objective: str = "wideobj"
) -> Dict[str, float]:
    """
    For each task, choose alpha in [0,1] (weight on EN) maximizing (0.5*AUC+0.5*PR) on VAL observed labels.
    """
    alphas = np.linspace(0.0, 1.0, grid)
    best_alpha: Dict[str, float] = {}

    for t, lab in enumerate(label_cols):
        obs = (M_val[:, t] == 1.0)
        y = Y_val[obs, t].astype(int)
        if y.size == 0 or np.unique(y).size < 2:
            best_alpha[lab] = 1.0  # default to EN if no usable val labels
            continue

        pe = p_en_val[obs, t]
        pm = p_mtl_val[obs, t]

        best = (-1e18, 1.0)
        for a in alphas:
            pb = a * pe + (1.0 - a) * pm
            auc = safe_auc(y, pb)
            pr  = safe_ap(y, pb)
            # per-task score (same as wideobj per-task)
            score = 0.5 * (0.0 if not np.isfinite(auc) else auc) + 0.5 * (0.0 if not np.isfinite(pr) else pr)
            if score > best[0]:
                best = (score, float(a))
        best_alpha[lab] = best[1]

    return best_alpha


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="results/mtl_dataset_full.tsv")
    ap.add_argument("--outdir", required=True, help="results/<RUNID>")
    ap.add_argument("--seed", type=int, default=12345)

    # HPO controls
    ap.add_argument("--n_trials", type=int, default=60)
    ap.add_argument("--timeout_minutes", type=int, default=0, help="0 means no timeout")
    ap.add_argument("--study_name", default="mtl_optuna")
    ap.add_argument("--storage", default="", help="e.g. sqlite:///.../optuna.db (empty = in-memory)")
    ap.add_argument("--pruner", default="median", choices=["none", "median"])
    ap.add_argument("--sampler", default="tpe", choices=["tpe"])

    # Training
    ap.add_argument("--min_obs_train", type=int, default=50)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--hpo_max_epochs", type=int, default=30)
    ap.add_argument("--final_max_epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=8)

    # Objective mode
    ap.add_argument("--objective_mode", default="wideobj", choices=["wideobj", "auc_only"])

    # Optional: compute bootstrap CIs for final TEST metrics (slower)
    ap.add_argument("--final_boot_ci", action="store_true")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    set_all_seeds(args.seed)

    # device (CPU forced by CUDA_VISIBLE_DEVICES="" in slurm if desired)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Respect SLURM threads if set; else default to min(16, cpu_count)
    try:
        n_threads = int(os.environ.get("OMP_NUM_THREADS", "0"))
    except Exception:
        n_threads = 0
    if n_threads <= 0:
        n_threads = min(16, os.cpu_count() or 1)
    torch.set_num_threads(n_threads)

    print(f"[env] device={device} torch={torch.__version__} cuda_available={torch.cuda.is_available()} threads={n_threads}", flush=True)

    # load data
    data = load_and_prepare(args.data)
    label_cols = data.label_cols

    print(f"[data] n_train={len(data.ix_tr)} n_val={len(data.ix_va)} n_test={len(data.ix_te)}", flush=True)
    print(f"[data] n_features_raw={len([c for c in data.df.columns if c not in set(data.key_cols+data.label_cols+data.mask_cols)])}", flush=True)
    print(f"[data] dropped_all_nan_features={len(data.dropped_all_nan_cols)} -> usable_features={len(data.feature_cols)}", flush=True)
    print(f"[data] tasks={label_cols}", flush=True)

    # save manifest + dropped list
    run_manifest = {
        "data_path": args.data,
        "seed": args.seed,
        "device": str(device),
        "threads": n_threads,
        "n_train": int(len(data.ix_tr)),
        "n_val": int(len(data.ix_va)),
        "n_test": int(len(data.ix_te)),
        "label_cols": label_cols,
        "mask_cols": data.mask_cols,
        "key_cols": data.key_cols,
        "feature_cols": data.feature_cols,
        "n_features": int(len(data.feature_cols)),
        "dropped_all_nan_cols": data.dropped_all_nan_cols,
        "objective_mode": args.objective_mode,
    }
    with open(os.path.join(args.outdir, "run_manifest.json"), "w") as f:
        json.dump(run_manifest, f, indent=2)
    with open(os.path.join(args.outdir, "dropped_all_nan_features.txt"), "w") as f:
        for c in data.dropped_all_nan_cols:
            f.write(c + "\n")

    # baselines (LR + EN) + baseline predictions
    baselines = fit_baselines_and_predict(data, min_obs_train=args.min_obs_train)

    with open(os.path.join(args.outdir, "baselines_lr_elasticnet.json"), "w") as f:
        json.dump({k: v for k, v in baselines.items() if k != "preds"}, f, indent=2)

    # Create baseline prediction matrices aligned to VAL/TEST rows and tasks
    def pack_preds(split: str, model_kind: str) -> np.ndarray:
        # shape (Nsplit, T)
        if split == "val":
            n = data.X_va_s.shape[0]
        else:
            n = data.X_te_s.shape[0]
        P = np.full((n, len(label_cols)), np.nan, dtype=float)
        for t, lab in enumerate(label_cols):
            P[:, t] = baselines["preds"][model_kind][split][lab]
        return P

    P_lr_val   = pack_preds("val",  "baseline_lr")
    P_lr_test  = pack_preds("test", "baseline_lr")
    P_en_val   = pack_preds("val",  "baseline_elasticnet")
    P_en_test  = pack_preds("test", "baseline_elasticnet")

    # helper: write baseline predictions files
    def write_baseline_preds(path: str, split: str, tag: str, P: np.ndarray):
        if split == "val":
            ix = data.ix_va
            Y = data.Y_va
            M = data.M_va
        else:
            ix = data.ix_te
            Y = data.Y_te
            M = data.M_te

        df0 = data.df.loc[ix, data.key_cols].copy()
        for t, lab in enumerate(label_cols):
            df0[lab] = Y[:, t]
            df0[lab.replace("y_", "m_", 1)] = M[:, t]
            df0[lab.replace("y_", f"p_{tag}_", 1)] = P[:, t]
        df0.to_csv(path, sep="\t", index=False)

    write_baseline_preds(os.path.join(args.outdir, "lr_predictions_val.tsv"),   "val",  "lr",   P_lr_val)
    write_baseline_preds(os.path.join(args.outdir, "lr_predictions_test.tsv"),  "test", "lr",   P_lr_test)
    write_baseline_preds(os.path.join(args.outdir, "enet_predictions_val.tsv"), "val",  "enet", P_en_val)
    write_baseline_preds(os.path.join(args.outdir, "enet_predictions_test.tsv"),"test", "enet", P_en_test)

    # torch tensors
    Xtr_t = torch.tensor(data.X_tr_s, dtype=torch.float32)
    Ytr_t = torch.tensor(data.Y_tr, dtype=torch.float32)
    Mtr_t = torch.tensor(data.M_tr, dtype=torch.float32)

    Xva_t = torch.tensor(data.X_va_s, dtype=torch.float32)
    Xte_t = torch.tensor(data.X_te_s, dtype=torch.float32)

    def predict_probs(model, X_tensor, batch=4096) -> np.ndarray:
        model.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, X_tensor.shape[0], batch):
                xb = X_tensor[i:i+batch].to(device)
                logits = model(xb)
                p = torch.sigmoid(logits).detach().cpu().numpy()
                probs.append(p)
        return np.vstack(probs)

    # precompute train label stats for stable pos_weight defaults
    # pos_weight = n_neg / n_pos (PyTorch convention), clipped
    base_pos_w = []
    for t, lab in enumerate(label_cols):
        obs_tr = (data.M_tr[:, t] == 1.0)
        y = data.Y_tr[obs_tr, t].astype(int)
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        if pos <= 0:
            pw = 1.0
        else:
            pw = float(neg) / float(pos)
        pw = float(np.clip(pw, 1.0, 100.0))
        base_pos_w.append(pw)

    base_pos_w = np.asarray(base_pos_w, dtype=float)
    print("[data] base_pos_weight (train n_neg/n_pos clipped):", dict(zip(label_cols, base_pos_w.tolist())), flush=True)

    # -------------------------
    # Optuna objective
    # -------------------------
    def objective(trial: optuna.Trial) -> float:
        # architecture
        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden0 = trial.suggest_categorical("hidden0", [64, 128, 256, 512])
        hidden_dims = [hidden0]
        for i in range(1, n_layers):
            hidden_dims.append(trial.suggest_categorical(f"hidden{i}", [32, 64, 128, 256]))

        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        use_bn = trial.suggest_categorical("use_bn", [False, True])

        use_adapter = trial.suggest_categorical("use_adapter", [False, True])
        adapter_dim = trial.suggest_categorical("adapter_dim", [32, 64, 128]) if use_adapter else 64

        # wide path strongly helps beat EN
        use_wide = trial.suggest_categorical("use_wide", [True, False])

        # optimization
        lr = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)
        batch = trial.suggest_categorical("batch", [128, 256, 512])

        # mild label smoothing sometimes helps generalization
        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.05)

        # task weights: keep moderate ranges
        w_alcohol  = trial.suggest_float("w_alcohol", 0.5, 2.0)
        w_any      = trial.suggest_float("w_any", 0.5, 2.0)
        w_nicotine = trial.suggest_float("w_nicotine", 0.8, 6.0)
        w_cannabis = trial.suggest_float("w_cannabis", 0.8, 6.0)

        # pos_weight multipliers around the data-driven default
        pw_mult_alcohol  = trial.suggest_float("pw_mult_alcohol", 0.5, 2.0)
        pw_mult_any      = trial.suggest_float("pw_mult_any", 0.5, 2.0)
        pw_mult_nicotine = trial.suggest_float("pw_mult_nicotine", 0.5, 2.0)
        pw_mult_cannabis = trial.suggest_float("pw_mult_cannabis", 0.5, 2.5)

        # map task weights and pos weights by label order
        task_w = []
        pos_w = []
        for lab, base_pw in zip(label_cols, base_pos_w.tolist()):
            name = lab.replace("y_", "")
            if name == "alcohol":
                task_w.append(w_alcohol);  pos_w.append(base_pw * pw_mult_alcohol)
            elif name == "any":
                task_w.append(w_any);      pos_w.append(base_pw * pw_mult_any)
            elif name == "nicotine":
                task_w.append(w_nicotine); pos_w.append(base_pw * pw_mult_nicotine)
            elif name == "cannabis":
                task_w.append(w_cannabis); pos_w.append(base_pw * pw_mult_cannabis)
            else:
                task_w.append(1.0);        pos_w.append(base_pw)

        task_w_t = torch.tensor(task_w, dtype=torch.float32, device=device)
        pos_w_t  = torch.tensor(np.clip(pos_w, 1.0, 200.0), dtype=torch.float32, device=device)

        # model
        set_all_seeds(args.seed + trial.number)
        model = MTLNet(
            input_dim=data.X_tr_s.shape[1],
            n_tasks=len(label_cols),
            hidden_dims=tuple(hidden_dims),
            dropout=float(dropout),
            use_adapter=bool(use_adapter),
            adapter_dim=int(adapter_dim),
            use_wide=bool(use_wide),
            use_bn=bool(use_bn),
        ).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(wd))

        train_loader = DataLoader(
            TensorDataset(Xtr_t, Ytr_t, Mtr_t),
            batch_size=int(batch),
            shuffle=True,
            drop_last=False
        )

        best_score = -1e18
        bad = 0

        def eval_val_score() -> float:
            Pva = predict_probs(model, Xva_t, batch=4096)
            val_metrics = evaluate_probs(Pva, data.Y_va, data.M_va, label_cols, do_ci=False)
            return compute_val_objective(val_metrics, mode=args.objective_mode)

        for ep in range(1, args.hpo_max_epochs + 1):
            model.train()
            for xb, yb, mb in train_loader:
                xb = xb.to(device); yb = yb.to(device); mb = mb.to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = masked_weighted_bce_with_logits(
                    logits, yb, mb,
                    task_weights=task_w_t,
                    pos_weights=pos_w_t,
                    label_smoothing=float(label_smoothing),
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite loss. Check weights / data.")
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
                opt.step()

            score = eval_val_score()
            trial.report(score, step=ep)

            if args.pruner == "median" and trial.should_prune():
                raise optuna.TrialPruned()

            if score > best_score + 1e-6:
                best_score = score
                bad = 0
            else:
                bad += 1
                if bad >= args.patience:
                    break

        return float(best_score)

    # Optuna setup
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=12, n_warmup_steps=5) if args.pruner == "median" else optuna.pruners.NopPruner()

    storage = args.storage.strip() if args.storage else None
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=bool(storage),
        direction="maximize",
        sampler=sampler,
        pruner=pruner
    )

    timeout = None if args.timeout_minutes <= 0 else int(args.timeout_minutes * 60)
    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials, timeout=timeout, show_progress_bar=False)

    print(f"[optuna] completed trials={len(study.trials)} best_value={study.best_value:.6f}", flush=True)
    print(f"[optuna] best_params={study.best_params}", flush=True)
    print(f"[optuna] wall_time_sec={time.time()-t0:.1f}", flush=True)

    # save trials
    trials_rows = []
    for tr in study.trials:
        row = {"number": tr.number, "value": tr.value, "state": str(tr.state)}
        row.update(tr.params)
        trials_rows.append(row)
    pd.DataFrame(trials_rows).to_csv(os.path.join(args.outdir, "optuna_trials.tsv"), sep="\t", index=False)
    with open(os.path.join(args.outdir, "best_params.json"), "w") as f:
        json.dump(study.best_params, f, indent=2)

    # -------------------------
    # Retrain best trial with longer budget + early stopping on VAL
    # -------------------------
    best = study.best_trial
    best_seed = args.seed + int(best.number)
    set_all_seeds(best_seed)

    p = best.params

    n_layers = int(p["n_layers"])
    hidden_dims = [int(p["hidden0"])]
    for i in range(1, n_layers):
        hidden_dims.append(int(p[f"hidden{i}"]))

    dropout = float(p["dropout"])
    use_bn = to_bool(p["use_bn"])
    use_adapter = to_bool(p["use_adapter"])
    adapter_dim = int(p.get("adapter_dim", 64))
    use_wide = to_bool(p.get("use_wide", True))

    lr = float(p["lr"])
    wd = float(p["weight_decay"])
    batch = int(p["batch"])
    label_smoothing = float(p.get("label_smoothing", 0.0))

    # task weights
    wmap = {
        "alcohol": float(p["w_alcohol"]),
        "any": float(p["w_any"]),
        "nicotine": float(p["w_nicotine"]),
        "cannabis": float(p["w_cannabis"]),
    }
    # pos_weight multipliers
    pwmap = {
        "alcohol": float(p["pw_mult_alcohol"]),
        "any": float(p["pw_mult_any"]),
        "nicotine": float(p["pw_mult_nicotine"]),
        "cannabis": float(p["pw_mult_cannabis"]),
    }

    task_w = []
    pos_w = []
    for lab, base_pw in zip(label_cols, base_pos_w.tolist()):
        name = lab.replace("y_", "")
        task_w.append(wmap.get(name, 1.0))
        pos_w.append(base_pw * pwmap.get(name, 1.0))

    task_w_t = torch.tensor(task_w, dtype=torch.float32, device=device)
    pos_w_t  = torch.tensor(np.clip(pos_w, 1.0, 200.0), dtype=torch.float32, device=device)

    model = MTLNet(
        input_dim=data.X_tr_s.shape[1],
        n_tasks=len(label_cols),
        hidden_dims=tuple(hidden_dims),
        dropout=dropout,
        use_adapter=use_adapter,
        adapter_dim=adapter_dim,
        use_wide=use_wide,
        use_bn=use_bn,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    train_loader = DataLoader(
        TensorDataset(Xtr_t, Ytr_t, Mtr_t),
        batch_size=batch,
        shuffle=True,
        drop_last=False
    )

    best_score = -1e18
    best_state = None
    bad = 0

    for ep in range(1, args.final_max_epochs + 1):
        model.train()
        for xb, yb, mb in train_loader:
            xb = xb.to(device); yb = yb.to(device); mb = mb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = masked_weighted_bce_with_logits(
                logits, yb, mb,
                task_weights=task_w_t,
                pos_weights=pos_w_t,
                label_smoothing=label_smoothing,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss during retrain.")
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
            opt.step()

        # VAL score
        Pva = predict_probs(model, Xva_t, batch=4096)
        val_metrics = evaluate_probs(Pva, data.Y_va, data.M_va, label_cols, do_ci=False)
        score = compute_val_objective(val_metrics, mode=args.objective_mode)

        if score > best_score + 1e-6:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("Failed to train best model (no best_state).")

    model.load_state_dict(best_state)

    # final probs
    Pva = predict_probs(model, Xva_t, batch=4096)
    Pte = predict_probs(model, Xte_t, batch=4096)

    val_metrics = evaluate_probs(Pva, data.Y_va, data.M_va, label_cols, do_ci=False)
    test_metrics = evaluate_probs(Pte, data.Y_te, data.M_te, label_cols, seed_for_ci=best_seed, do_ci=args.final_boot_ci)

    # -------------------------
    # Blend EN + MTL using VAL, evaluate on TEST
    # -------------------------
    # Convert baseline val/test preds (NaN for masked out) to matrices aligned with VAL/TEST
    # Need shape match to (Nsplit, T). We already built P_en_val/P_en_test above.
    # Fill NaNs with 0.5 just for blending safety; they won't be evaluated because mask filters them out.
    P_en_val_f = np.nan_to_num(P_en_val, nan=0.5)
    P_en_test_f = np.nan_to_num(P_en_test, nan=0.5)

    best_alpha = choose_blend_alpha_on_val(
        p_en_val=P_en_val_f,
        p_mtl_val=Pva,
        Y_val=data.Y_va,
        M_val=data.M_va,
        label_cols=label_cols,
        grid=101,
        objective=args.objective_mode
    )

    # build blended probs on VAL/TEST
    Pblend_va = np.zeros_like(Pva, dtype=float)
    Pblend_te = np.zeros_like(Pte, dtype=float)
    for t, lab in enumerate(label_cols):
        a = float(best_alpha.get(lab, 1.0))  # alpha on EN
        Pblend_va[:, t] = a * P_en_val_f[:, t] + (1.0 - a) * Pva[:, t]
        Pblend_te[:, t] = a * P_en_test_f[:, t] + (1.0 - a) * Pte[:, t]

    blend_val_metrics = evaluate_probs(Pblend_va, data.Y_va, data.M_va, label_cols, do_ci=False)
    blend_test_metrics = evaluate_probs(Pblend_te, data.Y_te, data.M_te, label_cols, seed_for_ci=best_seed+999, do_ci=args.final_boot_ci)

    # -------------------------
    # Write MTL predictions VAL/TEST 
    # -------------------------
    def write_mtl_preds(path: str, split: str, P: np.ndarray):
        if split == "val":
            ix = data.ix_va
            Y = data.Y_va
            M = data.M_va
        else:
            ix = data.ix_te
            Y = data.Y_te
            M = data.M_te

        df0 = data.df.loc[ix, data.key_cols].copy()
        for t, lab in enumerate(label_cols):
            df0[lab] = Y[:, t]
            df0[lab.replace("y_", "m_", 1)] = M[:, t]
            df0[lab.replace("y_", "p_mlp_", 1)] = P[:, t]
        df0.to_csv(path, sep="\t", index=False)

    write_mtl_preds(os.path.join(args.outdir, "predictions_val.tsv"), "val", Pva)
    write_mtl_preds(os.path.join(args.outdir, "predictions_test.tsv"), "test", Pte)

    # also write blended predictions 
    def write_blend_preds(path: str, split: str, P: np.ndarray):
        if split == "val":
            ix = data.ix_va
            Y = data.Y_va
            M = data.M_va
        else:
            ix = data.ix_te
            Y = data.Y_te
            M = data.M_te

        df0 = data.df.loc[ix, data.key_cols].copy()
        for t, lab in enumerate(label_cols):
            df0[lab] = Y[:, t]
            df0[lab.replace("y_", "m_", 1)] = M[:, t]
            df0[lab.replace("y_", "p_blend_", 1)] = P[:, t]
        df0.to_csv(path, sep="\t", index=False)

    write_blend_preds(os.path.join(args.outdir, "blend_predictions_val.tsv"), "val", Pblend_va)
    write_blend_preds(os.path.join(args.outdir, "blend_predictions_test.tsv"), "test", Pblend_te)

    # -------------------------
    # Save checkpoint
    # -------------------------
    ckpt = {
        "model_state": model.state_dict(),
        "scaler": data.scaler,
        "feature_cols": data.feature_cols,
        "label_cols": label_cols,
        "mask_cols": data.mask_cols,
        "best_hparams": {
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "use_bn": use_bn,
            "use_adapter": use_adapter,
            "adapter_dim": adapter_dim,
            "use_wide": use_wide,
            "lr": lr,
            "weight_decay": wd,
            "batch": batch,
            "label_smoothing": label_smoothing,
            "task_weights": task_w,
            "pos_weights": pos_w,
            "grad_clip": args.grad_clip,
            "objective_mode": args.objective_mode,
        },
        "seed": best_seed,
    }
    torch.save(ckpt, os.path.join(args.outdir, "best_model.pt"))

    # -------------------------
    # Build compact metrics.json (test-focused, easy to compare)
    # -------------------------
    def extract_test_metrics_from_baselines(bdict: Dict[str, Any], key: str) -> Dict[str, Any]:
        outm = {}
        for lab in label_cols:
            d = bdict.get(key, {}).get(lab, {})
            if "test" not in d:
                outm[lab] = {"n_obs": 0, "prev": float("nan"), "auc": float("nan"), "pr_auc": float("nan"), "brier": float("nan")}
                continue
            outm[lab] = d["test"]
        return outm

    metrics_json = {
        "mtl_mlp": test_metrics,
        "baseline_lr": extract_test_metrics_from_baselines(baselines, "baseline_lr"),
        "baseline_elasticnet": extract_test_metrics_from_baselines(baselines, "baseline_elasticnet"),
        "blend_enet_mtl": blend_test_metrics,
        "blend_alpha_enet": best_alpha,
        "run": {
            "seed": best_seed,
            "device": str(device),
            "n_features": int(len(data.feature_cols)),
            "dropped_all_nan_features": int(len(data.dropped_all_nan_cols)),
            "objective_mode": args.objective_mode,
            "best_val_objective": float(best_score),
        }
    }
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics_json, f, indent=2)

    # more verbose best_model_metrics.json
    out = {
        "run": {
            "seed": best_seed,
            "device": str(device),
            "threads": n_threads,
            "n_features": int(len(data.feature_cols)),
            "n_tasks": int(len(label_cols)),
            "n_train": int(len(data.ix_tr)),
            "n_val": int(len(data.ix_va)),
            "n_test": int(len(data.ix_te)),
            "best_val_objective": float(best_score),
            "objective_mode": args.objective_mode,
            "dropped_all_nan_features": data.dropped_all_nan_cols,
        },
        "best_hparams": ckpt["best_hparams"],
        "val_metrics_mtl": val_metrics,
        "test_metrics_mtl": test_metrics,
        "val_metrics_blend": blend_val_metrics,
        "test_metrics_blend": blend_test_metrics,
        "baselines_full": {k: v for k, v in baselines.items() if k != "preds"},
        "blend_alpha_enet": best_alpha,
    }
    with open(os.path.join(args.outdir, "best_model_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("[done] wrote:", args.outdir, flush=True)
    print("[done] usable_features:", len(data.feature_cols), " dropped_all_nan:", len(data.dropped_all_nan_cols), flush=True)
    print("[done] best_val_objective:", best_score, flush=True)

    print("\n[TEST] Summary (AUC / PR-AUC):", flush=True)
    for lab in label_cols:
        mtl = test_metrics.get(lab, {})
        en  = metrics_json["baseline_elasticnet"].get(lab, {})
        bl  = blend_test_metrics.get(lab, {})
        print(f"  {lab:12s}  MTL auc={mtl.get('auc',np.nan):.4f} pr={mtl.get('pr_auc',np.nan):.4f} | "
              f"EN auc={en.get('auc',np.nan):.4f} pr={en.get('pr_auc',np.nan):.4f} | "
              f"BLEND auc={bl.get('auc',np.nan):.4f} pr={bl.get('pr_auc',np.nan):.4f}  alpha_EN={best_alpha.get(lab,1.0):.2f}",
              flush=True)


if __name__ == "__main__":
    main()

