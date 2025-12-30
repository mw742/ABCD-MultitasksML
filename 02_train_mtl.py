#!/usr/bin/env python3
import os, json, argparse, random, math, time
from typing import Dict, Any, Tuple, List

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
        try:
            vals.append(metric_fn(y[b], p[b]))
        except Exception:
            vals.append(np.nan)
    vals = np.array(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 10:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (float(lo), float(hi))


# -------------------------
# Model
# -------------------------
class SharedTrunk(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x):
        return self.net(x)


class MTLNet(nn.Module):
    """
    If use_adapter=False:
      shared -> Linear(shared_dim, n_tasks)
    If use_adapter=True:
      shared -> per-task adapter -> per-task head
    """
    def __init__(self, input_dim, n_tasks, hidden_dims=(256, 128), dropout=0.2,
                 use_adapter=False, adapter_dim=64):
        super().__init__()
        self.n_tasks = n_tasks
        self.use_adapter = use_adapter

        self.shared = SharedTrunk(input_dim, hidden_dims, dropout)
        sd = self.shared.out_dim

        if not use_adapter:
            self.head = nn.Linear(sd, n_tasks)
        else:
            self.adapters = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(sd, adapter_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ) for _ in range(n_tasks)
            ])
            self.task_heads = nn.ModuleList([nn.Linear(adapter_dim, 1) for _ in range(n_tasks)])

    def forward(self, x):
        h = self.shared(x)
        if not self.use_adapter:
            return self.head(h)  # (N, T)
        outs = []
        for t in range(self.n_tasks):
            a = self.adapters[t](h)
            logit = self.task_heads[t](a)  # (N,1)
            outs.append(logit)
        return torch.cat(outs, dim=1)  # (N, T)


def masked_weighted_bce_with_logits(logits, y, m, task_weights=None, pos_weights=None):
    """
    logits,y,m shape: (N,T), m in {0,1}
    task_weights: tensor shape (T,)
    pos_weights: tensor shape (T,) used by BCEWithLogitsLoss(pos_weight=...)
    """
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
def load_and_prepare(data_path):
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
    feature_cols = [c for c in df.columns if c not in key_cols + label_cols + mask_cols]

    # M then Y (avoid NaNs in loss)
    M_df = df[mask_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)
    M_df = (M_df > 0).astype(np.float32)

    Y_df = df[label_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32).fillna(0.0)
    Y = (Y_df.values * M_df.values).astype(np.float32)

    splits = df["split"].values
    ix_tr = np.where(splits == "train")[0]
    ix_va = np.where(splits == "val")[0]
    ix_te = np.where(splits == "test")[0]
    if len(ix_tr) == 0 or len(ix_va) == 0 or len(ix_te) == 0:
        raise ValueError("Need non-empty train/val/test splits.")

    # features numeric + impute
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    X = X.apply(pd.to_numeric, errors="coerce")
    all_nan_cols = X.columns[X.isna().all()].tolist()
    if all_nan_cols:
        print(f"[warn] dropping all-NaN feature cols: {len(all_nan_cols)}", flush=True)
        X = X.drop(columns=all_nan_cols)
        feature_cols = [c for c in feature_cols if c not in set(all_nan_cols)]
    if len(feature_cols) == 0:
        raise ValueError("No feature columns remain after dropping all-NaN.")

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

    return {
        "df": df,
        "label_cols": label_cols,
        "mask_cols": mask_cols,
        "feature_cols": feature_cols,
        "key_cols": key_cols,
        "ix_tr": ix_tr, "ix_va": ix_va, "ix_te": ix_te,
        "X_tr_s": X_tr_s, "X_va_s": X_va_s, "X_te_s": X_te_s,
        "Y_tr": Y_tr, "Y_va": Y_va, "Y_te": Y_te,
        "M_tr": M_tr, "M_va": M_va, "M_te": M_te,
        "scaler": scaler
    }


# -------------------------
# Baselines
# -------------------------
def fit_baselines_once(data, min_obs_train=50):
    X_tr_s = data["X_tr_s"]; X_va_s = data["X_va_s"]; X_te_s = data["X_te_s"]
    Y_tr = data["Y_tr"]; Y_va = data["Y_va"]; Y_te = data["Y_te"]
    M_tr = data["M_tr"]; M_va = data["M_va"]; M_te = data["M_te"]
    label_cols = data["label_cols"]

    out = {"baseline_lr": {}, "baseline_elasticnet": {}}

    def run_model(model_name, **kwargs):
        res = {}
        for t, lab in enumerate(label_cols):
            obs_tr = (M_tr[:, t] == 1.0)
            obs_va = (M_va[:, t] == 1.0)
            obs_te = (M_te[:, t] == 1.0)

            if obs_tr.sum() < min_obs_train or np.unique(Y_tr[obs_tr, t]).size < 2:
                res[lab] = {"note": "insufficient train labels"}
                continue

            clf = LogisticRegression(**kwargs)
            clf.fit(X_tr_s[obs_tr], Y_tr[obs_tr, t].astype(int))

            def eval_split(Xs, Ys, obsm):
                y = Ys[obsm, t].astype(int)
                p = clf.predict_proba(Xs[obsm])[:, 1]
                return {
                    "n_obs": int(obsm.sum()),
                    "prev": float(y.mean()) if len(y) else float("nan"),
                    "auc": safe_auc(y, p),
                    "pr_auc": safe_ap(y, p),
                }

            res[lab] = {
                "val": eval_split(X_va_s, Y_va, obs_va),
                "test": eval_split(X_te_s, Y_te, obs_te)
            }

        out[model_name] = res

    run_model("baseline_lr", solver="liblinear", penalty="l2", C=1.0, max_iter=300)
    run_model("baseline_elasticnet", solver="saga", penalty="elasticnet", l1_ratio=0.5, C=1.0, max_iter=4000)
    return out


# -------------------------
# Objective + evaluation
# -------------------------
def compute_val_objective(val_metrics_by_label: Dict[str, Dict[str, float]]) -> float:
    """
    Single objective (maximize):
      0.25*AUC(y_alcohol) + 0.25*AUC(y_any) + 0.25*PR(y_nicotine) + 0.25*PR(y_cannabis)
    Missing/NaN contributes 0.
    """
    def get(label, key):
        d = val_metrics_by_label.get(label, {})
        x = d.get(key, float("nan"))
        if x is None:
            return 0.0
        try:
            x = float(x)
        except Exception:
            return 0.0
        return 0.0 if (math.isnan(x) or not math.isfinite(x)) else x

    score = 0.0
    score += 0.25 * get("y_alcohol", "auc")
    score += 0.25 * get("y_any", "auc")
    score += 0.25 * get("y_nicotine", "pr_auc")
    score += 0.25 * get("y_cannabis", "pr_auc")
    return float(score)


def evaluate_probs(P, Y, M, label_cols, seed_for_ci=12345, do_ci=False):
    out = {}
    for t, lab in enumerate(label_cols):
        obs = (M[:, t] == 1.0)
        y = Y[obs, t].astype(int)
        p = P[obs, t]

        auc = safe_auc(y, p)
        apv = safe_ap(y, p)
        brier = float(brier_score_loss(y, p)) if len(y) else float("nan")

        row = {
            "n_obs": int(obs.sum()),
            "prev": float(y.mean()) if len(y) else float("nan"),
            "auc": float(auc),
            "pr_auc": float(apv),
            "brier": brier
        }

        if do_ci and np.isfinite(auc):
            auc_ci = bootstrap_ci(lambda yy, pp: roc_auc_score(yy, pp), y, p, n_boot=500, seed=seed_for_ci + 7 + t)
            row["auc_ci95"] = [auc_ci[0], auc_ci[1]]
        if do_ci and np.isfinite(apv):
            ap_ci = bootstrap_ci(lambda yy, pp: average_precision_score(yy, pp), y, p, n_boot=500, seed=seed_for_ci + 11 + t)
            row["pr_auc_ci95"] = [ap_ci[0], ap_ci[1]]

        out[lab] = row
    return out


def to_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(int(x))
    s = str(x).strip().lower()
    return s in {"true", "1", "yes", "y", "t"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="results/mtl_dataset_full.tsv")
    ap.add_argument("--outdir", required=True, help="results/<RUNID>")
    ap.add_argument("--seed", type=int, default=12345)

    # HPO controls
    ap.add_argument("--n_trials", type=int, default=60)
    ap.add_argument("--timeout_minutes", type=int, default=0, help="0 means no timeout")
    ap.add_argument("--study_name", default="mtl_optuna")
    ap.add_argument("--storage", default="", help="e.g. sqlite:///results/RUNID/optuna.db (empty = in-memory)")
    ap.add_argument("--pruner", default="median", choices=["none", "median"])
    ap.add_argument("--sampler", default="tpe", choices=["tpe"])

    # Training
    ap.add_argument("--max_epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--min_obs_train", type=int, default=50)
    ap.add_argument("--grad_clip", type=float, default=5.0)

    # Compute CIs for final test only (costs time)
    ap.add_argument("--final_boot_ci", action="store_true")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    set_all_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    print(f"[env] device={device} torch={torch.__version__} cuda_available={torch.cuda.is_available()}", flush=True)

    # load data
    data = load_and_prepare(args.data)
    label_cols = data["label_cols"]
    print(f"[data] n_train={len(data['ix_tr'])} n_val={len(data['ix_va'])} n_test={len(data['ix_te'])}", flush=True)
    print(f"[data] n_features={data['X_tr_s'].shape[1]} n_tasks={len(label_cols)}", flush=True)
    print(f"[data] tasks={label_cols}", flush=True)

    # baselines
    baselines = fit_baselines_once(data, min_obs_train=args.min_obs_train)
    with open(os.path.join(args.outdir, "baselines_lr_elasticnet.json"), "w") as f:
        json.dump(baselines, f, indent=2)

    # torch tensors
    Xtr_t = torch.tensor(data["X_tr_s"], dtype=torch.float32)
    Ytr_t = torch.tensor(data["Y_tr"], dtype=torch.float32)
    Mtr_t = torch.tensor(data["M_tr"], dtype=torch.float32)

    Xva_t = torch.tensor(data["X_va_s"], dtype=torch.float32)
    Yva_t = torch.tensor(data["Y_va"], dtype=torch.float32)
    Mva_t = torch.tensor(data["M_va"], dtype=torch.float32)

    Xte_t = torch.tensor(data["X_te_s"], dtype=torch.float32)
    Yte_t = torch.tensor(data["Y_te"], dtype=torch.float32)
    Mte_t = torch.tensor(data["M_te"], dtype=torch.float32)

    df = data["df"]
    ix_te = data["ix_te"]
    key_cols = data["key_cols"]

    def predict_probs(model, X_tensor, batch=2048):
        model.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, X_tensor.shape[0], batch):
                xb = X_tensor[i:i+batch].to(device)
                logits = model(xb)
                p = torch.sigmoid(logits).detach().cpu().numpy()
                probs.append(p)
        return np.vstack(probs)

    def objective(trial: optuna.Trial):
        # architecture
        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden0 = trial.suggest_categorical("hidden0", [64, 128, 256, 512])
        hidden_dims = [hidden0]
        for i in range(1, n_layers):
            hidden_dims.append(trial.suggest_categorical(f"hidden{i}", [32, 64, 128, 256]))

        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        use_adapter = trial.suggest_categorical("use_adapter", [False, True])
        adapter_dim = trial.suggest_categorical("adapter_dim", [32, 64, 128]) if use_adapter else 0

        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
        batch = trial.suggest_categorical("batch", [128, 256, 512])

        w_alcohol = trial.suggest_float("w_alcohol", 0.5, 2.0)
        w_any = trial.suggest_float("w_any", 0.5, 2.0)
        w_nicotine = trial.suggest_float("w_nicotine", 1.0, 8.0)
        w_cannabis = trial.suggest_float("w_cannabis", 1.0, 8.0)

        pw_alcohol = trial.suggest_float("pw_alcohol", 1.0, 5.0)
        pw_any = trial.suggest_float("pw_any", 1.0, 5.0)
        pw_nicotine = trial.suggest_float("pw_nicotine", 1.0, 50.0)
        pw_cannabis = trial.suggest_float("pw_cannabis", 1.0, 80.0)

        # map weights by label order
        task_w = []
        pos_w = []
        for lab in label_cols:
            name = lab.replace("y_", "")
            if name == "alcohol":
                task_w.append(w_alcohol); pos_w.append(pw_alcohol)
            elif name == "any":
                task_w.append(w_any); pos_w.append(pw_any)
            elif name == "nicotine":
                task_w.append(w_nicotine); pos_w.append(pw_nicotine)
            elif name == "cannabis":
                task_w.append(w_cannabis); pos_w.append(pw_cannabis)
            else:
                task_w.append(1.0); pos_w.append(1.0)

        task_w_t = torch.tensor(task_w, dtype=torch.float32, device=device)
        pos_w_t = torch.tensor(pos_w, dtype=torch.float32, device=device)

        # model
        set_all_seeds(args.seed + trial.number)
        model = MTLNet(
            input_dim=data["X_tr_s"].shape[1],
            n_tasks=len(label_cols),
            hidden_dims=tuple(hidden_dims),
            dropout=dropout,
            use_adapter=use_adapter,
            adapter_dim=(adapter_dim if use_adapter else 64)
        ).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        train_loader = DataLoader(TensorDataset(Xtr_t, Ytr_t, Mtr_t), batch_size=batch, shuffle=True)
        best_score = -1e18
        bad = 0

        def eval_val_score():
            Pva = predict_probs(model, Xva_t, batch=4096)
            val_metrics = evaluate_probs(Pva, data["Y_va"], data["M_va"], label_cols, do_ci=False)
            score = compute_val_objective(val_metrics)
            return score

        for ep in range(1, args.max_epochs + 1):
            model.train()
            for xb, yb, mb in train_loader:
                xb = xb.to(device); yb = yb.to(device); mb = mb.to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = masked_weighted_bce_with_logits(logits, yb, mb, task_weights=task_w_t, pos_weights=pos_w_t)

                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite loss. Check weights / data.")
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
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
    sampler = optuna.samplers.TPESampler(seed=args.seed)  # NOT multivariate (your space is dynamic)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=5) if args.pruner == "median" else optuna.pruners.NopPruner()

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
    # Retrain best trial + save artifacts
    # -------------------------
    best = study.best_trial
    best_seed = args.seed + best.number
    set_all_seeds(best_seed)

    params = best.params
    n_layers = int(params["n_layers"])
    hidden_dims = [int(params["hidden0"])]
    for i in range(1, n_layers):
        hidden_dims.append(int(params[f"hidden{i}"]))

    dropout = float(params["dropout"])
    use_adapter = to_bool(params["use_adapter"])
    adapter_dim = int(params.get("adapter_dim", 64)) if use_adapter else 0
    lr = float(params["lr"])
    wd = float(params["weight_decay"])
    batch = int(params["batch"])

    wmap = {
        "alcohol": float(params["w_alcohol"]),
        "any": float(params["w_any"]),
        "nicotine": float(params["w_nicotine"]),
        "cannabis": float(params["w_cannabis"]),
    }
    pwmap = {
        "alcohol": float(params["pw_alcohol"]),
        "any": float(params["pw_any"]),
        "nicotine": float(params["pw_nicotine"]),
        "cannabis": float(params["pw_cannabis"]),
    }

    task_w = []
    pos_w = []
    for lab in label_cols:
        name = lab.replace("y_", "")
        task_w.append(wmap.get(name, 1.0))
        pos_w.append(pwmap.get(name, 1.0))
    task_w_t = torch.tensor(task_w, dtype=torch.float32, device=device)
    pos_w_t = torch.tensor(pos_w, dtype=torch.float32, device=device)

    model = MTLNet(
        input_dim=data["X_tr_s"].shape[1],
        n_tasks=len(label_cols),
        hidden_dims=tuple(hidden_dims),
        dropout=dropout,
        use_adapter=use_adapter,
        adapter_dim=(adapter_dim if use_adapter else 64)
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    train_loader = DataLoader(TensorDataset(Xtr_t, Ytr_t, Mtr_t), batch_size=batch, shuffle=True)

    best_score = -1e18
    best_state_dict = None
    bad = 0

    for ep in range(1, args.max_epochs + 1):
        model.train()
        for xb, yb, mb in train_loader:
            xb = xb.to(device); yb = yb.to(device); mb = mb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = masked_weighted_bce_with_logits(logits, yb, mb, task_weights=task_w_t, pos_weights=pos_w_t)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss during retrain.")
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            opt.step()

        Pva = predict_probs(model, Xva_t, batch=4096)
        val_metrics = evaluate_probs(Pva, data["Y_va"], data["M_va"], label_cols, do_ci=False)
        score = compute_val_objective(val_metrics)

        if score > best_score + 1e-6:
            best_score = score
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break

    if best_state_dict is None:
        raise RuntimeError("Failed to train best model.")

    model.load_state_dict(best_state_dict)

    Pva = predict_probs(model, Xva_t, batch=4096)
    Pte = predict_probs(model, Xte_t, batch=4096)

    val_metrics = evaluate_probs(Pva, data["Y_va"], data["M_va"], label_cols, do_ci=False)
    test_metrics = evaluate_probs(Pte, data["Y_te"], data["M_te"], label_cols, seed_for_ci=best_seed, do_ci=args.final_boot_ci)

    out = {
        "run": {
            "seed": best_seed,
            "device": str(device),
            "n_features": int(data["X_tr_s"].shape[1]),
            "n_tasks": int(len(label_cols)),
            "n_train": int(len(data["ix_tr"])),
            "n_val": int(len(data["ix_va"])),
            "n_test": int(len(data["ix_te"])),
            "best_val_objective": float(best_score),
            "objective_definition": "0.25*AUC(y_alcohol)+0.25*AUC(y_any)+0.25*PR(y_nicotine)+0.25*PR(y_cannabis)",
        },
        "best_hparams": {
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "use_adapter": use_adapter,
            "adapter_dim": adapter_dim,
            "lr": lr,
            "weight_decay": wd,
            "batch": batch,
            "task_weights": task_w,
            "pos_weights": pos_w,
            "grad_clip": args.grad_clip,
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "baselines": baselines
    }

    with open(os.path.join(args.outdir, "best_model_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    # save predictions on test
    pred_df = df.loc[ix_te, key_cols].copy()
    for t, lab in enumerate(label_cols):
        pred_df[lab] = data["Y_te"][:, t]
        pred_df[lab.replace("y_", "m_", 1)] = data["M_te"][:, t]
        pred_df[lab.replace("y_", "p_mlp_", 1)] = Pte[:, t]
    pred_df.to_csv(os.path.join(args.outdir, "predictions_test.tsv"), sep="\t", index=False)

    # save torch checkpoint
    ckpt = {
        "model_state": model.state_dict(),
        "scaler": data["scaler"],
        "feature_cols": data["feature_cols"],
        "label_cols": label_cols,
        "mask_cols": data["mask_cols"],
        "best_hparams": out["best_hparams"],
        "seed": best_seed
    }
    torch.save(ckpt, os.path.join(args.outdir, "best_model.pt"))

    print("[done] wrote:", args.outdir, flush=True)
    print("[done] best_val_objective:", best_score, flush=True)
    print("[done] test metrics summary:", flush=True)
    for lab, d in test_metrics.items():
        print(" ", lab, "AUC=", d["auc"], "PR=", d["pr_auc"], "prev=", d["prev"], "n=", d["n_obs"], flush=True)


if __name__ == "__main__":
    main()

