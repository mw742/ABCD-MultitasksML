#!/usr/bin/env python3
import argparse, os, json
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def find_tasks(df):
    tasks = [c for c in df.columns if c.startswith("y_")]
    tasks = sorted(tasks)
    if not tasks:
        raise SystemExit("No y_* task columns found.")
    return tasks


def feature_columns(df, tasks):
    drop_prefixes = ("y_", "m_")
    drop_cols = set(["FID", "IID", "split"])
    for c in df.columns:
        if c.startswith(drop_prefixes):
            drop_cols.add(c)
    feats = [c for c in df.columns if c not in drop_cols]

    # keep only numeric features
    num_feats = []
    for c in feats:
        if pd.api.types.is_numeric_dtype(df[c]):
            num_feats.append(c)
    return num_feats


def get_mask(df, ycol):
    mcol = ycol.replace("y_", "m_", 1)
    if mcol in df.columns:
        return df[mcol].fillna(0).astype(int).values
    # fallback: if y has NaN, use non-NaN as mask
    return (~df[ycol].isna()).astype(int).values


def metrics(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    out = {}
    out["n_obs"] = int(len(y_true))
    out["prev"] = float(y_true.mean()) if len(y_true) else float("nan")
    out["auc"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan")
    out["pr_auc"] = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan")
    out["brier"] = float(brier_score_loss(y_true, y_prob))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/mtl_dataset_full.tsv")
    ap.add_argument("--outdir", default="results/final_baselines_elasticnet")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--cv", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.data, sep="\t")
    tasks = find_tasks(df)
    feats = feature_columns(df, tasks)

    # Drop all-NaN feature columns
    allnan = [c for c in feats if df[c].isna().all()]
    if allnan:
        df = df.drop(columns=allnan)
        feats = [c for c in feats if c not in allnan]

    # Fill remaining NaNs in features with 0 
    df[feats] = df[feats].fillna(0)

    print(f"[data] rows={len(df)} feats={len(feats)} tasks={tasks}")
    print(f"[data] split counts:\n{df['split'].value_counts(dropna=False)}")

    results = {"data": args.data, "outdir": args.outdir, "tasks": tasks, "n_features": len(feats)}
    results["per_task"] = {}

    # ElasticNet logistic regression pipeline
    base = Pipeline([
        ("scaler", StandardScaler(with_mean=True)),
        ("clf", LogisticRegression(
            solver="saga",
            penalty="elasticnet",
            max_iter=5000,
            random_state=args.seed,
        )),
    ])

    # Tune for AUCPR (important for rare tasks)
    param_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__l1_ratio": [0.1, 0.5, 0.9],
        "clf__class_weight": [None, "balanced"],
    }

    for ycol in tasks:
        mask = get_mask(df, ycol)

        # build splits with masks
        tr = (df["split"] == "train").values & (mask == 1)
        va = (df["split"] == "val").values & (mask == 1)
        te = (df["split"] == "test").values & (mask == 1)

        X_tr, y_tr = df.loc[tr, feats].values, df.loc[tr, ycol].fillna(0).astype(int).values
        X_va, y_va = df.loc[va, feats].values, df.loc[va, ycol].fillna(0).astype(int).values
        X_te, y_te = df.loc[te, feats].values, df.loc[te, ycol].fillna(0).astype(int).values

        if len(np.unique(y_tr)) < 2:
            print(f"[skip] {ycol}: train has <2 classes after masking")
            continue

        cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)

        gs = GridSearchCV(
            estimator=base,
            param_grid=param_grid,
            scoring="average_precision",   # AUCPR
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=0
        )
        gs.fit(X_tr, y_tr)

        best = gs.best_estimator_
        best_params = gs.best_params_
        best_cv = float(gs.best_score_)
        print(f"[{ycol}] best_cv_AUCPR={best_cv:.4f} best_params={best_params}")

        # refit on train+val (still no test leakage)
        X_tv = np.vstack([X_tr, X_va]) if len(X_va) else X_tr
        y_tv = np.concatenate([y_tr, y_va]) if len(y_va) else y_tr
        best.fit(X_tv, y_tv)

        p_va = best.predict_proba(X_va)[:, 1] if len(X_va) else np.array([])
        p_te = best.predict_proba(X_te)[:, 1] if len(X_te) else np.array([])

        out = {
            "best_params": best_params,
            "best_cv_auprc_train": best_cv,
            "val": metrics(y_va, p_va) if len(X_va) else None,
            "test": metrics(y_te, p_te) if len(X_te) else None,
            "n_train": int(len(y_tr)),
            "n_val": int(len(y_va)),
            "n_test": int(len(y_te)),
        }
        results["per_task"][ycol] = out

        # Save predictions for test set (paper figures / error analysis)
        pred_path = os.path.join(args.outdir, f"preds_{ycol}_test.tsv")
        pd.DataFrame({
            "FID": df.loc[te, "FID"].values if "FID" in df.columns else np.arange(len(y_te)),
            "IID": df.loc[te, "IID"].values if "IID" in df.columns else np.arange(len(y_te)),
            "y_true": y_te,
            "y_prob": p_te,
        }).to_csv(pred_path, sep="\t", index=False)

        # Save coefficients (interpretable)
        clf = best.named_steps["clf"]
        coefs = clf.coef_.ravel()
        coef_path = os.path.join(args.outdir, f"coef_{ycol}.tsv")
        pd.DataFrame({"feature": feats, "coef": coefs}).sort_values("coef", ascending=False)\
          .to_csv(coef_path, sep="\t", index=False)

    # Write summary json
    with open(os.path.join(args.outdir, "final_baselines_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"[done] wrote: {args.outdir}/final_baselines_summary.json")
    print("[done] wrote per-task preds_*_test.tsv and coef_*.tsv")


if __name__ == "__main__":
    main()

