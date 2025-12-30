#!/usr/bin/env python3
import os, argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surv_dir", default="/gpfs/group/ehlers/mwei/ABCD_2/ABCD/work_provided_assets/Env_surv")
    ap.add_argument("--ml_outdir", required=True)
    ap.add_argument("--out", default="paper_table_cox_msm_ml.tsv")
    args = ap.parse_args()

    ml_imp = pd.read_csv(os.path.join(args.ml_outdir, "feature_importance_mean.tsv"), sep="\t")
    # Keep top 50 per task
    ml_imp["rank"] = ml_imp.groupby("task")["auc_drop"].rank(ascending=False, method="dense")
    ml_top = ml_imp[ml_imp["rank"] <= 50].copy()
    ml_top["outcome"] = ml_top["task"].str.replace("^y_", "", regex=True)
    ml_top.rename(columns={"feature":"term", "auc_drop":"ml_perm_auc_drop"}, inplace=True)

    # Cox univariate with q/bonf if present
    cox_files = {
        "alcohol": "cox_tv_univar_alcohol_with_q_bonf.tsv",
        "nicotine": "cox_tv_univar_nicotine_with_q_bonf.tsv",
        "cannabis": "cox_tv_univar_cannabis_with_q_bonf.tsv",
        "any": "cox_tv_univar_any_with_q_bonf.tsv",
    }
    cox_list = []
    for oc, fn in cox_files.items():
        p = os.path.join(args.surv_dir, fn)
        if os.path.exists(p):
            d = pd.read_csv(p, sep="\t")
            d["outcome"] = oc
            keep = [c for c in ["outcome","term","HR","p","q","p_bonf","bonf_sig"] if c in d.columns]
            cox_list.append(d[keep])
    cox = pd.concat(cox_list, axis=0, ignore_index=True) if cox_list else pd.DataFrame(columns=["outcome","term"])

    # MSM stable table (primary) 
    msm_path = os.path.join(args.surv_dir, "msm_primary_eps0.01_q0.95_c10_STABLE_with_q.tsv")
    if os.path.exists(msm_path):
        msm = pd.read_csv(msm_path, sep="\t")
        keep = [c for c in ["outcome","exposure","OR_num","p_num","q","sw_p99","cap"] if c in msm.columns]
        msm = msm[keep].rename(columns={"exposure":"term", "q":"msm_q", "p_num":"msm_p", "OR_num":"msm_OR"})
    else:
        msm = pd.DataFrame(columns=["outcome","term"])

    # Merge
    out = ml_top.merge(cox, on=["outcome","term"], how="left").merge(msm, on=["outcome","term"], how="left")
    out.to_csv(args.out, sep="\t", index=False)
    print("Wrote:", args.out)

if __name__ == "__main__":
    main()

