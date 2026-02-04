import os
import glob
import pandas as pd
import numpy as np

# Root folder
root_dir = "/postproc_results/processed_avg_rewards/organized_by_alg_env"

# Process all algorithm folders
for alg_folder in glob.glob(os.path.join(root_dir, "*")):
    if not os.path.isdir(alg_folder):
        continue

    alg_name = os.path.basename(alg_folder)
    print(f"\nProcessing algorithm: {alg_name}")

    for env_folder in glob.glob(os.path.join(alg_folder, "*")):
        if not os.path.isdir(env_folder):
            continue

        env = os.path.basename(env_folder)
        reward_file = os.path.join(env_folder, f"reward_summary_{env}.csv")
        quantile_file = os.path.join(env_folder, f"reward_quantiles_{env}.csv")

        if not (os.path.exists(reward_file) and os.path.exists(quantile_file)):
            print(f"Missing files for {env} in {alg_name}")
            continue

        # Load data
        df = pd.read_csv(reward_file)
        quantiles = pd.read_csv(quantile_file).iloc[0]
        p5, p95 = quantiles["p5"], quantiles["p95"]

        # Identify step columns and min length
        step_cols = [col for col in df.columns if col.startswith("step_")]
        min_len = df[step_cols].notna().sum(axis=1).min()

        # Calculate normalized and truncated AUC
        auc_records = []
        for _, row in df.iterrows():
            rewards = row[step_cols].values[:min_len].astype(float)
            normalized = (rewards - p5) / (p95 - p5)
            auc = np.sum(normalized)

            auc_records.append({
                "actorlr": row["actorlr"],
                "criticlr": row["criticlr"],
                "entcoef": row["entcoef"],
                "gaelambda": row["gaelambda"],
                "auc": auc
            })

        auc_df = pd.DataFrame(auc_records)
        auc_df.to_csv(os.path.join(env_folder, f"auc_{env}.csv"), index=False)

        # Compute AUC quantiles
        auc_p5, auc_p95 = np.percentile(auc_df["auc"], [5, 95])
        quantile_df = pd.DataFrame([{
            "env": env,
            "auc_p5": auc_p5,
            "auc_p95": auc_p95
        }])
        quantile_df.to_csv(os.path.join(env_folder, f"auc_quantiles_{env}.csv"), index=False)

        print(f"Saved AUC table and quantiles for {env} in {alg_name}")
