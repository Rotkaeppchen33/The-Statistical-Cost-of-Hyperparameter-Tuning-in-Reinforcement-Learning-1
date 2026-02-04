import os
import glob
import pandas as pd
import numpy as np

# Root folder containing organized normalized reward files
root_dir = "/path/processed_avg_rewards/organized_by_lambda"
# Traverse each algorithm folder
for alg_folder in glob.glob(os.path.join(root_dir, "*")):
    if not os.path.isdir(alg_folder):
        continue

    alg_name = os.path.basename(alg_folder)
    print(f"\nProcessing lambda value: {alg_name}")

    # Traverse each environment
    for env_folder in glob.glob(os.path.join(alg_folder, "*")):
        if not os.path.isdir(env_folder):
            continue

        env = os.path.basename(env_folder)
        print(f"  Environment: {env}")

        # Collect all *_normalized.csv files
        norm_files = glob.glob(os.path.join(env_folder, "*_normalized.csv"))
        if not norm_files:
            print(f"    No normalized reward files found.")
            continue

        auc_records = []

        for file_path in norm_files:
            rewards = pd.read_csv(file_path, header=None).iloc[0].values.astype(float)
            auc = np.sum(rewards)

            # Extract hyperparameter values from filename
            basename = os.path.basename(file_path).replace("_normalized.csv", "")
            parts = basename.split("_")
            hp = {
                "actorlr": None,
                "criticlr": None,
                "entcoef": None,
                "gaelambda": None,
            }
            for i in range(len(parts)):
                if parts[i] in hp:
                    hp[parts[i]] = float(parts[i + 1])

            auc_records.append({
                "actorlr": hp["actorlr"],
                "criticlr": hp["criticlr"],
                "entcoef": hp["entcoef"],
                "gaelambda": hp["gaelambda"],
                "auc": auc
            })

        # Save AUC table
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

        print(f"Saved AUC values and quantiles for {env}")
