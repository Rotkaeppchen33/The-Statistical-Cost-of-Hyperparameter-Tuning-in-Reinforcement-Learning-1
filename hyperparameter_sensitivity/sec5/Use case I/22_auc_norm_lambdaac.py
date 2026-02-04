import pandas as pd
import os

# Define root directory and input file path
lambda_ac_root = "/postproc_results/processed_avg_rewards/organized_by_alg_env/lambda_ac"
multi_auc_path = os.path.join(lambda_ac_root, "multi_N_auc_lambda_ac_1.csv")

# Load multi-N AUC file
multi_auc_df = pd.read_csv(multi_auc_path)

# Prepare list to collect normalized results
normalized_auc_records = []

# Normalize AUC per environment using its own quantiles
for env in multi_auc_df["env"].unique():
    env_folder = os.path.join(lambda_ac_root, env)
    quantile_path = os.path.join(env_folder, f"auc_quantiles_{env}.csv")

    if not os.path.exists(quantile_path):
        print(f"Skipping {env}: quantile file not found.")
        continue

    # Read quantiles for this environment
    q = pd.read_csv(quantile_path).iloc[0]
    p5, p95 = q["auc_p5"], q["auc_p95"]

    # Normalize AUC values within this environment
    env_df = multi_auc_df[multi_auc_df["env"] == env].copy()
    env_df["normalized_auc"] = (env_df["auc"] - p5) / (p95 - p5)
    normalized_auc_records.append(env_df)

# Combine all environments and save result
normalized_auc_df = pd.concat(normalized_auc_records, ignore_index=True)
output_path = os.path.join(lambda_ac_root, "normalized_multi_N_auc_lambda_ac_1.csv")
normalized_auc_df.to_csv(output_path, index=False)
