import os
import glob
import pandas as pd
import numpy as np

# Set root directory
lambda_ac_root = "/postproc_results/processed_avg_rewards/organized_by_alg_env/lambda_ac"

# Load best hyperparameters
best_hyper_path = os.path.join(lambda_ac_root, "best_hyperparams_lambda_ac_auc.csv")
best_hyper = pd.read_csv(best_hyper_path).iloc[0]

# Extract best individual values
best_actorlr = best_hyper["actorlr"]
best_criticlr = best_hyper["criticlr"]
best_entcoef = best_hyper["entcoef"]
best_gaelambda = best_hyper["gaelambda"]

all_results = []

# Iterate through each environment
for env_folder in glob.glob(os.path.join(lambda_ac_root, "*")):
    if not os.path.isdir(env_folder):
        continue

    env = os.path.basename(env_folder)
    print(f"Processing env: {env}")

    # Load normalized reward files
    normalized_files = glob.glob(os.path.join(env_folder, "*_normalized.csv"))
    rewards_by_config = []
    T = None

    # Collect rewards and parameter info
    for file_path in normalized_files:
        filename = os.path.basename(file_path).replace("_normalized.csv", "")
        parts = filename.split("_")
        params = {
            "actorlr": parts[1],
            "criticlr": parts[3],
            "entcoef": parts[5],
            "gaelambda": parts[7]
        }
        rewards = pd.read_csv(file_path, header=None).iloc[0].values.astype(float)
        if T is None:
            T = len(rewards)
        rewards_by_config.append((params, rewards))

    if T is None:
        continue

    # N = 4 to 0
    for N in range(4, -1, -1):
        for params, rewards in rewards_by_config:
            match = False
            if N == 4:
                match = True
            elif N == 3:
                match = params["actorlr"] == str(best_actorlr)
            elif N == 2:
                match = params["actorlr"] == str(best_actorlr) and params["criticlr"] == str(best_criticlr)
            elif N == 1:
                match = (
                    params["actorlr"] == str(best_actorlr) and
                    params["criticlr"] == str(best_criticlr) and
                    params["entcoef"] == str(best_entcoef)
                )
            elif N == 0:
                match = (
                    params["actorlr"] == str(best_actorlr) and
                    params["criticlr"] == str(best_criticlr) and
                    params["entcoef"] == str(best_entcoef) and
                    params["gaelambda"] == str(best_gaelambda)
                )
            if match:
                trunc_len = T // (3**N)
                auc = np.sum(rewards[:trunc_len]) * (3**N)
                all_results.append({
                    "actorlr": params["actorlr"],
                    "criticlr": params["criticlr"],
                    "entcoef": params["entcoef"],
                    "gaelambda": params["gaelambda"],
                    "env": env,
                    "N": N,
                    "auc": auc
                })

# Convert to DataFrame and save
final_auc_df = pd.DataFrame(all_results)
output_path = os.path.join(lambda_ac_root, "multi_N_auc_lambda_ac_1.csv")
final_auc_df.to_csv(output_path, index=False)
