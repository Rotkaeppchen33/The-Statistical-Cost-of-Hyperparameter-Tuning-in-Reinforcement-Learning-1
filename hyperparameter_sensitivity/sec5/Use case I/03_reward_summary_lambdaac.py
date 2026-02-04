import os
import glob
import pandas as pd
import numpy as np

# Extract config params from filename
def extract_params(filename):
    name = filename.replace(".csv", "")
    parts = name.split("_")
    param_dict = {}
    i = 0
    while i < len(parts):
        key = parts[i]
        if key in ["actorlr", "criticlr", "entcoef", "gaelambda"]:
            value = parts[i + 1]
            param_dict[key] = value
            i += 2
        else:
            i += 1
    return param_dict

# Set root path to lambda_ac
lambda_ac_root = "/path/lambda_ac_avg"
# advn_norm_root = "/path/advn_norm_mean_avg"

# Loop through each environment folder
for env_folder in glob.glob(os.path.join(lambda_ac_root, "*")):
    if not os.path.isdir(env_folder):
        continue

    print(f"Processing environment folder: {env_folder}")
    reward_records = []

    avg_files = glob.glob(os.path.join(env_folder, "*_avg.csv"))

    for file_path in avg_files:
        filename = os.path.basename(file_path)
        rewards = pd.read_csv(file_path, header=None).iloc[0].values.astype(float)
        params = extract_params(filename)

        record = {
            "actorlr": params["actorlr"],
            "criticlr": params["criticlr"],
            "entcoef": params["entcoef"],
            "gaelambda": params["gaelambda"]
        }

        # Add reward steps as step_1, step_2, ...
        for i, r in enumerate(rewards):
            record[f"step_{i+1}"] = r

        reward_records.append(record)

    # Save per-environment summary table
    reward_df = pd.DataFrame(reward_records)
    reward_df.to_csv(os.path.join(env_folder, "reward_summary_env.csv"), index=False)

print("All per-environment summaries saved.")
