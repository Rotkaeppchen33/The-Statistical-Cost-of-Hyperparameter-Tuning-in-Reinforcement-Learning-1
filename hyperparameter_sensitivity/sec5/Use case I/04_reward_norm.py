import os
import glob
import pandas as pd
import numpy as np

# Root directory containing all algorithm folders
root_dir = "/path/organized_by_alg_env"

# Traverse each algorithm folder
for alg_folder in glob.glob(os.path.join(root_dir, "*")):
    if not os.path.isdir(alg_folder):
        continue

    alg_name = os.path.basename(alg_folder)
    print(f"\nNormalizing for algorithm: {alg_name}")

    # Traverse each environment folder
    for env_folder in glob.glob(os.path.join(alg_folder, "*")):
        if not os.path.isdir(env_folder):
            continue

        env = os.path.basename(env_folder)
        quantile_path = os.path.join(env_folder, f"reward_quantiles_{env}.csv")
        if not os.path.exists(quantile_path):
            print(f"Skipping {env}: quantile file not found.")
            continue

        # Load quantiles
        q = pd.read_csv(quantile_path).iloc[0]
        p5, p95 = q["p5"], q["p95"]

        # Collect all _avg.csv files
        avg_files = glob.glob(os.path.join(env_folder, "*_avg.csv"))
        reward_lengths = []

        # First pass: determine minimum length
        for file_path in avg_files:
            if "normalized" in file_path:
                continue
            rewards = pd.read_csv(file_path, header=None).iloc[0].values
            reward_lengths.append(len(rewards))

        if not reward_lengths:
            continue

        min_len = min(reward_lengths)

        # Second pass: normalize and truncate
        for file_path in avg_files:
            if "normalized" in file_path:
                continue

            rewards = pd.read_csv(file_path, header=None).iloc[0].values.astype(float)
            truncated = rewards[:min_len]
            normalized = (truncated - p5) / (p95 - p5)

            base_name = os.path.splitext(os.path.basename(file_path))[0]
            out_path = os.path.join(env_folder, f"{base_name}_normalized.csv")
            pd.DataFrame([normalized]).to_csv(out_path, index=False, header=False)

        print(f"Normalized and truncated to {min_len} steps for {env}")
