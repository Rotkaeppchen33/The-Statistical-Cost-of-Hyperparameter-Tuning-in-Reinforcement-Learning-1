import os
import glob
import pandas as pd
import numpy as np

# Root directory containing all algorithm folders
root_dir = "/path/organized_by_lambda"

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

        # Path to shared quantiles file
        quantile_path = f"/path/{env}/reward_quantiles_{env}.csv"
        if not os.path.exists(quantile_path):
            print(f"Skipping {env}: quantile file not found.")
            continue

        try:
            qdf = pd.read_csv(quantile_path)
            p5 = float(qdf.loc[qdf["env"] == env, "p5"].iloc[0])
            p95 = float(qdf.loc[qdf["env"] == env, "p95"].iloc[0])
        except Exception as e:
            print(f"Error reading quantiles for {env}: {e}")
            continue

        # Collect all _avg.csv files
        avg_files = glob.glob(os.path.join(env_folder, "*_avg.csv"))

        for file_path in avg_files:
            if "normalized" in file_path:
                continue

            try:
                rewards = pd.read_csv(file_path, header=None).iloc[0].values.astype(float)
                normalized = (rewards - p5) / (p95 - p5)

                base_name = os.path.splitext(os.path.basename(file_path))[0]
                out_path = os.path.join(env_folder, f"{base_name}_normalized.csv")
                pd.DataFrame([normalized]).to_csv(out_path, index=False, header=False)
            except Exception as e:
                print(f"Failed to normalize {file_path}: {e}")

        print(f"Normalized {env} using p5 = {p5:.3f}, p95 = {p95:.3f}")