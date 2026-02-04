import os
import glob
import pandas as pd
import numpy as np

# Step 1: Load min reward length per environment
min_len_path = "/path/env_min_reward_length_normalized.csv"
env_min_len_df = pd.read_csv(min_len_path, index_col=0)
env_to_min_len = env_min_len_df.iloc[:, 0].to_dict()

# Step 2: Algorithms
alg_info = [
    {
        "name": "GRPO",
        "N": 2,
        "path": "/path/"
    },
    {
        "name": "vanilla PPO",
        "N": 3,
        "path": "/path/"
    },
    {
        "name": "PPO+GAE",
        "N": 4,
        "path": "/path/"
    },
]

# Step 3: Output path
output_root = "/path/"
output_path = os.path.join(output_root, "step_to_p90_all_algorithms.csv")
all_step_records = []

# Step 3.5: Compute p5 and p95 from all PPO+GAE reward files 
ppo_gae_path = [alg["path"] for alg in alg_info if alg["name"] == "PPO+GAE"][0]
env_to_percentiles = {}

for env in env_to_min_len:
    env_path = os.path.join(ppo_gae_path, env)
    if not os.path.isdir(env_path):
        print(f"[WARN] Missing PPO+GAE folder for environment: {env}")
        continue

    min_T = int(env_to_min_len[env])
    reward_values = []

    for reward_file in glob.glob(os.path.join(env_path, "*_avg.csv")):
        try:
            df = pd.read_csv(reward_file, header=None)
            # rewards = df.iloc[0, :min_T].values.astype(float)
            rewards = df.iloc[0, :].values.astype(float)
            reward_values.extend(rewards)  
        except Exception as e:
            print(f"[WARN] Failed to process file {reward_file}: {e}")
            continue

    if reward_values:
        p5 = np.percentile(reward_values, 5)
        p90 = np.percentile(reward_values, 90)
        p95 = np.percentile(reward_values, 95)
        env_to_percentiles[env] = {"p5": p5, "p90": p90, "p95": p95}
        print(f"  {env}: p5 = {p5:.3f}, p90 = {p90:.3f}")
    else:
        print(f"  [WARN] No valid reward data for {env}")

percentile_df = pd.DataFrame.from_dict(env_to_percentiles, orient="index").reset_index()
percentile_df.columns = ["env", "p5", "p90", "p95"]
percentile_save_path = os.path.join(output_root, "reward_percentiles.csv")
percentile_df.to_csv(percentile_save_path, index=False)

print(f"\n Saved reward percentiles to: {percentile_save_path}")

# Step 4: Process each algorithm and environment, now using computed p5/p95
percentile_df = pd.read_csv(os.path.join(output_root, "reward_percentiles.csv"))
env_to_p5 = dict(zip(percentile_df["env"], percentile_df["p5"]))
env_to_p90 = dict(zip(percentile_df["env"], percentile_df["p90"]))

for alg in alg_info:
    alg_name = alg["name"]
    alg_path = alg["path"]
    print(f"\nProcessing algorithm: {alg_name}")

    for env_folder in glob.glob(os.path.join(alg_path, "*")):
        if not os.path.isdir(env_folder):
            continue

        env = os.path.basename(env_folder)
        print(f"  Environment: {env}")

        if env not in env_to_min_len:
            print(f"[WARN] No min_T for {env}, skipping.")
            continue
        min_T = int(env_to_min_len[env])

        # Load p90 threshold from PPO+GAE computed values
        try:
            p90 = env_to_p90[env]  
        except KeyError:
            print(f"[ERROR] Missing p90 for env {env}")
            continue

        # Process each reward file
        for reward_file in glob.glob(os.path.join(env_folder, "*_avg.csv")):
            try:
                df = pd.read_csv(reward_file, header=None)
                rewards = df.iloc[0, :].values.astype(float)
                # rewards = df.iloc[0, :min_T].values.astype(float)

                name = os.path.basename(reward_file).replace("_avg.csv", "")
                parts = name.split("_")

                def get_val(tag):
                    return float(parts[parts.index(tag) + 1]) if tag in parts else None

                record = {
                    "algorithm": alg_name,
                    "env": env,
                    "actorlr": get_val("actorlr"),
                    "criticlr": get_val("criticlr"),
                    "entcoef": get_val("entcoef"),
                    "gaelambda": get_val("gaelambda"),
                    "step_to_p90": np.argmax(rewards >= p90) + 1 if np.any(rewards >= p90) else np.nan,
                }
                all_step_records.append(record)
            except Exception as e:
                print(f"    [WARN] Failed to process {reward_file}: {e}")
                continue

# Step 5: Save final combined DataFrame
if all_step_records:
    final_df = pd.DataFrame(all_step_records)
    final_df.to_csv(output_path, index=False)
    print(f"\n All results saved to: {output_path}")
else:
    print(" No records to save.")
