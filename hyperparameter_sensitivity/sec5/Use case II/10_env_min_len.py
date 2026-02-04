import os
import glob
import pandas as pd
import numpy as np

alg_info = [
    {
        "name": "GRPO",
        "N": 2,
        "path": "/path/organized_by_lambda/gaelambda_1.0"
    },
    {
        "name": "vanilla PPO",
        "N": 3,
        "path": "/path/processed_avg_rewards/organized_by_lambda/gaelambda_1.0"
    },
    {
        "name": "PPO+GAE",
        "N": 4,
        "path": "/path/processed_avg_rewards/organized_by_alg_env/lambda_ac"
    },
]

env_min_len = {} 

for alg in alg_info:
    alg_path = alg["path"]
    print("processing",alg)

    for env_folder in glob.glob(os.path.join(alg_path, "*")):
        if not os.path.isdir(env_folder):
            continue

        env = os.path.basename(env_folder)

        for file_path in glob.glob(os.path.join(env_folder, "*_normalized.csv")):
            try:
                rewards = pd.read_csv(file_path, header=None).iloc[0].values.astype(float)
                length = len(rewards)
                if env not in env_min_len or length < env_min_len[env]:
                    env_min_len[env] = length
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

output_dir = "/path/"
os.makedirs(output_dir, exist_ok=True)
out_path = os.path.join(output_dir, "env_min_reward_length_normalized.csv")

df = pd.DataFrame([
    {"Environment": env, "Min Reward Length": min_len}
    for env, min_len in env_min_len.items()
])

df.to_csv(out_path, index=False)
print(f"Saved to {out_path}")
