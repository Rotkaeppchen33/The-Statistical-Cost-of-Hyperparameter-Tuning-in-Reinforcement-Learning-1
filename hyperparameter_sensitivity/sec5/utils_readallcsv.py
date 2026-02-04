import os
import pandas as pd
import numpy as np
import re

folder_path = "/path/csv"
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
env_alg_best = {}

for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)

    try:
        data = pd.read_csv(file_path, header=None) 
        rewards = data.values.flatten()  
        mean_reward = np.mean(rewards) if rewards.size > 0 else None 

        match = re.search(r"env_[^_]+_alg_lambda_ac", csv_file)
        if match:
            env_alg_key = match.group(0)

            if env_alg_key not in env_alg_best or mean_reward > env_alg_best[env_alg_key][1]:
                env_alg_best[env_alg_key] = (csv_file, mean_reward)

        print(f"Mean reward for {csv_file}: {mean_reward}")

    except Exception as e:
        print(f"Error reading {csv_file}: {e}")

print("\nBest files for each env_{}_alg_lambda_ac group:")
for key, (best_file, best_reward) in env_alg_best.items():
    print(f"{key}: {best_file} with mean reward {best_reward}")