import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

env_names = ["hopper", "halfcheetah", "walker2d", "ant", "swimmer"]
output_dir = "/path/"

reward_map = {}  # key: (alg, env), value: list of (file, rewards)

for alg in alg_info:
    alg_name, alg_path = alg["name"], alg["path"]
    for env in env_names:
        pattern = os.path.join(alg_path, env, f"*env_{env}_*normalized.csv")
        for file in glob.glob(pattern):
            try:
                rewards = pd.read_csv(file, header=None).iloc[0].values.astype(float)
                if len(rewards) == 0:
                    continue
                reward_map.setdefault((alg_name, env), []).append((file, rewards))
            except:
                continue

best_records = []

for (alg, env), file_rewards in reward_map.items():
    min_len = min(len(r) for (_, r) in file_rewards)
    auc_list = []
    for file, rewards in file_rewards:
        r_trunc = rewards[:min_len]
        auc = np.sum(r_trunc)
        auc_list.append((file, r_trunc, auc))
    best_file, best_reward, best_auc = max(auc_list, key=lambda x: x[2])
    best_records.append({
        "Algorithm": alg,
        "Environment": env,
        "File": best_file,
        "Length": len(best_reward),
        "AUC": best_auc
    })

df_best = pd.DataFrame(best_records)
best_config_path = os.path.join(output_dir, "best_config_by_auc.csv")
df_best.to_csv(best_config_path, index=False)
print(f"Saved best config table to {best_config_path}")

min_T = min(df_best["Length"])
curve_data = []

for _, row in df_best.iterrows():
    alg = row["Algorithm"]
    env = row["Environment"]
    file = row["File"]
    try:
        rewards = pd.read_csv(file, header=None).iloc[0].values.astype(float)
        curve_data.append((alg, env, rewards[:min_T]))
    except Exception as e:
        print(f"Error reading {file}: {e}")

algo_to_smoothed = {}
algo_to_cumulative = {}

for alg in [a["name"] for a in alg_info]:
    rs = [r[:min_T] for (a, _, r) in curve_data if a == alg] 
    if not rs:
        continue
    smoothed = [pd.Series(r).rolling(window=100, min_periods=1).mean().values for r in rs]
    cumulative = [np.cumsum(r) for r in rs]
    algo_to_smoothed[alg] = np.mean(smoothed, axis=0)
    algo_to_cumulative[alg] = np.mean(cumulative, axis=0)

plt.figure(figsize=(10, 6))
for alg, curve in algo_to_smoothed.items():
    plt.plot(np.arange(min_T), curve, linewidth=4, label=alg)
plt.xlabel("Episode", fontsize=20)
plt.xticks(fontsize=20)
plt.ylabel("Smoothed Normalized Reward", fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=16)
plt.grid(True, linewidth=0.5, alpha=0.7) 
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "average_smoothed_reward.png"))
plt.close()

plt.figure(figsize=(10, 6))
for alg, curve in algo_to_cumulative.items():
    plt.plot(np.arange(min_T), curve, linewidth=4, label=alg)
plt.xlabel("Episode", fontsize=20)
plt.xticks(fontsize=20)
plt.ylabel("Cumulative Normalized Reward", fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=16)
plt.grid(True, linewidth=0.5, alpha=0.7) 
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "average_cumulative_reward.png"))
plt.close()