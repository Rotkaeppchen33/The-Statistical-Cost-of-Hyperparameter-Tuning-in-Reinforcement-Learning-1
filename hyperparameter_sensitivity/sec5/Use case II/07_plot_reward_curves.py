import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

alg_info = {
    "GRPO": "/path/",
    "vanilla PPO": "/path",
    "PPO+GAE": "/path/",
}
    
env_names = ["hopper", "halfcheetah", "walker2d", "ant", "swimmer"]
output_dir = "/path/"
best_config_path = os.path.join(output_dir, "best_config_by_normalized_auc_1.csv")
df_best = pd.read_csv(best_config_path)


def build_filename_prefix(params):
    return f"actorlr_{params['actorlr']}_criticlr_{params['criticlr']}_entcoef_{params['entcoef']}_gaelambda_{params['gaelambda']}"

env_to_min_len = {}
for env in env_names:
    min_len = None
    for alg, alg_path in alg_info.items():
        best_row = df_best[df_best["Algorithm"] == alg]
        if best_row.empty:
            continue
        row = best_row.iloc[0]
        params = {
            "actorlr": row["actorlr"],
            "criticlr": row["criticlr"],
            "entcoef": row["entcoef"],
            "gaelambda": row["gaelambda"],
        }
        prefix = build_filename_prefix(params)
        pattern = os.path.join(alg_path, env, f"{prefix}_env_{env}_*_normalized.csv")
        files = glob.glob(pattern)
        if not files:
            continue
        try:
            rewards = pd.read_csv(files[0], header=None).iloc[0].values.astype(float)
            L = len(rewards)
            if L == 0:
                continue
            if min_len is None or L < min_len:
                min_len = L
        except:
            continue
    if min_len is not None:
        env_to_min_len[env] = min_len

min_len_df = pd.DataFrame([
    {"Environment": env, "Min Reward Length": min_len}
    for env, min_len in env_to_min_len.items()
])
min_len_df.to_csv(os.path.join(output_dir, "env_min_reward_length.csv"), index=False)

for env in env_names:
    min_T = env_to_min_len.get(env)
    if min_T is None:
        print(f"Skipped {env}, no valid reward lengths.")
        continue

    plt.figure(figsize=(10, 6))
    for _, row in df_best.iterrows():
        alg = row["Algorithm"]
        alg_path = alg_info[alg]

        params = {
            "actorlr": row["actorlr"],
            "criticlr": row["criticlr"],
            "entcoef": row["entcoef"],
            "gaelambda": row["gaelambda"],
        }
        prefix = build_filename_prefix(params)
        pattern = os.path.join(alg_path, env, f"{prefix}_env_{env}_*_normalized.csv")
        files = glob.glob(pattern)
        if not files:
            continue
        try:
            rewards = pd.read_csv(files[0], header=None).iloc[0].values.astype(float)
            rewards = rewards[:min_T]
            smoothed = pd.Series(rewards).rolling(window=100, min_periods=1).mean().values
            episodes = np.arange(min_T)

            plt.plot(episodes, rewards, alpha=0.3, label=f"{alg} (raw)")
            plt.plot(episodes, smoothed, label=f"{alg} (smoothed)")

        except Exception as e:
            print(f"Failed reading {files[0]}: {e}")

    plt.title(f"{env}: Reward per Episode (Raw + Smoothed)")
    plt.xlabel("Episode")
    plt.ylabel("Normalized Reward")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"rewards_{env}_raw_and_smooth.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    for _, row in df_best.iterrows():
        alg = row["Algorithm"]
        alg_path = alg_info[alg]

        params = {
            "actorlr": row["actorlr"],
            "criticlr": row["criticlr"],
            "entcoef": row["entcoef"],
            "gaelambda": row["gaelambda"],
        }
        prefix = build_filename_prefix(params)
        pattern = os.path.join(alg_path, env, f"{prefix}_env_{env}_*_normalized.csv")
        files = glob.glob(pattern)
        if not files:
            continue
        try:
            rewards = pd.read_csv(files[0], header=None).iloc[0].values.astype(float)
            rewards = rewards[:min_T]
            smoothed = pd.Series(rewards).rolling(window=100, min_periods=1).mean().values
            episodes = np.arange(min_T)

            plt.plot(episodes, smoothed, label=alg)

        except Exception as e:
            print(f"Failed reading {files[0]}: {e}")

    plt.title(f"{env}: Smoothed Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Normalized Reward")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"rewards_{env}_smooth.png"))
    plt.close()
    
    plt.figure(figsize=(10, 6))

    for _, row in df_best.iterrows():
        alg = row["Algorithm"]
        alg_path = alg_info[alg]

        params = {
            "actorlr": row["actorlr"],
            "criticlr": row["criticlr"],
            "entcoef": row["entcoef"],
            "gaelambda": row["gaelambda"],
        }
        prefix = build_filename_prefix(params)
        pattern = os.path.join(alg_path, env, f"{prefix}_env_{env}_*_normalized.csv")
        files = glob.glob(pattern)
        if not files:
            continue

        try:
            rewards = pd.read_csv(files[0], header=None).iloc[0].values.astype(float)
            rewards = rewards[:min_T]
            cumulative = np.cumsum(rewards)
            episodes = np.arange(min_T)

            plt.plot(episodes, cumulative, label=alg)
        except Exception as e:
            print(f"Failed reading {files[0]}: {e}")

    plt.title(f"{env}: Cumulative AUC per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Normalized Reward (AUC)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"auc_{env}_cumulative.png"))
    plt.close()
