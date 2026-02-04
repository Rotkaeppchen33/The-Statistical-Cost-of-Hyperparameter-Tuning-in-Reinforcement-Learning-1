import os
import glob
import re
import numpy as np
import pandas as pd

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
os.makedirs(output_dir, exist_ok=True)

def extract_config_prefix(filename):
    match = re.match(r"(actorlr_\S+_criticlr_\S+_entcoef_\S+_gaelambda_\S+)", filename)
    return match.group(1) if match else filename

def parse_hyperparams(config_name):
    parts = config_name.split("_")
    hyperparams = {}
    for i in range(0, len(parts) - 1, 2):
        key = parts[i]
        try:
            value = float(parts[i + 1])
        except ValueError:
            value = parts[i + 1]
        hyperparams[key] = value
    return hyperparams

env_to_min_len = {}
for env in env_names:
    min_len = None
    for alg in alg_info:
        env_path = os.path.join(alg["path"], env)
        if not os.path.isdir(env_path):
            continue

        for file in glob.glob(os.path.join(env_path, "*_normalized.csv")):
            try:
                rewards = pd.read_csv(file, header=None).iloc[0].values.astype(float)
                L = len(rewards)
                if L == 0:
                    continue
                if min_len is None or L < min_len:
                    min_len = L
            except Exception as e:
                print(f"[WARN] Read failed: {file}, reason: {e}")
    if min_len is not None:
        env_to_min_len[env] = min_len
    else:
        print(f"[WARN] No valid reward files for environment {env}")

print("\n=== Min Reward Length per Environment ===")
for env, min_len in env_to_min_len.items():
    print(f"{env}: {min_len}")

all_records = []

for alg in alg_info:
    alg_name = alg["name"]
    N = alg["N"]
    base_path = alg["path"]

    for env in env_names:
        env_path = os.path.join(base_path, env)
        if not os.path.isdir(env_path):
            continue

        min_T = env_to_min_len.get(env)
        if min_T is None:
            continue
        trunc_len = min_T // (3 ** (N-2))

        for file_path in glob.glob(os.path.join(env_path, "*_normalized.csv")):
            filename = os.path.basename(file_path)
            config_prefix = extract_config_prefix(filename)

            try:
                rewards = pd.read_csv(file_path, header=None).iloc[0].values.astype(float)
                if len(rewards) < trunc_len:
                    continue  
                auc = np.sum(rewards[:trunc_len]) * (3 ** (N-2))

                quantile_path = f"/path/auc_quantiles_{env}.csv"
                if not os.path.exists(quantile_path):
                    continue
                qdf = pd.read_csv(quantile_path, sep=None, engine='python')
                auc_p5 = float(qdf.loc[qdf["env"] == env, "auc_p5"].iloc[0])
                auc_p95 = float(qdf.loc[qdf["env"] == env, "auc_p95"].iloc[0])
                norm_auc = (auc - auc_p5) / (auc_p95 - auc_p5) if (auc_p95 - auc_p5) != 0 else 0

                row = {
                    "Algorithm": alg_name,
                    "Environment": env,
                    "Truncated AUC": auc,
                    "Normalized AUC": norm_auc
                }
                row.update(parse_hyperparams(config_prefix))
                all_records.append(row)

            except Exception as e:
                print(f"Failed processing {file_path}: {e}")

df_all = pd.DataFrame(all_records)
df_all.to_csv(os.path.join(output_dir, "all_env_auc_normalized_1.csv"), index=False)

group_cols = ["Algorithm", "actorlr", "criticlr", "entcoef", "gaelambda"]
best_configs = (
    df_all.groupby(group_cols)["Normalized AUC"]
    .sum()
    .reset_index()
    .sort_values(["Algorithm", "Normalized AUC"], ascending=[True, False])
)

df_best = (
    best_configs.groupby("Algorithm")
    .first()
    .reset_index()
    .rename(columns={
        "Normalized AUC": "Best Normalized AUC Sum"
    })
)
df_best.to_csv(os.path.join(output_dir, "best_config_by_normalized_auc_1.csv"), index=False)
