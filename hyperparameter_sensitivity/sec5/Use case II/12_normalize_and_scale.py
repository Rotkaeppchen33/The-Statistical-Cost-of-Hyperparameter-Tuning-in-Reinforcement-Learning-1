import pandas as pd
import numpy as np
import os

envs = ["hopper", "walker2d", "halfcheetah", "swimmer", "ant"]
hyperparams = ["actorlr", "criticlr", "entcoef", "gaelambda"]
metric = "step_to_p90"
output_dir = "/path/"
input_file = os.path.join(output_dir, "step_to_p90_all_algorithms.csv")
df_all = pd.read_csv(input_file)

baseline_df = df_all[df_all["algorithm"] == "PPO+GAE"]
baseline_values = {}

for env in envs:
    baseline_env = baseline_df[baseline_df["env"] == env][metric]
    max_valid = baseline_env.dropna().max()
    penalty = 10 * max_valid
    baseline_filled = baseline_env.fillna(penalty)
    baseline = np.percentile(baseline_filled, 5)
    baseline_values[env] = baseline

print("Baseline values from PPO+GAE (N=4):")
print(baseline_values)

baseline_df_out = pd.DataFrame([
    {"env": env, "baseline_p5": baseline}
    for env, baseline in baseline_values.items()
])
baseline_file = os.path.join(output_dir, f"{metric}_baseline_from_PPO_GAE.csv")
baseline_df_out.to_csv(baseline_file, index=False)
print(f" Saved baseline values to {baseline_file}")

def get_N_from_algorithm(alg_name):
    return {
        "GRPO": 2,
        "vanilla PPO": 3,
        "PPO+GAE": 4
    }[alg_name]

all_records = []

for alg_name in df_all["algorithm"].unique():
    N = get_N_from_algorithm(alg_name)
    print(f"\nProcessing algorithm: {alg_name} (N={N})")

    for env in envs:
        df_sub = df_all[(df_all["algorithm"] == alg_name) & (df_all["env"] == env)].copy()
        if df_sub.empty:
            continue

        max_valid = df_sub[metric].dropna().max()
        penalty = 10 * max_valid
        df_sub[metric] = df_sub[metric].fillna(penalty)

        baseline = baseline_values[env]
        df_sub[f"{metric}_normalized"] = df_sub[metric] / baseline
        df_sub["scaled_sc"] = df_sub[f"{metric}_normalized"] * (3 ** (N - 2))

        all_records.append(df_sub)

final_df = pd.concat(all_records, ignore_index=True)
save_path = os.path.join(output_dir, f"{metric}_normalized_and_scaled.csv")
final_df.to_csv(save_path, index=False)
print(f"\n All results saved to {save_path}")

group_cols = hyperparams
grouped = final_df.groupby(group_cols)[f"{metric}_normalized"].mean().reset_index()
best_global_row = grouped.loc[grouped[f"{metric}_normalized"].idxmin()]
best_global_params = best_global_row[hyperparams].to_dict()

print(f"\nBest global hyperparameter combination for {metric}:")
print(best_global_params)
