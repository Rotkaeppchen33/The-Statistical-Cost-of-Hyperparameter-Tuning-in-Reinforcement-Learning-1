import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

root_dir = "/path/"
metric = "step_to_p90"
input_path = os.path.join(root_dir, f"{metric}_normalized_and_scaled.csv")
df = pd.read_csv(input_path)
envs = ["hopper", "halfcheetah", "walker2d", "ant", "swimmer"]
hyperparams = ["actorlr", "criticlr", "entcoef", "gaelambda"]
algorithms = ["GRPO", "vanilla PPO", "PPO+GAE"]
INF_LINE_Y = 1e2

best_configs = []
plot_data = []

for alg in algorithms:
    alg_df = df[df["algorithm"] == alg]

    if alg_df.empty:
        print(f"Warning: No data for algorithm {alg}")
        continue
        
    #### penalty
    for env in envs:
        sub_df = alg_df[alg_df["env"] == env]
        if sub_df.empty:
            continue
        max_valid = sub_df["scaled_sc"].dropna().max()
        penalty = 10 * max_valid
        alg_df.loc[alg_df["env"] == env, "scaled_sc"] = sub_df["scaled_sc"].fillna(penalty)

    grouped = alg_df.groupby(hyperparams)
    best_total = np.inf
    best_group = None

    for hp_values, group in grouped:
        if set(envs).issubset(set(group["env"])):  
            total_sc = group["scaled_sc"].sum()
            if total_sc < best_total:
                best_total = total_sc
                best_group = group

    if best_group is None:
        print(f"Warning: No valid full-environment config for {alg}")
        continue

    for _, row in best_group.iterrows():
        plot_data.append({
            "env": row["env"],
            "algorithm": alg,
            "scaled_normalized_sc": row["scaled_sc"]
        })
        config = {
            "env": row["env"],
            "algorithm": alg,
            "scaled_normalized_sc": row["scaled_sc"]
        }
        for hp in hyperparams:
            config[hp] = row[hp]
        best_configs.append(config)

best_df = pd.DataFrame(best_configs)
best_config_path = os.path.join(root_dir, f"best_cross_env_config_per_algorithm_{metric}.csv")
best_df.to_csv(best_config_path, index=False)
print(f" Saved best configuration table to {best_config_path}")

plot_df = pd.DataFrame(plot_data)
# pivot = plot_df.pivot(index="algorithm", columns="env", values="scaled_normalized_sc")
# pivot["mean"] = pivot[envs].mean(axis=1)
plot_df["is_infinite"] = False

for env in envs:
    for alg in algorithms:
        mask = (plot_df["algorithm"] == alg) & (plot_df["env"] == env)
        if not mask.any():
            plot_df = pd.concat([
                plot_df,
                pd.DataFrame([{
                    "env": env,
                    "algorithm": alg,
                    "scaled_normalized_sc": INF_LINE_Y,
                    "is_infinite": True
                }])
            ], ignore_index=True)
        elif plot_df.loc[mask, "scaled_normalized_sc"].isnull().all():
            plot_df.loc[mask, "scaled_normalized_sc"] = INF_LINE_Y
            plot_df.loc[mask, "is_infinite"] = True

pivot = plot_df.pivot(index="algorithm", columns="env", values="scaled_normalized_sc")

def safe_mean(row):
    if (row[envs] == INF_LINE_Y).any():
        return INF_LINE_Y
    return row[envs].mean()

pivot["mean"] = pivot.apply(safe_mean, axis=1)
pivot = pivot.reindex(["GRPO", "vanilla PPO", "PPO+GAE"])
plt.figure(figsize=(10, 6))
env_names = ["hopper", "halfcheetah", "walker2d", "ant", "swimmer"]
env_color_map = {env: f"C{i}" for i, env in enumerate(env_names)}
env_marker_map = {
    "hopper": "o",
    "halfcheetah": "s",
    "walker2d": "^",
    "ant": "D",
    "swimmer": "v",
}
for i, env in enumerate(envs):
    for alg in pivot.index:
        val = pivot.loc[alg, env]
        is_inf = val == INF_LINE_Y
        plt.scatter(
            alg, val,
            color=env_color_map[env],
            marker=env_marker_map[env],
            s=81, alpha=0.8,
            facecolors='none' if is_inf else f"C{i}",
            edgecolors='gray' if is_inf else f"C{i}",
            linewidths=1.5,
            label=env if alg == pivot.index[0] else ""
        )
# for i, env in enumerate(envs):
#     plt.scatter(pivot.index, pivot[env], marker=markers[i % len(markers)], s = 81, label=env)
plt.plot(pivot.index, pivot["mean"], color="black", linestyle="--", linewidth=5, label="Mean")

plt.axhline(INF_LINE_Y, color="red", linestyle=":", linewidth=2)
plt.text(-0.3, INF_LINE_Y * 1.05, "Infinity", color="red", fontsize=12)

# plt.xlabel("Algorithm", fontsize=20)
plt.ylabel(f"Normalized Effective SC", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.yscale("log")
plt.grid(True, linewidth=0.5, alpha=0.7)  
plt.legend(fontsize=15, loc="lower left")
plt.tight_layout()

output_plot = os.path.join(root_dir, f"normalized_sc_vs_alg_per_env_best.png")
plt.savefig(output_plot, dpi=300)
plt.show()
print(f" Saved plot to {output_plot}")