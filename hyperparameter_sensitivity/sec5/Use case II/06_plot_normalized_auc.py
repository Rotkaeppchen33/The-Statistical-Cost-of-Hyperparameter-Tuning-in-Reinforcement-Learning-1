import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

output_dir = "/path"
df_all = pd.read_csv(os.path.join(output_dir, "all_env_auc_normalized_1.csv"))
df_best = pd.read_csv(os.path.join(output_dir, "best_config_by_normalized_auc_1.csv"))

env_names = ["hopper", "halfcheetah", "walker2d", "ant", "swimmer"]
alg_order = [
    "GRPO",
    "vanilla PPO",
    "PPO+GAE"
]
x_pos = np.arange(len(alg_order))

env_color_map = {env: f"C{i}" for i, env in enumerate(env_names)}
env_marker_map = {
    "hopper": "o",
    "halfcheetah": "s",
    "walker2d": "^",
    "ant": "D",
    "swimmer": "v",
}

plt.figure(figsize=(10, 6))
mean_auc_vals = []
scatter_handles = {}

for i, alg in enumerate(alg_order):
    env_aucs = []

    for env in env_names:
        df_filtered = df_all[
            (df_all["Algorithm"] == alg) &
            (df_all["Environment"] == env)
        ]
        if df_filtered.empty:
            continue

        best_row = df_filtered.loc[df_filtered["Normalized AUC"].idxmax()]
        auc_val = best_row["Normalized AUC"]
        env_aucs.append(auc_val)

        label = env if env not in scatter_handles else None
        plt.scatter(
            i, auc_val,
            color=env_color_map[env],
            marker=env_marker_map[env],
            s=80,
            alpha=0.8,
            label=label
        )
        scatter_handles[env] = True

    mean_val = np.mean(env_aucs) if env_aucs else np.nan
    mean_auc_vals.append(mean_val)

plt.plot(x_pos, mean_auc_vals, color='black', marker='o', linestyle="--", linewidth=5, label='Mean across envs')
plt.xticks(x_pos, alg_order, fontsize=20)
plt.ylabel("Normalized Effective AUC", fontsize=20)
# plt.title("Normalized AUC of Different Algorithms", fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=16, loc="lower left")
plt.grid(True, linewidth=0.5, alpha=0.7)  
plt.tight_layout()

plot_path = os.path.join(output_dir, "normalized_auc_perenv.png")
plt.savefig(plot_path)
print(f"Saved plot to: {plot_path}")