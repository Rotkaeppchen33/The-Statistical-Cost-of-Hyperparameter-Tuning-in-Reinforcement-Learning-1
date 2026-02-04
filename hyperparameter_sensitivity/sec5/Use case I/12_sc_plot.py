import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Algorithm name and metric to process
algorithm = "lambda_ac" # or "advn_norm_mean"
metric = "step_to_p90"  # or "step_to_p95"

# Load the full configuration file
root_dir = "/path/organized_by_alg_env"
input_path = os.path.join(root_dir, algorithm, f"sc_full_configs_{metric}_1.csv")
df = pd.read_csv(input_path)

# List of environments
envs = ["hopper", "halfcheetah", "walker2d", "ant", "swimmer"]
hyperparams = ["actorlr", "criticlr", "entcoef", "gaelambda"]

# List to collect results
plot_data = []
best_configs = []

# Process each environment
for env in envs:
    env_df = df[df["env"] == env]
    
    for n in range(5):  # N = 0,1,2,3,4
        subset = env_df[env_df["N"] == n]
        
        if subset.empty:
            print(f"Warning: No data for env {env} at N={n}")
            continue
        
        # Find the configuration with minimum normalized SC
        best_row = subset.loc[subset[f"{metric}_normalized"].idxmin()]
        
        # Get normalized SC and scale by 3^N
        scaled_sc = best_row[f"{metric}_normalized"] * (3 ** n)
        
        # Collect for plotting
        plot_data.append({
            "env": env,
            "N": n,
            "scaled_normalized_sc": scaled_sc
        })

        # Also save full best config
        config = {
            "env": env,
            "N": n,
            "scaled_normalized_sc": scaled_sc
        }
        for hp in hyperparams:
            config[hp] = best_row[hp]
        best_configs.append(config)

# Save best configurations before plotting
best_df = pd.DataFrame(best_configs)
best_config_path = os.path.join(root_dir, algorithm, f"best_config_per_env_per_n_{metric}.csv")
best_df.to_csv(best_config_path, index=False)
print(f"Saved best configuration table to {best_config_path}")

# --- Continue plotting as before ---
plot_df = pd.DataFrame(plot_data)

# Pivot table: rows are N, columns are envs
pivot = plot_df.pivot(index="N", columns="env", values="scaled_normalized_sc")

# Calculate mean across environments
pivot["mean"] = pivot[envs].mean(axis=1)

# Plotting
env_names = ["hopper", "halfcheetah", "walker2d", "ant", "swimmer"]
env_color_map = {env: f"C{i}" for i, env in enumerate(env_names)}
env_marker_map = {
    "hopper": "o",
    "halfcheetah": "s",
    "walker2d": "^",
    "ant": "D",
    "swimmer": "v",
}
plt.figure(figsize=(10, 6))
for i, env in enumerate(envs):
    plt.scatter(pivot.index, pivot[env], 
            color=env_color_map[env],
            marker=env_marker_map[env],
            s=80,
            alpha=0.7,
            label=env
        )
# markers = ['o', 's', 'D', '^', 'v', 'P', 'X', '*']
# for i, env in enumerate(envs):
    # plt.scatter(pivot.index, pivot[env], marker=markers[i % len(markers)], s = 80, alpha=0.7, label=env)

# Plot mean line
plt.plot(pivot.index, pivot["mean"], color="black", linestyle="--", linewidth=5, label="Mean")
# Labeling
plt.xlabel("Number of Tuned Hyperparameters", fontsize=20)
plt.ylabel(f"Normalized Effective SC", fontsize=20)
# plt.title(f"Normalized Sample Complexity vs N\n({algorithm}, {metric})", fontsize=18)
plt.xticks([0, 1, 2, 3, 4], fontsize=20)
plt.yticks(fontsize=20)
plt.yscale("log")
plt.ylim([0.08, 100])
plt.grid(True)
plt.legend(fontsize=17, loc="lower right")
plt.tight_layout()

# Save plot
output_plot = os.path.join(root_dir, algorithm, f"normalized_sc_vs_n_{metric}_1.png")
plt.savefig(output_plot, dpi=300)
plt.show()

print(f"Saved plot to {output_plot}")
