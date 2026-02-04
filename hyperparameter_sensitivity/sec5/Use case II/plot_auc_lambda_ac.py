import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the data
df = pd.read_csv("/path/normalized_multi_N_auc_lambda_ac_1.csv")

# Step 1: find the best hyperparameter configuration at N=0
baseline_best_row = df[df["N"] == 0].sort_values(by="normalized_auc", ascending=False).iloc[0]
baseline_best_params = baseline_best_row[["actorlr", "criticlr", "entcoef", "gaelambda"]].to_dict()

results = []

n_values = [0, 1, 2, 3, 4]

# List of environments
envs = df["env"].unique()

for n in n_values:
    for env in envs:
        env_df = df[(df["env"] == env) & (df["N"] == n)].copy()

        if n == 0:
            selected = env_df[
                (env_df["actorlr"] == baseline_best_params["actorlr"]) &
                (env_df["criticlr"] == baseline_best_params["criticlr"]) &
                (env_df["entcoef"] == baseline_best_params["entcoef"]) &
                (env_df["gaelambda"] == baseline_best_params["gaelambda"])
            ]
        elif n == 1:
            selected = env_df[
                (env_df["actorlr"] == baseline_best_params["actorlr"]) &
                (env_df["criticlr"] == baseline_best_params["criticlr"]) &
                (env_df["entcoef"] == baseline_best_params["entcoef"])
            ].sort_values(by="normalized_auc", ascending=False).head(1)
        elif n == 2:
            selected = env_df[
                (env_df["actorlr"] == baseline_best_params["actorlr"]) &
                (env_df["criticlr"] == baseline_best_params["criticlr"])
            ].sort_values(by="normalized_auc", ascending=False).head(1)
        elif n == 3:
            selected = env_df[
                (env_df["actorlr"] == baseline_best_params["actorlr"])
            ].sort_values(by="normalized_auc", ascending=False).head(1)
        elif n == 4:
            selected = env_df.sort_values(by="normalized_auc", ascending=False).head(1)

        if not selected.empty:
            row = selected.iloc[0]
            results.append({
                "N": n,
                "env": env,
                "actorlr": row["actorlr"],
                "criticlr": row["criticlr"],
                "entcoef": row["entcoef"],
                "gaelambda": row["gaelambda"],
                "normalized_auc": row["normalized_auc"]
            })


results_df = pd.DataFrame(results)

pivot_table = results_df.pivot(index="N", columns="env", values="normalized_auc")

pivot_table["mean"] = pivot_table.mean(axis=1)

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
# markers = ['o', 's', 'D', '^', 'v', 'P', 'X', '*']
for i, env in enumerate(envs):
    plt.scatter(pivot_table.index, pivot_table[env], 
            color=env_color_map[env],
            marker=env_marker_map[env],
            s=80,
            alpha=0.8,
            label=env
        )

plt.plot(pivot_table.index, pivot_table["mean"], color="black", linestyle="--", linewidth=5, label="Mean")
plt.xlabel("Number of Tuned Hyperparameters", fontsize=20)
plt.ylabel("Normalized Effective AUC", fontsize=20)
# plt.title("Normalized AUC vs N for Best Hyperparameter Configuration\n(lambda_ac)", fontsize=18)
plt.xticks([0, 1, 2, 3, 4], fontsize=20)
plt.yticks(fontsize=20)
plt.ylim([-0.9, 1.08])
plt.legend(fontsize=17)
plt.grid(True)
plt.tight_layout()

save_dir = "/path/lambda_ac"
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, "normalized_auc_vs_n_lambda_ac_2.png"), dpi=300)

# Save selected configurations
# results_df.to_csv(os.path.join(save_dir, "selected_hyperparams_and_auc.csv"), index=False)
