import os
import glob
import pandas as pd
import numpy as np

# Root folder containing all algorithm folders
root_dir = "/path/organized_by_alg_env"

# Loop through each algorithm folder (lambda_ac_avg, advn_norm_mean_avg, etc.)
for alg_folder in glob.glob(os.path.join(root_dir, "*")):
    if not os.path.isdir(alg_folder):
        continue

    alg_name = os.path.basename(alg_folder)
    print(f"\nProcessing algorithm: {alg_name}")

    # Loop through each environment folder inside this algorithm
    for env_folder in glob.glob(os.path.join(alg_folder, "*")):
        if not os.path.isdir(env_folder):
            continue

        env = os.path.basename(env_folder)
        summary_file = os.path.join(env_folder, f"reward_summary_{env}.csv")
        if not os.path.exists(summary_file):
            print(f"Missing: {summary_file}")
            continue

        df = pd.read_csv(summary_file)

        # Extract reward step columns
        step_cols = [col for col in df.columns if col.startswith("step_")]
        
        min_T = df[step_cols].apply(lambda row: row.first_valid_index() is not None and row.count(), axis=1).min()
        all_rewards_truncated = df[step_cols].iloc[:, :min_T].values.flatten()
        all_rewards_truncated = all_rewards_truncated[~np.isnan(all_rewards_truncated)]
        # Compute quantiles from truncated rewards
        p5, p90, p95 = np.percentile(all_rewards_truncated, [5, 90, 95])
        
        # all_rewards = df[step_cols].values.flatten()
        # all_rewards = all_rewards[~np.isnan(all_rewards)]
        # p5, p90, p95 = np.percentile(all_rewards, [5, 90, 95])

        # Save per-env quantile file
        quantile_df = pd.DataFrame([{
            "env": env,
            "p5": p5,
            "p90": p90,
            "p95": p95
        }])
        quantile_df.to_csv(os.path.join(env_folder, f"reward_quantiles_{env}.csv"), index=False)

        # Compute step_to_p90 and step_to_p95 for each row
        step_records = []
        for _, row in df.iterrows():
            record = {
                "actorlr": row["actorlr"],
                "criticlr": row["criticlr"],
                "entcoef": row["entcoef"],
                "gaelambda": row["gaelambda"]
            }

            reward_seq = row[step_cols].values.astype(float)
            reward_seq = reward_seq[:min_T]

            record["step_to_p90"] = np.argmax(reward_seq >= p90) + 1 if np.any(reward_seq >= p90) else np.nan
            record["step_to_p95"] = np.argmax(reward_seq >= p95) + 1 if np.any(reward_seq >= p95) else np.nan

            step_records.append(record)

        step_df = pd.DataFrame(step_records)
        step_df.to_csv(os.path.join(env_folder, f"step_to_quantiles_{env}.csv"), index=False)

        print(f"Saved quantiles and step counts for {env} in {alg_name}")
