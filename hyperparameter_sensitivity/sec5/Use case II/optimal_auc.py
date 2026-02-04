import os
import glob
import pandas as pd

# Root folder containing all algorithm subfolders
root_dir = "/path/processed_avg_rewards/organized_by_lambda"

# Process each algorithm
for alg_folder in glob.glob(os.path.join(root_dir, "*")):
    if not os.path.isdir(alg_folder):
        continue

    alg_name = os.path.basename(alg_folder)
    print(f"\nAggregating AUC across environments for: {alg_name}")

    auc_files = glob.glob(os.path.join(alg_folder, "*", "auc_*.csv"))

    auc_data = []
    for file_path in auc_files:
        df = pd.read_csv(file_path)
        auc_data.append(df)

    if not auc_data:
        print(f"No AUC files found for {alg_name}")
        continue

    # Concatenate all AUC records from different environments
    all_auc_df = pd.concat(auc_data, ignore_index=True)

    # Convert to numeric in case of string types
    for col in ["actorlr", "criticlr", "entcoef", "gaelambda"]:
        all_auc_df[col] = pd.to_numeric(all_auc_df[col], errors="coerce")

    # Group by hyperparameter combination and sum AUCs
    grouped_auc = (
        all_auc_df.groupby(["actorlr", "criticlr", "entcoef", "gaelambda"])["auc"]
        .sum()
        .reset_index()
        .rename(columns={"auc": "total_auc"})
    )

    # Save full aggregated table
    total_auc_path = os.path.join(alg_folder, "total_auc_across_envs.csv")
    grouped_auc.to_csv(total_auc_path, index=False)

    # Find and save best config
    best_config = grouped_auc.loc[grouped_auc["total_auc"].idxmax()]
    best_config_df = best_config.to_frame().T
    best_config_df.to_csv(os.path.join(alg_folder, f"best_hyperparams_{alg_name}_auc.csv"), index=False)

    print("Best config saved for", alg_name)
    print(best_config_df)
