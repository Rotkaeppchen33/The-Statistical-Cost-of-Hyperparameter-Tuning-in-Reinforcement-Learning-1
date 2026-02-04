import os
import re
import glob
import numpy as np
import pandas as pd

algorithms = ["lambda_ac", "advn_norm_mean"]
valid_envs = {"hopper", "swimmer", "halfcheetah", "walker2d", "ant"}
input_folder = "/postproc_results"  
rewards_folder = "/postproc_results"  
output_folder = "/postproc_results"  
os.makedirs(output_folder, exist_ok=True)

m_new = 3**0  # can be changed to different m values 
best_hp_auc_per_alg = {
    "lambda_ac": (0.0001, 0.0001, 0.01, 0.5),
    "advn_norm_mean": (0.0001, 0.001, 0.001, 0.3)
}

best_hp_traj_per_alg = {
    "lambda_ac": (0.0001, 0.001, 0.001, 0.9),
    "advn_norm_mean": (3e-05, 1e-05, 0.001, 0.7)
}

best_actor_lr_lambda_auc = best_hp_auc_per_alg["lambda_ac"][0]
best_critic_lr_lambda_auc = best_hp_auc_per_alg["lambda_ac"][1]
best_entcoef_lambda_auc    = best_hp_auc_per_alg["lambda_ac"][2]
best_gaelambda_lambda_auc  = best_hp_auc_per_alg["lambda_ac"][3]

best_actor_lr_advn_auc = best_hp_auc_per_alg["advn_norm_mean"][0]
best_critic_lr_advn_auc = best_hp_auc_per_alg["advn_norm_mean"][1]
best_entcoef_advn_auc    = best_hp_auc_per_alg["advn_norm_mean"][2]
best_gaelambda_advn_auc  = best_hp_auc_per_alg["advn_norm_mean"][3]

best_actor_lr_lambda_traj = best_hp_traj_per_alg["lambda_ac"][0]
best_critic_lr_lambda_traj = best_hp_traj_per_alg["lambda_ac"][1]
best_entcoef_lambda_traj    = best_hp_traj_per_alg["lambda_ac"][2]
best_gaelambda_lambda_traj  = best_hp_traj_per_alg["lambda_ac"][3]

best_actor_lr_advn_traj = best_hp_traj_per_alg["advn_norm_mean"][0]
best_critic_lr_advn_traj = best_hp_traj_per_alg["advn_norm_mean"][1]
best_entcoef_advn_traj    = best_hp_traj_per_alg["advn_norm_mean"][2]
best_gaelambda_advn_traj  = best_hp_traj_per_alg["advn_norm_mean"][3]

def calculate_new_auc(rewards_file, p5, p95, N_value):
    try:
        with open(rewards_file, 'r') as file:
            unique_rewards_list = []
            for line in file:
                try:
                    rewards = [float(x) for x in line.strip().split(',') if x.strip()]
                    unique_rewards_list.append(rewards)
                except ValueError:
                    print(f"Skipping malformed line in {rewards_file}: {line.strip()}")
    except Exception as e:
        print(f"Error reading {rewards_file}: {e}")
        return None
    if not unique_rewards_list:
        return None

    min_T_env = min(len(rewards) for rewards in unique_rewards_list)
    auc_values = []
    if min_T_env > 0:
        m = int(min_T_env // N_value)
        if m > 0:
            for rewards in unique_rewards_list:
                if len(rewards) < m:
                    continue
                norm_rewards = [(r - p5) / (p95 - p5) for r in rewards[:m] if p95 != p5]
                auc_values.append(sum(norm_rewards))
    return (sum(auc_values) / len(auc_values)) * N_value if auc_values else None

quantile_summary_file = os.path.join(input_folder, "quantile5_95_summary_by_env.csv")
if os.path.exists(quantile_summary_file):
    quantile_summary_df = pd.read_csv(quantile_summary_file)
else:
    print(f"Warning: {quantile_summary_file} not found, cannot perform re-normalization.")
    quantile_summary_df = None

# ------------------- AUC Branch -------------------
for env in valid_envs:
    # Read global quantiles for reward normalization.
    quantile_file = os.path.join(input_folder, f"global_quantiles_{env}.csv")
    if os.path.exists(quantile_file):
        quantile_df = pd.read_csv(quantile_file)
        p5, p95 = quantile_df.iloc[0]['p5'], quantile_df.iloc[0]['p95']
    else:
        print(f"Warning: No quantile file found for {env}, skipping...")
        continue

    for alg in ["lambda_ac", "advn_norm_mean"]:
        summary_file = os.path.join(input_folder, f"N=4 hyperparam_summary_{env}_{alg}.csv")
        if not os.path.exists(summary_file):
            continue
        df_env = pd.read_csv(summary_file)
        
        # Select rows for the AUC branch based on fixed best hyperparameters.
        if alg == "lambda_ac":
            df_env_auc = df_env[(df_env['actorlr'] == best_actor_lr_lambda_auc) &
                                (df_env['criticlr'] == best_critic_lr_lambda_auc) &
                                (df_env['entcoef'] == best_entcoef_lambda_auc) &
                                (df_env['gaelambda'] == best_gaelambda_lambda_auc)].copy()
        else:  # advn_norm_mean
            df_env_auc = df_env[(df_env['actorlr'] == best_actor_lr_advn_auc) &
                                (df_env['criticlr'] == best_critic_lr_advn_auc) &
                                (df_env['entcoef'] == best_entcoef_advn_auc) &
                                (df_env['gaelambda'] == best_gaelambda_advn_auc)].copy()
        if df_env_auc.empty:
            print(f"Warning: AUC branch - No matching hyperparameter combination for {env}, {alg}, skipping...")
        else:
            updated_aucs = []
            for _, row in df_env_auc.iterrows():
                rewards_filename = (f"actorlr_{row['actorlr']}_criticlr_{row['criticlr']}_"
                                    f"entcoef_{row['entcoef']}_gaelambda_{row['gaelambda']}_"
                                    f"env_{env}_alg_{alg}.csv")
                rewards_filepath = os.path.join(rewards_folder, rewards_filename)
                if os.path.exists(rewards_filepath):
                    new_auc = calculate_new_auc(rewards_filepath, p5, p95, m_new)
                else:
                    print(f"Warning: Rewards file {rewards_filename} not found (AUC branch), setting AUC to NaN.")
                    new_auc = None
                updated_aucs.append(new_auc)
            df_env_auc['Normalized AUC'] = updated_aucs
            df_env_auc['Trajectory Metric'] = df_env_auc['Trajectory Count'] * m_new

            # Second normalization using quantile5_95_summary_by_env.csv.
            if quantile_summary_df is not None:
                env_quantiles = quantile_summary_df[quantile_summary_df['env'] == env]
                if not env_quantiles.empty:
                    auc_q5 = env_quantiles.iloc[0]['Normalized AUC q5']
                    auc_q95 = env_quantiles.iloc[0]['Normalized AUC q95']
                    traj_q5 = env_quantiles.iloc[0]['Trajectory Metric q5']
                    traj_q95 = env_quantiles.iloc[0]['Trajectory Metric q95']
                    if (auc_q95 - auc_q5) != 0:
                        df_env_auc['Normalized AUC'] = (df_env_auc['Normalized AUC'] - auc_q5) / (auc_q95 - auc_q5)
                    else:
                        print(f"Warning: For env {env}, AUC quantiles are equal; setting normalized AUC to NaN.")
                        df_env_auc['Normalized AUC'] = None
                    if (traj_q95 - traj_q5) != 0:
                        df_env_auc['Trajectory Metric'] = (df_env_auc['Trajectory Metric'] - traj_q5) / (traj_q95 - traj_q5)
                    else:
                        print(f"Warning: For env {env}, Trajectory Metric quantiles are equal; setting normalized Trajectory Metric to NaN.")
                        df_env_auc['Trajectory Metric'] = None
                else:
                    print(f"Warning: No quantile summary found for env {env} in quantile5_95_summary_by_env.csv, skipping re-normalization.")
            else:
                print("Warning: No quantile summary available, skipping re-normalization.")

            output_csv_auc = os.path.join(output_folder, f"N=0 hyperparam_summary_{env}_{alg}_auc_norm.csv")
            df_env_auc.to_csv(output_csv_auc, index=False)
            print(f"Saved updated AUC branch hyper-parameter summary for {env}, {alg} to {output_csv_auc}")

# ------------------- Trajectory Branch -------------------
for env in valid_envs:
    # Read global quantiles for reward normalization.
    quantile_file = os.path.join(input_folder, f"global_quantiles_{env}.csv")
    if os.path.exists(quantile_file):
        quantile_df = pd.read_csv(quantile_file)
        p5, p95 = quantile_df.iloc[0]['p5'], quantile_df.iloc[0]['p95']
    else:
        print(f"Warning: No quantile file found for {env}, skipping Trajectory branch...")
        continue

    for alg in ["lambda_ac", "advn_norm_mean"]:
        summary_file = os.path.join(input_folder, f"N=4 hyperparam_summary_{env}_{alg}.csv")
        if not os.path.exists(summary_file):
            continue
        df_env = pd.read_csv(summary_file)
        
        # Select rows for the Trajectory branch based on fixed best hyperparameters.
        if alg == "lambda_ac":
            df_env_traj = df_env[(df_env['actorlr'] == best_actor_lr_lambda_traj) &
                                 (df_env['criticlr'] == best_critic_lr_lambda_traj) &
                                 (df_env['entcoef'] == best_entcoef_lambda_traj) &
                                 (df_env['gaelambda'] == best_gaelambda_lambda_traj)].copy()
        else:  # advn_norm_mean
            df_env_traj = df_env[(df_env['actorlr'] == best_actor_lr_advn_traj) &
                                 (df_env['criticlr'] == best_critic_lr_advn_traj) &
                                 (df_env['entcoef'] == best_entcoef_advn_traj) &
                                 (df_env['gaelambda'] == best_gaelambda_advn_traj)].copy()
        if df_env_traj.empty:
            print(f"Warning: Trajectory branch - No matching hyperparameter combination for {env}, {alg}, skipping...")
        else:
            updated_aucs = []
            for _, row in df_env_traj.iterrows():
                rewards_filename = (f"actorlr_{row['actorlr']}_criticlr_{row['criticlr']}_"
                                      f"entcoef_{row['entcoef']}_gaelambda_{row['gaelambda']}_"
                                      f"env_{env}_alg_{alg}.csv")
                rewards_filepath = os.path.join(rewards_folder, rewards_filename)
                if os.path.exists(rewards_filepath):
                    new_auc = calculate_new_auc(rewards_filepath, p5, p95, m_new)
                else:
                    print(f"Warning: Rewards file {rewards_filename} not found (Trajectory branch), setting AUC to NaN.")
                    new_auc = None
                updated_aucs.append(new_auc)
            df_env_traj['Normalized AUC'] = updated_aucs
            df_env_traj['Trajectory Metric'] = df_env_traj['Trajectory Count'] * m_new

            # Second normalization using quantile5_95_summary_by_env.csv.
            if quantile_summary_df is not None:
                env_quantiles = quantile_summary_df[quantile_summary_df['env'] == env]
                if not env_quantiles.empty:
                    auc_q5 = env_quantiles.iloc[0]['Normalized AUC q5']
                    auc_q95 = env_quantiles.iloc[0]['Normalized AUC q95']
                    traj_q5 = env_quantiles.iloc[0]['Trajectory Metric q5']
                    traj_q95 = env_quantiles.iloc[0]['Trajectory Metric q95']
                    if (auc_q95 - auc_q5) != 0:
                        df_env_traj['Normalized AUC'] = (df_env_traj['Normalized AUC'] - auc_q5) / (auc_q95 - auc_q5)
                    else:
                        print(f"Warning: For env {env}, AUC quantiles are equal; setting normalized AUC to NaN.")
                        df_env_traj['Normalized AUC'] = None
                    if (traj_q95 - traj_q5) != 0:
                        df_env_traj['Trajectory Metric'] = (df_env_traj['Trajectory Metric'] - traj_q5) / (traj_q95 - traj_q5)
                    else:
                        print(f"Warning: For env {env}, Trajectory Metric quantiles are equal; setting normalized Trajectory Metric to NaN.")
                        df_env_traj['Trajectory Metric'] = None
                else:
                    print(f"Warning: No quantile summary found for env {env} in quantile5_95_summary_by_env.csv, skipping re-normalization.")
            else:
                print("Warning: No quantile summary available, skipping re-normalization.")

            output_csv_traj = os.path.join(output_folder, f"m=0 hyperparam_summary_{env}_{alg}_traj_norm.csv")
            df_env_traj.to_csv(output_csv_traj, index=False)
            print(f"Saved updated Trajectory branch hyper-parameter summary for {env}, {alg} to {output_csv_traj}")