import re
import numpy as np
import pandas as pd
import os

valid_envs = {"hopper", "swimmer", "halfcheetah", "walker2d", "ant"}

input_folder = "/path/rewards"
# npy_files = [f for f in os.listdir(input_folder) if f.endswith('.npy')]
output_folder = "/path/post"
os.makedirs(output_folder, exist_ok=True)

N_value = 3**4

def extract_unique_rewards(data):
    num_experiments, num_steps, num_substeps, num_envs = data.shape
    all_unique_rewards = []
    for exp in range(num_experiments):
        rewards_list = []
        seen = set()
        for step in range(num_steps):
            for substep in range(num_substeps):
                for env in range(num_envs):
                    reward = data[exp, step, substep, env]
                    if reward != 0 and reward not in seen:
                        rewards_list.append(reward)
                        seen.add(reward)
        all_unique_rewards.append(rewards_list)
    return all_unique_rewards

def parse_filename(filename):
    name = os.path.splitext(filename)[0]
    # hp_pairs = re.findall(r'([a-zA-Z]+)_([\d\.]+)', name)
    hp_pairs = re.findall(r'([a-zA-Z]+)_([\deE\+\-\.]+)', name)

    env_match = re.search(r"env_([^_]+)", name)
    alg_match = re.search(r"alg_([^\.]+)", name)

    hp_dict = {}
    for key, val in hp_pairs:
        if key not in {"env", "alg", "norm"}:
            hp_dict[key] = val
    if env_match:
        hp_dict["env"] = env_match.group(1)
    if alg_match:
        hp_dict["alg"] = alg_match.group(1)
    
    return hp_dict

files_by_env = {env: [] for env in valid_envs}
for f in os.listdir(input_folder):
    if f.endswith('.npy'):
        m = re.search(r"env_([^_]+)", f)
        if m:
            env = m.group(1)
            if env in valid_envs:
                files_by_env[env].append(f)
            else:
                print(f"File {f} has env {env} which is not in valid_envs, skipping")
        else:
            print(f"File {f} does not contain env info, skipping")

for env in valid_envs:
    env_files = files_by_env.get(env, [])
    if not env_files:
        print(f"No files found for env {env}.")
        continue
    print(f"Processing environment: {env}")
    
    global_rewards = []
    global_lengths = []
    for npy_file in env_files:
        file_path = os.path.join(input_folder, npy_file)
        try:
            data = np.load(file_path, allow_pickle=True)
        except Exception as e:
            print(f"Error loading {npy_file}: {e}")
            continue

        unique_rewards_list = extract_unique_rewards(data)
        
        for rewards in unique_rewards_list:
            global_rewards.extend(rewards)
            if len(rewards) > 0:
                global_lengths.append(len(rewards))

    if global_lengths:
        min_T_env = min(global_lengths)
    else:
        min_T_env = 0
    print(f"For env {env}, global minimum T across all files = {min_T_env}")

    if not global_rewards:
        print(f"No rewards for env {env}")
        continue
    p5, p95 = np.percentile(global_rewards, [5, 95])
    print(f"For env {env}, Global p5: {p5}, p95: {p95}")

    quantile_csv_path = os.path.join(output_folder, f"global_quantiles_{env}.csv")
    pd.DataFrame([{"p5": p5, "p95": p95}]).to_csv(quantile_csv_path, index=False)
    print(f"Saved global quantiles for env {env} to {quantile_csv_path}")

    summary_results = []  
    summary_results_by_alg = {}
    for npy_file in env_files:
        file_path = os.path.join(input_folder, npy_file)
        if os.path.getsize(file_path) < 1024:
            continue
        try:
            data = np.load(file_path, allow_pickle=True)
        except Exception as e:
            print(f"Error loading {npy_file}: {e}")
            continue

        unique_rewards_list = extract_unique_rewards(data)

        experiment_traj_counts = []
        for rewards in unique_rewards_list:
            traj_count = None
            for idx, r in enumerate(rewards):
                if r >= p95:
                    traj_count = idx + 1
                    break
            if traj_count is not None:
                experiment_traj_counts.append(traj_count)
        if not experiment_traj_counts:
            print(f"{npy_file}: did not reach p95 in any experiment.")
            best_traj_count = np.nan
            traj_metric = np.nan
            # continue
        elif len(experiment_traj_counts) < 5:
            print(f"{npy_file}: did not reach p95 in any experiment.")
            best_traj_count = np.nan
            traj_metric = np.nan
            # continue
        else:
            experiment_traj_counts.sort()
            best_traj_count = experiment_traj_counts[4] 
            traj_metric = N_value * best_traj_count
        
        auc_values = []
        if min_T_env > 0:
            m = min_T_env // N_value  
            if m > 0:
                for rewards in unique_rewards_list:
                    norm_rewards = [(r - p5) / (p95 - p5) for r in rewards[:m]]
                    # print(rewards[:m])
                    auc_values.append(sum(norm_rewards))
        if auc_values:
            normalized_auc = (sum(auc_values) / len(auc_values)) * N_value
        else:
            normalized_auc = None
        
        hp_dict = parse_filename(npy_file)     
        hp_dict["Trajectory Count"] = best_traj_count
        hp_dict["Trajectory Metric"] = traj_metric
        hp_dict["Normalized AUC"] = normalized_auc
         
        # hp_str = ", ".join(f"{k}={v}" for k, v in hp_dict.items())
        alg_val = hp_dict.get("alg", "unknown")
        if alg_val not in summary_results_by_alg:
            summary_results_by_alg[alg_val] = []
        summary_results_by_alg[alg_val].append(hp_dict)
        
    for alg, results in summary_results_by_alg.items():
        if results:
            df = pd.DataFrame(results)
            desired_cols = [
                "actorlr", "criticlr", "entcoef", "gaelambda", 
                "env", "alg",
                "Trajectory Count", "Trajectory Metric", "Normalized AUC"
            ]
            final_cols = [c for c in desired_cols if c in df.columns]
            df = df[final_cols]

            summary_csv_path = os.path.join(output_folder, f"N=4 hyperparam_summary_{env}_{alg}.csv")
            df.to_csv(summary_csv_path, index=False)
            print(f"Saved hyper-parameter summary CSV for env {env}, alg {alg} to {summary_csv_path}")
        else:
            print(f"No hyper-parameter set reached p95 for env {env} and alg {alg}.")