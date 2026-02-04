import pandas as pd
import numpy as np
import os
from itertools import product

root_dir = "/postproc_results/processed_avg_rewards/organized_by_alg_env"
algorithm = "advn_norm_mean" #lambda_ac"
alg_root = os.path.join(root_dir, algorithm)
envs = ["hopper", "walker2d", "halfcheetah", "swimmer", "ant"]
hyperparams = ["actorlr", "criticlr", "entcoef", "gaelambda"]
metrics = ["step_to_p90", "step_to_p95"]

for metric in metrics:
    print(f"Processing {metric}...")
    env_data = dict()
    penalty_values = dict()
    baseline_values = dict()

    for env in envs:
        file_path = os.path.join(alg_root, env, f"step_to_quantiles_{env}.csv")
        df = pd.read_csv(file_path)

        # Calculate penalty value: 10x maximum valid step_to_X
        max_valid = df[metric].dropna().max()
        penalty_value = 10 * max_valid
        penalty_values[env] = penalty_value
        df[metric] = df[metric].fillna(penalty_value)
        # Normalize
        baseline = np.percentile(df[metric], 5)
        baseline_values[env] = baseline
        df[f"{metric}_normalized"] = df[metric] / baseline
        env_data[env] = df.copy()

    # Find the global best across environments
    full_configs = pd.concat(env_data.values(), ignore_index=True)
    group_cols = hyperparams
    grouped = full_configs.groupby(group_cols)[f"{metric}_normalized"].mean().reset_index()

    best_global_row = grouped.loc[grouped[f"{metric}_normalized"].idxmin()]
    best_global_params = best_global_row[hyperparams].to_dict()

    print(f"Best global hyperparameter combination for {metric}:")
    print(best_global_params)

    # Determine the releasing order
    released = []
    remaining = hyperparams.copy()
    releasing_order = []

    for step in range(4):
        min_avg_min_sc = np.inf
        next_hp = None
        for candidate in remaining:
            env_min_scs = []
            for env in envs:
                df = env_data[env]
                released_vals = [df[hp].unique() for hp in released]
                candidate_vals = df[candidate].unique()
                if released_vals:
                    combos = list(product(*released_vals, candidate_vals))
                else:
                    combos = [(val,) for val in candidate_vals]

                min_sc = np.inf

                for combo in combos:
                    condition = np.ones(len(df), dtype=bool)
                    for i, hp in enumerate(released):
                        condition &= (df[hp] == combo[i])
                    condition &= (df[candidate] == combo[-1])
                    for hp in hyperparams:
                        if hp not in released and hp != candidate:
                            condition &= (df[hp] == best_global_params[hp])

                    filtered = df[condition]

                    if not filtered.empty:
                        sc = filtered[f"{metric}_normalized"].mean()
                    else:
                        sc = penalty_values[env] / baseline_values[env]
                    if sc < min_sc:
                        min_sc = sc

                env_min_scs.append(min_sc)

            avg_min_sc = np.mean(env_min_scs)

            if avg_min_sc < min_avg_min_sc:
                min_avg_min_sc = avg_min_sc
                next_hp = candidate
                
        released.append(next_hp)
        remaining.remove(next_hp)
        releasing_order.append(next_hp)

    print(f"Releasing order for {metric}:")
    print(releasing_order)

    # full table for N=0,1,2,3,4
    all_records = []

    for env in envs:
        df = env_data[env]
        for n_free in range(0, 5):
            released = releasing_order[:n_free]
            fixed = [hp for hp in hyperparams if hp not in released]

            if released:
                released_vals = [df[hp].unique() for hp in released]
                combos = list(product(*released_vals))
            else:
                combos = [()]
            for combo in combos:
                condition = np.ones(len(df), dtype=bool)
                for i, hp in enumerate(released):
                    condition &= (df[hp] == combo[i])
                for hp in fixed:
                    condition &= (df[hp] == best_global_params[hp])

                filtered = df[condition]

                for _, row in filtered.iterrows():
                    record = {
                        "env": env,
                        "N": n_free,
                        f"{metric}_normalized": row[f"{metric}_normalized"],
                    }
                    for hp in hyperparams:
                        record[hp] = row[hp]
                    all_records.append(record)

    sc_full_df = pd.DataFrame(all_records)
    save_path = os.path.join(alg_root, f"sc_full_configs_{metric}_1.csv")
    sc_full_df.to_csv(save_path, index=False)
    print(f"Saved full SC configurations for {metric} to {save_path}\n")