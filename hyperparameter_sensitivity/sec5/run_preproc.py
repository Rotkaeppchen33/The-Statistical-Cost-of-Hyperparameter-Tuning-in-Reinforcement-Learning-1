"""
Unified post-processing script for progressive hyperparameter release.

Semantics:
- Start from full grid M=3^4 (N=4 summaries).
- Identify global best config per algorithm (max avg Normalized AUC across envs).
- For a target N in {0,1,2,3,4}, keep (4-N) hyperparams fixed at global best
  and release N hyperparams according to a dynamic release order (can differ by algorithm).
- For the remaining configs, recompute "Normalized AUC" by reading reward CSVs,
  truncating length using min_T_env // (3^N), and multiplying back by (3^N).
- Save filtered+updated summaries to "N={N} hyperparam_summary_{env}_{alg}.csv".
"""

import os
import argparse
import pandas as pd


DEFAULT_VALID_ENVS = ["hopper", "swimmer", "halfcheetah", "walker2d", "ant"]
ALGORITHMS = ["lambda_ac", "advn_norm_mean"]
HP_COLS = ["actorlr", "criticlr", "entcoef", "gaelambda"]


RELEASE_ORDER = {
    "lambda_ac":       ["gaelambda", "entcoef",  "criticlr", "actorlr"],
    "advn_norm_mean":  ["gaelambda", "criticlr", "entcoef",  "actorlr"],
}


def read_quantiles(input_folder: str, env: str):
    qfile = os.path.join(input_folder, f"global_quantiles_{env}.csv")
    if not os.path.exists(qfile):
        return None
    dfq = pd.read_csv(qfile)
    if dfq.empty:
        return None
    return float(dfq.iloc[0]["p5"]), float(dfq.iloc[0]["p95"])


def calculate_new_auc_from_rewards_csv(rewards_file: str, p5: float, p95: float, N_value: int):
    """
    Rewards CSV format: each line is a comma-separated reward trajectory.
    We:
      - read all lines -> unique_rewards_list (not necessarily unique, but matches your script)
      - min_T_env = min(len(traj))
      - m = min_T_env // N_value
      - truncate each traj to length m
      - normalize each reward using (r-p5)/(p95-p5)
      - AUC = sum(normalized_rewards)
      - return mean(AUC over runs) * N_value
    """
    try:
        unique_rewards_list = []
        with open(rewards_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rewards = [float(x) for x in line.split(",") if x.strip()]
                    unique_rewards_list.append(rewards)
                except ValueError:
                    # malformed line; skip like your code
                    continue
    except Exception:
        return None

    if not unique_rewards_list:
        return None

    min_T_env = min(len(r) for r in unique_rewards_list)
    if min_T_env <= 0:
        return None

    if p95 == p5:
        return None

    m = int(min_T_env // N_value)
    if m <= 0:
        return None

    auc_values = []
    for rewards in unique_rewards_list:
        if len(rewards) < m:
            continue
        norm_rewards = [(r - p5) / (p95 - p5) for r in rewards[:m]]
        auc_values.append(sum(norm_rewards))

    if not auc_values:
        return None
    return (sum(auc_values) / len(auc_values)) * N_value


def find_global_best_hp(input_folder: str, envs, alg: str, base_summary_N: int):
    """
    Find hp tuple (actorlr, criticlr, entcoef, gaelambda) maximizing sum of Normalized AUC across envs.
    Reads: f"N={base_summary_N} hyperparam_summary_{env}_{alg}.csv"
    """
    auc_sum_per_hp = {}
    for env in envs:
        summary_file = os.path.join(input_folder, f"N={base_summary_N} hyperparam_summary_{env}_{alg}.csv")
        if not os.path.exists(summary_file):
            continue
        df = pd.read_csv(summary_file)
        if df.empty:
            continue

        # group by full hp tuple and sum Normalized AUC within that env
        grouped = df.groupby(HP_COLS).agg({"Normalized AUC": "sum"})
        for hp_tuple, auc in grouped["Normalized AUC"].items():
            auc_sum_per_hp[hp_tuple] = auc_sum_per_hp.get(hp_tuple, 0.0) + float(auc)

    if not auc_sum_per_hp:
        return None
    return max(auc_sum_per_hp, key=auc_sum_per_hp.get)


def fixed_hps_for_N(alg: str, N: int):
    """
    Return which HP columns should be fixed (NOT released) at target N.
    released = first N items in RELEASE_ORDER[alg]
    fixed = others
    """
    order = RELEASE_ORDER[alg]
    released = set(order[:N])
    fixed = [hp for hp in HP_COLS if hp not in released]
    return fixed, list(released)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=True, help="How many hyperparameters are released/tunable (0..4).")
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--rewards_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--base_summary_N", type=int, default=4, help="Which N summary to use to find global best (default 4).")
    parser.add_argument("--source_summary_N", type=int, default=4, help="Which N summary to load per-env before filtering (default 4).")
    parser.add_argument("--envs", type=str, default=",".join(DEFAULT_VALID_ENVS))
    args = parser.parse_args()

    N = args.N
    if N < 0 or N > 4:
        raise ValueError("N must be in {0,1,2,3,4}.")

    envs = [e.strip() for e in args.envs.split(",") if e.strip()]
    os.makedirs(args.output_folder, exist_ok=True)

    N_value = 3 ** N  # M = 3^N

    # 1) global best per algorithm
    best_hp_per_alg = {}
    for alg in ALGORITHMS:
        best_hp = find_global_best_hp(args.input_folder, envs, alg, args.base_summary_N)
        if best_hp is None:
            raise RuntimeError(f"Could not find global best hp for alg={alg}. Check summary CSVs.")
        best_hp_per_alg[alg] = {
            "actorlr": float(best_hp[0]),
            "criticlr": float(best_hp[1]),
            "entcoef": float(best_hp[2]),
            "gaelambda": float(best_hp[3]),
        }
        print(f"[Global best] {alg}: {best_hp_per_alg[alg]}")

    # 2) per env: filter configs + recompute AUC from reward files
    for env in envs:
        q = read_quantiles(args.input_folder, env)
        if q is None:
            print(f"Warning: quantiles missing for env={env}, skip.")
            continue
        p5, p95 = q

        for alg in ALGORITHMS:
            summary_file = os.path.join(args.input_folder, f"N={args.source_summary_N} hyperparam_summary_{env}_{alg}.csv")
            if not os.path.exists(summary_file):
                print(f"Warning: missing summary {summary_file}, skip.")
                continue

            df_env = pd.read_csv(summary_file)
            if df_env.empty:
                print(f"Warning: empty summary {summary_file}, skip.")
                continue

            fixed_cols, released_cols = fixed_hps_for_N(alg, N)
            best = best_hp_per_alg[alg]

            # filter: fixed hyperparams equal to global best
            mask = pd.Series([True] * len(df_env))
            for col in fixed_cols:
                # be robust to float formatting
                mask = mask & (df_env[col].astype(float) == float(best[col]))
            df_sel = df_env[mask].copy()

            if df_sel.empty:
                print(f"Warning: after filtering, no rows for env={env}, alg={alg}, N={N}.")
                continue

            # recompute normalized AUC from rewards csv
            updated_aucs = []
            for _, row in df_sel.iterrows():
                rewards_filename = (
                    f"actorlr_{row['actorlr']}_criticlr_{row['criticlr']}_entcoef_{row['entcoef']}_"
                    f"gaelambda_{row['gaelambda']}_env_{env}_alg_{alg}.csv"
                )
                rewards_path = os.path.join(args.rewards_folder, rewards_filename)
                if os.path.exists(rewards_path):
                    new_auc = calculate_new_auc_from_rewards_csv(rewards_path, p5, p95, N_value)
                else:
                    print(f"Warning: rewards file missing: {rewards_filename}")
                    new_auc = None
                updated_aucs.append(new_auc)

            df_sel["Normalized AUC"] = updated_aucs
            if "Trajectory Count" in df_sel.columns:
                df_sel["Trajectory Metric"] = df_sel["Trajectory Count"] * N_value

            out_csv = os.path.join(args.output_folder, f"N={N} hyperparam_summary_{env}_{alg}.csv")
            df_sel.to_csv(out_csv, index=False)

            print(
                f"Saved env={env}, alg={alg}, N={N} "
                f"(released={sorted(released_cols)}, fixed={fixed_cols}) -> {out_csv}"
            )


if __name__ == "__main__":
    main()