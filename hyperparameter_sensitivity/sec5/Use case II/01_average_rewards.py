import os
import csv
import glob
import numpy as np
import pandas as pd

# Define input/output folders
input_folder = "/path/"
output_folder = os.path.join(input_folder, "processed_avg_rewards")
os.makedirs(output_folder, exist_ok=True)

# Find all CSV files with hyperparameter info in the filename
reward_files = [f for f in glob.glob(os.path.join(input_folder, "*.csv")) if "actorlr" in os.path.basename(f)]

# Helper function: Read a CSV file with variable-length rows
def read_variable_length_csv(file_path):
    sequences = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            cleaned_row = [float(x) for x in row if x.strip() != ""]
            sequences.append(cleaned_row)
    return sequences

summary_records = []

# Process each reward file
for file_path in reward_files:
    reward_sequences = read_variable_length_csv(file_path)

    # Truncate all reward sequences to the shortest length
    min_len = min(len(seq) for seq in reward_sequences)
    truncated = np.array([seq[:min_len] for seq in reward_sequences])

    # Compute the average reward across runs
    avg_reward = np.mean(truncated, axis=0)

    # Save individual avg file (as a single row)
    filename = os.path.basename(file_path)
    base, ext = os.path.splitext(filename)
    output_file = os.path.join(output_folder, base + "_avg.csv")
    pd.DataFrame([avg_reward]).to_csv(output_file, index=False, header=False)  

    # Append to summary
#     summary_records.append([filename] + avg_reward.tolist())

# # Save summary file
# summary_df = pd.DataFrame(summary_records)
# summary_df.to_csv(os.path.join(output_folder, "all_avg_rewards_summary.csv"), index=False, header=False)
