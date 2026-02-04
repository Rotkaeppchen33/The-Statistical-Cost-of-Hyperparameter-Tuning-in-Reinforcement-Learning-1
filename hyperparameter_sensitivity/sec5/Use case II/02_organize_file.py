import os
import shutil
import glob
import re

def extract_env_lambda_from_filename(filename):
    name = filename.replace(".csv", "")
    parts = name.split("_")
    env = None
    gaelambda = None
    i = 0
    while i < len(parts):
        if parts[i] == "env":
            env = parts[i + 1]
            i += 2
        elif parts[i] == "gaelambda":
            gaelambda = parts[i + 1]
            i += 2
        else:
            i += 1

    return env, gaelambda

# input and output paths
source_folder = "/path/processed_avg_rewards"
output_root = os.path.join(source_folder, "organized_by_lambda")

#  *_avg.csv files
all_files = glob.glob(os.path.join(source_folder, "*_avg.csv"))

# Process files
for file_path in all_files:
    filename = os.path.basename(file_path)
    env, gaelambda = extract_env_lambda_from_filename(filename)

    if env is None or gaelambda is None:
        print(f"Skipped file (missing env or gaelambda): {filename}")
        continue

    dest_dir = os.path.join(output_root, f"gaelambda_{gaelambda}", env)
    os.makedirs(dest_dir, exist_ok=True)

    dest_path = os.path.join(dest_dir, filename)
    shutil.copy2(file_path, dest_path)

print("Files have been successfully organized by gaelambda and environment.")
