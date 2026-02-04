import os
import shutil
import glob

# Extract env and alg from filename
def extract_env_alg_from_filename(filename):
    name = filename.replace(".csv", "")
    parts = name.split("_")
    env = None
    alg = None
    i = 0
    while i < len(parts):
        if parts[i] == "env":
            env = parts[i + 1]
            i += 2
        elif parts[i] == "alg":
            alg = "_".join(parts[i + 1:])
            break
        else:
            i += 1
    return env, alg

# Set source folder and output root
source_folder = "/postproc_results/processed_avg_rewards"
output_root = os.path.join(source_folder, "organized_by_alg_env")

# Collect all *_avg.csv files
all_files = glob.glob(os.path.join(source_folder, "*_avg.csv"))

# Process and copy files
for file_path in all_files:
    filename = os.path.basename(file_path)
    env, alg = extract_env_alg_from_filename(filename)

    # Define destination directory
    dest_dir = os.path.join(output_root, alg, env)
    os.makedirs(dest_dir, exist_ok=True)

    # Copy file to new location
    dest_path = os.path.join(dest_dir, filename)
    shutil.copy2(file_path, dest_path)

print("Files have been successfully organized by algorithm and environment.")
