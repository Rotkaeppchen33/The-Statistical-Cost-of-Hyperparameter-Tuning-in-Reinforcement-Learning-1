import numpy as np
import pandas as pd
import os

folder_path = "/path/tuned_rewards/"

npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

def extract_unique_rewards(data):
    num_experiments, num_steps, num_substeps, num_envs = data.shape
    all_unique_rewards = []

    for exp in range(num_experiments):  
        unique_rewards = set()
        for step in range(num_steps):
            for substep in range(num_substeps):  
                for env in range(num_envs):
                    reward = data[exp, step, substep, env]  
                    if reward != 0:  
                        unique_rewards.add(reward)
        all_unique_rewards.append(sorted(unique_rewards))  

    return all_unique_rewards  

output_folder = "/path/csv"
os.makedirs(output_folder, exist_ok=True)

for npy_file in npy_files:
    file_path = os.path.join(folder_path, npy_file)

    if os.path.getsize(file_path) < 1024:  
        print(f"Skipping empty or corrupted file: {npy_file}")
        continue

    data = np.load(file_path, allow_pickle=True)
    
    mean_value = np.mean(data)
    print(f"{npy_file} - all_mean:", mean_value)

    unique_rewards_list = extract_unique_rewards(data)
    # print(f"{npy_file} - unique_mean:", np.mean(unique_rewards_list))

    csv_filename = os.path.splitext(npy_file)[0] + ".csv"
    csv_path = os.path.join(output_folder, csv_filename)

    with open(csv_path, "w") as f:
        for rewards in unique_rewards_list:
            f.write(",".join(map(str, rewards)) + "\n")

    print(f"Saved {csv_filename}")
