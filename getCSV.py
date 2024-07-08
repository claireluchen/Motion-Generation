#merge poses into one csv file to be used in dataloader
import numpy as np
import os

# Define the paths to the .npz files
npz_files = [
    'D:/Claire/CMUsmplx/CMU/01/01_01_stageii.npz',
    'D:/Claire/CMUsmplx/CMU/01/01_02_stageii.npz',
    'D:/Claire/CMUsmplx/CMU/01/01_03_stageii.npz',
    'D:/Claire/CMUsmplx/CMU/01/01_05_stageii.npz',
    'D:/Claire/CMUsmplx/CMU/01/01_06_stageii.npz',
    'D:/Claire/CMUsmplx/CMU/01/01_07_stageii.npz',
    'D:/Claire/CMUsmplx/CMU/01/01_08_stageii.npz',
    'D:/Claire/CMUsmplx/CMU/01/01_09_stageii.npz',
    'D:/Claire/CMUsmplx/CMU/01/01_10_stageii.npz',
    'D:/Claire/CMUsmplx/CMU/01/01_11_stageii.npz',

]

# Initialize an empty list to store the poses
all_poses = []

# Loop through each .npz file and extract the 'poses' array
for file_path in npz_files:
    data = np.load(file_path)
    poses = data['poses']
    print(np.shape(poses))
    all_poses.append(poses)

# Concatenate all the poses arrays
merged_poses = np.concatenate(all_poses, axis=0)

output_dir = 'D:/Claire/CMUsmplx/CMU/01/'

# Define the path to the output CSV file
csv_file_path = os.path.join(output_dir, 'merged_poses.csv')

# Save the merged poses array to the CSV file
np.savetxt(csv_file_path, merged_poses, delimiter=',')

print(f"Merged poses saved to {csv_file_path}")
