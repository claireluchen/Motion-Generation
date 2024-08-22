import numpy as np
import csv

def getFrameBoundaries(npz_files):
    frames = []
    idx = 0

    for file_path in npz_files:
        data = np.load(file_path)
        curNumFrames = np.shape(data['poses'])[0]
        frames.append((idx, idx + curNumFrames - 1))
        idx += curNumFrames
    
    print(frames)  # return [(0, 2750), (2751, 7096), (7097, 11606) ...]
    return frames

def normalize_column(values):
    min_val = min(values)
    max_val = max(values)
    if min_val == max_val:
        # If all values in the column are the same, return a list of zeros
        return [0] * len(values)
    return [2 * (x - min_val) / (max_val - min_val) - 1 for x in values]

def process_and_normalize_csv(csv_file, npz_files, output_csv):
    # Load CSV file
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        data = [list(map(float, row)) for row in reader]
    
    # Get frame boundaries
    frame_boundaries = getFrameBoundaries(npz_files)

    # Initialize a list to hold normalized data
    normalized_data = []

    # Process each motion segment
    for start, end in frame_boundaries:
        segment = data[start:end + 1]
        
        # Transpose the segment to work with columns
        segment_transposed = list(zip(*segment))
        
        # Normalize each column in the segment
        normalized_segment_transposed = [normalize_column(col) for col in segment_transposed]
        
        # Transpose back to original format
        normalized_segment = list(zip(*normalized_segment_transposed))
        
        # Append normalized segment to the normalized_data list
        normalized_data.extend(normalized_segment)
    
    # Save the normalized data to a new CSV file
    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(normalized_data)

# # Example usage:
# csv_file = 'D:/Claire/CMUsmplx/CMU/01/merged_poses.csv'  # Path to the input CSV file
# npz_files = [
#     'D:/Claire/CMUsmplx/CMU/01/01_01_stageii.npz',
#     'D:/Claire/CMUsmplx/CMU/01/01_02_stageii.npz',
#     'D:/Claire/CMUsmplx/CMU/01/01_03_stageii.npz',
#     'D:/Claire/CMUsmplx/CMU/01/01_05_stageii.npz',
#     'D:/Claire/CMUsmplx/CMU/01/01_06_stageii.npz',
#     'D:/Claire/CMUsmplx/CMU/01/01_07_stageii.npz',
#     'D:/Claire/CMUsmplx/CMU/01/01_08_stageii.npz',
#     'D:/Claire/CMUsmplx/CMU/01/01_09_stageii.npz',
#     'D:/Claire/CMUsmplx/CMU/01/01_10_stageii.npz',
#     'D:/Claire/CMUsmplx/CMU/01/01_11_stageii.npz',
# ]
# output_csv = 'D:/Claire/CMUsmplx/CMU/01/merged_poses_normalized.csv'  # Path to the output CSV file

# process_and_normalize_csv(csv_file, npz_files, output_csv)
