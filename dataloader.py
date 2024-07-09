import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

def getFrameBoundaries(npz_files):
    frames = []
    idx = 0

    for file_path in npz_files:
        data = np.load(file_path)
        curNumFrames = np.shape(data['poses'])[0]
        frames.append((idx, idx + curNumFrames - 1))
        idx += curNumFrames
    
    return frames


class PoseDataset(Dataset):
    def __init__(self, csv_file_path, frame_boundaries):
        self.data = np.loadtxt(csv_file_path, delimiter=',')
        print("data shape is: " + str(np.shape(self.data)))
        self.frame_boundaries = frame_boundaries
        
        # number of samples, last valid idx == num_samples - 1
        self.num_samples = frame_boundaries[-1][1] + 1 - len(frame_boundaries)


        bounds = set()
        for start, end in self.frame_boundaries:
            bounds.add(end)

        # csv[i] maps idx given by user to row number of the csv data
        self.csv = {}
        idxCounter = 0
        rowCounter = 0
        lastIdx = frame_boundaries[-1][1]- len(frame_boundaries)
        lastRow = frame_boundaries[-1][1]
        while idxCounter <= lastIdx and rowCounter <= lastRow:
            if rowCounter in bounds:
                rowCounter += 1
            else:
                self.csv[idxCounter] = rowCounter
                idxCounter += 1
                rowCounter += 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx >= len(self.csv):
            raise IndexError("Index out of range of dataset boundaries")
        i = self.csv[idx];
        input_frame = self.data[i];
        output_frame = self.data[i + 1];

        input_frame = torch.tensor(input_frame, dtype=torch.float32)
        output_frame = torch.tensor(output_frame, dtype=torch.float32)
                
        return input_frame, output_frame, idx

# path to the CSV file
csv_file_path = 'D:/Claire/CMUsmplx/CMU/01/merged_poses.csv'

# frame boundaries for each file, inclusive
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
frame_boundaries = getFrameBoundaries(npz_files) #[(0, 2750), (2751, 7096), (7097, 11606) ...]

# dataset and dataloader
pose_dataset = PoseDataset(csv_file_path, frame_boundaries)
pose_dataloader = DataLoader(pose_dataset, batch_size=32, shuffle=False)

# Example usage: iterate over the dataloader
for input_frame, output_frame,idx in pose_dataloader:
    # pass input_frame to your neural network and use output_frame for the target
    print("Input frame:", input_frame[0,:])
    print("Output frame:", output_frame[0,:])
    print(idx)
    break
