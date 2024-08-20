import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

#input_frame = n consecutive frames
#output_frame = subsequent target frame (pi+1)
#output_frame_2 = next frame following the output_frame (pi+2)

def getFrameBoundaries(npz_files):
    frames = []
    idx = 0

    for file_path in npz_files:
        data = np.load(file_path)
        curNumFrames = np.shape(data['poses'])[0]
        frames.append((idx, idx + curNumFrames - 1))
        idx += curNumFrames
    
    # print(frames) # return [(0, 2750), (2751, 7096), (7097, 11606) ...]
    return frames

class PoseDataset(Dataset):
    def __init__(self, csv_file_path, frame_boundaries, n):
        self.data = np.loadtxt(csv_file_path, delimiter=',')
        # print("data shape is: " + str(np.shape(self.data)))
        self.frame_boundaries = frame_boundaries
        #input is n consecutive frames
        self.n = n 
        
        # number of samples, last valid idx == num_samples - 1
        self.num_samples = frame_boundaries[-1][1] + 1 - 2 * len(frame_boundaries) * n

        bounds = set()
        for start, end in self.frame_boundaries:
            for i in range(n):
                bounds.add(end - i)

        # csv[i] maps idx given by user to row number of the csv data
        self.csv = {}
        idxCounter = 0
        rowCounter = 0
        lastIdx = frame_boundaries[-1][1] - len(frame_boundaries) * n
        lastRow = frame_boundaries[-1][1] - n
        while idxCounter <= lastIdx and rowCounter <= lastRow:
            if rowCounter in bounds or rowCounter + 1 in bounds:
                rowCounter += 1
            else:
                self.csv[idxCounter] = rowCounter
                idxCounter += 1
                rowCounter += 1

        self.velocities = self.calculate_joint_velocities()
        self.accelerations = self.calculate_joint_accelerations()

        # print(self.velocities[self.csv[0]])
        # print(self.accelerations[self.csv[0]])
        # print(self.velocities[self.csv[1]])
        # print(self.accelerations[self.csv[1]])
        # print(self.velocities[self.csv[2749]])
        # print(self.accelerations[self.csv[2749]])

    def calculate_joint_velocities(self):
        num_frames = self.data.shape[0]
        num_joints = (self.data.shape[1]) // 3 
        t = 1 / 120  # 120 Hz frame rate
        
        velocities = np.zeros((num_frames, num_joints, 3))
        
        for start, end in self.frame_boundaries:
            for frame in range(start + 1, end):
                for i in range(num_joints):
                    euler_values1 = self.data[frame - 1, 3 * i:3 * (i + 1)]
                    euler_values2 = self.data[frame + 1, 3 * i:3 * (i + 1)]
                    velocities[frame, i] = (euler_values2 - euler_values1) / (2 * t)

            # Boundary conditions
            for i in range(num_joints):
                velocities[start, i] = 2 * velocities[start + 1, i] - velocities[start + 2, i]
                velocities[end, i] = 2 * velocities[end - 1, i] - velocities[end - 2, i]

        velocities = velocities.reshape(num_frames, -1)
        return velocities

    def calculate_joint_accelerations(self):
        velocities = self.velocities
        num_frames = velocities.shape[0]
        num_joints = velocities.shape[1] // 3
        t = 1 / 120
        
        accelerations = np.zeros((num_frames, num_joints, 3))
        
        for start, end in self.frame_boundaries:
            for frame in range(start + 1, end):
                for i in range(num_joints):
                    velocity1 = velocities[frame - 1, 3 * i:3 * (i + 1)]
                    velocity2 = velocities[frame + 1, 3 * i:3 * (i + 1)]
                    accelerations[frame, i] = (velocity2 - velocity1) / (2 * t)

            # Boundary conditions
            for i in range(num_joints):
                accelerations[start, i] = 2 * accelerations[start + 1, i] - accelerations[start + 2, i]
                accelerations[end, i] = 2 * accelerations[end - 1, i] - accelerations[end - 2, i]

        accelerations = accelerations.reshape(num_frames, -1)
        return accelerations

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx >= len(self.csv):
            raise IndexError("Index out of range of dataset boundaries")
        i = self.csv[idx]
        input_frame = self.data[i:i + self.n] #.flatten()
        output_frame = self.data[i + self.n]
        output_frame_2 = self.data[i + self.n + 1]

        input_velocity = self.velocities[i].reshape(-1)  # Flatten velocities for the specific frame
        input_acceleration = self.accelerations[i].reshape(-1)  # Flatten accelerations for the specific frame

        input_frame = torch.tensor(input_frame, dtype=torch.float32)
        output_frame = torch.tensor(output_frame, dtype=torch.float32)
        input_velocity = torch.tensor(input_velocity, dtype=torch.float32)
        input_acceleration = torch.tensor(input_acceleration, dtype=torch.float32)

        return input_frame, output_frame, output_frame_2, input_velocity, input_acceleration, idx

    def getDim(self):
        return np.shape(self.data)

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
pose_dataset = PoseDataset(csv_file_path, frame_boundaries, 2)
pose_dataloader = DataLoader(pose_dataset, batch_size=32, shuffle=False)

# Example usage: iterate over the dataloader
for input_frame, output_frame, output_frame_2, input_velocity, input_acceleration, idx in pose_dataloader:
    #pass input_frame to neural network and use output_frame for the target
    print("Input frame:", input_frame)
    print("Output frame:", output_frame)
    break
