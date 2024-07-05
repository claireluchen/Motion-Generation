import numpy as np
from getVelocity import calculate_joint_velocities

def calculate_joint_accelerations(amass_data_path):
    #array of velocities using the function from getVelocity.py
    velocities = calculate_joint_velocities(amass_data_path)
    
    num_frames = velocities.shape[0]
    num_joints = velocities.shape[1]
    t = 1 / np.load(amass_data_path)['mocap_frame_rate']

    #array for accelerations
    accelerations = np.zeros((num_frames, num_joints, 3))

    #calculate accelerations for each joint from frame 1 to num_frames - 2
    for frame in range(1, num_frames - 1):
        for i in range(num_joints):
            velocity1 = velocities[frame - 1, i]
            velocity2 = velocities[frame + 1, i]
            accelerations[frame, i] = (velocity2 - velocity1) / (2 * t)

    #calculate acceleration for the first frame
    for i in range(num_joints):
        accelerations[0, i] = 2 * accelerations[1, i] - accelerations[2, i]

    #calculate acceleration for the last frame
    for i in range(num_joints):
        accelerations[num_frames - 1, i] = 2 * accelerations[num_frames - 2, i] - accelerations[num_frames - 3, i]

    return accelerations

#example usage:
amass_data_path = 'D:/Claire/CMUsmplx/CMU/01/01_01_stageii.npz'
accelerations = calculate_joint_accelerations(amass_data_path)
print(accelerations.shape)
print(accelerations[0])
print(accelerations[1])
print(accelerations[-1])
