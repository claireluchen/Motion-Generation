import numpy as np

def calculate_joint_velocities(amass_data_path):
    amass_data = np.load(amass_data_path)

    # Extract pose data and other relevant information
    poses = amass_data['poses']  # 55 joints of SMPL-X
    num_frames = poses.shape[0]
    num_joints = int(poses.shape[1] / 3)
    t = 1 / amass_data['mocap_frame_rate']

    #velocities[i] stores the velocity for 55 joints in ith frame
    velocities = np.zeros((num_frames, num_joints, 3))

    # Calculate velocities for each joint from frame 1 to num_frames - 2
    for frame in range(1, num_frames - 1):
        for i in range(num_joints):
            euler_values1 = poses[frame - 1, 3 * i:3 * i + 3]
            euler_values2 = poses[frame + 1, 3 * i:3 * i + 3]
            velocities[frame, i] = (euler_values2 - euler_values1) / (2 * t)

    # Calculate velocity for the first frame
    for i in range(num_joints): #v1 = (v2 + v0) / 2, so v0 = 2v1 - v2
        velocities[0, i] = 2 * velocities[1, i] - velocities[2, i]

    # Calculate velocity for the last frame
    for i in range(num_joints): #v1 = (v2 + v0) / 2, so v2 = 2v1 - v0
        velocities[num_frames - 1, i] = 2 * velocities[num_frames - 2, i] - velocities[num_frames - 3, i]

    return velocities

#example usage:
amass_data_path = 'D:/Claire/CMUsmplx/CMU/01/01_01_stageii.npz'
velocities = calculate_joint_velocities(amass_data_path)
print(velocities.shape)
print(velocities[0])
print(velocities[1])
print(velocities[-1])
