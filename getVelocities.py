import numpy as np

#path to the AMASS dataset file
amass_data_path = 'D:/Claire/CMUsmplx/CMU/01/01_01_stageii.npz'
amass_data = np.load(amass_data_path)

poses = amass_data['poses']  # 55 joints of SMPL-X
num_frames = poses.shape[0]
num_joints = int(poses.shape[1] / 3)
t = 1 / amass_data['mocap_frame_rate']

velocities = []  #velocities[i] stores the velocity for 55 joints in ith frame

#velocities for each joint in each frame
for frame in range(1, num_frames-1):
    cur = []
    for i in range(num_joints):
        euler_values1 = poses[frame-1, 3*i:3*i+3]
        euler_values2 = poses[frame+1, 3*i:3*i+3]
        velocity = (euler_values2 - euler_values1) / (2 * t)
        
        cur.append(velocity)
    velocities.append(cur)


#velocity for the first frame
for i in range(num_joints): #v1 = (v2 + v0) / 2, so v0 = 2v1 - v2
    velocities[0, i] = 2 * velocities[1, i] - velocities[2, i]

#velocity for the last frame
for i in range(num_joints): #v2 = 2v1 - v0
    velocities[num_frames - 1, i] = 2 * velocities[num_frames - 2, i] - velocities[num_frames - 3, i]

#convert velocities to a numpy array
velocities = np.array(velocities)
print(velocities.shape)
print(velocities[0])
print(velocities[1])
print(velocities[num_frames - 1])
