import bpy
import numpy as np

# Fits SMPL-X data from AMASS to SMPL-X add-on in Blender and displays

# Path to the AMASS dataset file
amass_data_path = 'D:/Claire/CMUsmplx/CMU/01/01_01_stageii.npz'
amass_data = np.load(amass_data_path)
pose_body = amass_data['poses']  #55 joints of SMPL-X
num_frames = pose_body.shape[0]
print(pose_body.shape)  # shape (2751, 165) for 'poses' and (2751, 63) for 'pose_body'

# Joint names and their corresponding indices in SMPL-X. e.g. "pelvis" is 0th joint in amass "poses"
joint_names = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar",
    "right_collar", "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "jaw", "left_eye_smplhf", "right_eye_smplhf", "left_index1",
    "left_index2", "left_index3", "left_middle1", "left_middle2", "left_middle3", "left_pinky1",
    "left_pinky2", "left_pinky3", "left_ring1", "left_ring2", "left_ring3", "left_thumb1",
    "left_thumb2", "left_thumb3", "right_index1", "right_index2", "right_index3", "right_middle1",
    "right_middle2", "right_middle3", "right_pinky1", "right_pinky2", "right_pinky3", "right_ring1",
    "right_ring2", "right_ring3", "right_thumb1", "right_thumb2", "right_thumb3"
]

# Make sure the armature is a SMPL-X add-on in pose mode in Blender
armature_name = 'SMPLX-female'
armature = bpy.data.objects.get(armature_name)
bpy.ops.object.mode_set(mode='POSE')

# Loop through all frames and set keyframes for each joint
for frame in range(num_frames):
    for i, joint_name in enumerate(joint_names):
        print(i, joint_name)
        bone = armature.pose.bones.get(joint_name)
        bone.rotation_mode = 'XYZ'
        if bone.name == "pelvis":
            continue
        if bone is not None:
            euler_values = pose_body[frame, 3*i:3*i+3]
            bone.rotation_euler = euler_values
            bone.keyframe_insert(data_path="rotation_euler", frame=frame)

# Set the end frame for the animation
bpy.context.scene.frame_end = num_frames

# Set the frame rate to play the animation continuously
frame_rate = 1 / 0.00001 # 3ms per frame
bpy.context.scene.render.fps = int(frame_rate)

# Play the animation
bpy.ops.screen.animation_play()
