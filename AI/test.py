import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
from mediapipe_handler import MediaPipeHandler, add_new_label
import matplotlib.pyplot as plt
import PIL.Image as pl
from mpl_toolkits.mplot3d import Axes3D

ds = load_dataset("averrous/workout")
new_ds = ds.map(add_new_label)

# train = new_ds['train']
test = new_ds['test']
# validation = new_ds['validation']

# train = pd.DataFrame(train)
test = pd.DataFrame(test)
# validation = pd.DataFrame(validation)

media_pipe_handler = MediaPipeHandler()

#train_new = media_pipe_handler.pandas_add_detections_from_image_and_remove_nulls(train)
test_new = media_pipe_handler.pandas_add_detections_from_image_and_remove_nulls(test)
# test_new.to_csv('data/test_new.csv', index=False)
# validation_new = media_pipe_handler.pandas_add_detections_from_image(validation)

def normalise_pose(joint_coords, joint_names):
    # Get indices for key joints
    left_hip_idx = joint_names.get_loc("left_hip")
    right_hip_idx = joint_names.get_loc("right_hip")
    left_shoulder_idx = joint_names.get_loc("left_shoulder")
    right_shoulder_idx = joint_names.get_loc("right_shoulder")

    # Compute hip midpoint (translation step)
    hip_midpoint = (joint_coords[left_hip_idx] + joint_coords[right_hip_idx]) / 2
    for j in joint_names:
        if (j != "image") and (j != "label") and (j != "muscle group"):
            joint_coords[j] -= hip_midpoint  # Translate all points so hips are at origin

    # Compute shoulder width (scaling step)
    # shoulder_width = np.linalg.norm(joint_coords[left_shoulder_idx] - joint_coords[right_shoulder_idx])
    # if shoulder_width > 0:
    #     joint_coords /= shoulder_width  # Normalize by shoulder width to maintain scale

######

    # # Compute shoulder vector
    # shoulder_vector = joint_coords[right_shoulder_idx] - joint_coords[left_shoulder_idx]
    # shoulder_vector /= np.linalg.norm(shoulder_vector)  # Normalize

    # # Desired X-axis unit vector
    # target_vector = np.array([1, 0, 0])

    # # Compute rotation axis (cross product of shoulder vector and target X-axis)
    # rotation_axis = np.cross(shoulder_vector, target_vector)
    # rotation_axis /= np.linalg.norm(rotation_axis)  # Normalize

    # # Compute rotation angle (dot product)
    # angle = np.arccos(np.dot(shoulder_vector, target_vector))

    # # Construct rotation matrix using Rodrigues' formula
    # K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
    #               [rotation_axis[2], 0, -rotation_axis[0]],
    #               [-rotation_axis[1], rotation_axis[0], 0]])

    # R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    # # Apply rotation
    # joint_coords = np.dot(joint_coords, R.T)  # Rotate all joints

    return joint_coords


# Define joint connections (indices based on typical MediaPipe model)
joint_pairs = {
    ('left_shoulder', 'right_shoulder'),
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'),
    ('right_wrist', 'right_pinky'),
    ('right_wrist', 'right_index'),
    ('right_wrist', 'right_thumb'),
    ('left_wrist', 'left_pinky'),
    ('left_wrist', 'left_index'),
    ('left_wrist', 'left_thumb'),
    ('left_hip', 'right_hip'),
    ('left_hip', 'left_knee'),
    ('right_hip', 'right_knee'),
    ('left_knee', 'left_ankle'),
    ('right_knee', 'right_ankle'),
    ('left_ankle', 'left_heel'),
    ('right_ankle', 'right_heel'),
    ('left_heel', 'left_foot_index'),
    ('right_heel', 'right_foot_index'),
    ('left_foot_index', 'right_foot_index')
}

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

x = 8
for i in range (x,x+1):
    exercise = i
    exercise_data = test_new[test_new["label"]==i]
    #exercise_data = exercise_data.drop(columns=["image", "label", "muscle group"])
    for index, row in exercise_data.head(1).iterrows():
            for joint1, joint2 in joint_pairs:
                # new_row = normalise_pose(row, exercise_data.columns)
                # x1, y1, z1 = new_row[joint1]
                # x2, y2, z2 = new_row[joint2]
                
                x1, y1, z1 = row[joint1]
                x2, y2, z2 = row[joint2]

                ax.plot([x1, x2], [y1, y2], [z1, z2], marker='o', label=exercise, alpha=0.7)
                ax.text(x1, y1, z1, joint1, fontsize=8, color='black')
                ax.text(x2, y2, z2, joint2, fontsize=8, color='black')
            
            pl.fromarray(np.asarray(row['image']).astype(np.uint8)).show()

            # new_row = normalise_pose(row, exercise_data.columns)
            # x1, y1, z1 = new_row['right_shoulder']
            # ax.plot([x1], [y1], [z1], marker='o', label=exercise, alpha=0.7)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.title("Skeleton Overlays for Exercises")
plt.show()
