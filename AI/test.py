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

x = 15
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

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.title("Skeleton Overlays for Exercises")
plt.show()
