import mediapipe as mp
import time
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import ImageFormat
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MediaPipeHandler():
    def __init__(self):
        self.model_path = "/Users/yasinetawfeek/Downloads/pose_landmarker_full.task"

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        self.options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True,
            min_pose_detection_confidence=0.5,  # Try lowering this if no poses are detected
            min_pose_presence_confidence=0.5,   # Try lowering this if no poses are detected
            min_tracking_confidence=0.5         # Try lowering this if no poses are detected
        )
        self.detector = vision.PoseLandmarker.create_from_options(self.options)

        self.image_format = ImageFormat.SRGB

    def format_image_from_path(self, image_path):
        image = Image.open(image_path)
        image = image.convert("RGB")
        image_np = np.array(image)

        mp_image = mp.Image(
        data=image_np,
        image_format=self.image_format
        )

        return mp_image

    def format_image_from_PIL(self, image):
        image = image.convert("RGB")
        image_np = np.array(image)

        mp_image = mp.Image(
        data=image_np,
        image_format=self.image_format
        )

        return mp_image
    
    def predict_from_image(self, image):
        detection_result = self.detector.detect(image)

        return detection_result
    
    def predict_pose_from_image(self, image):
        

        detection_result = self.detector.detect(image)

        if detection_result.pose_landmarks:
            landmark_coordinates = []
            for pose_idx, pose_landmarks in enumerate(detection_result.pose_landmarks):
                pose_coordinates = []

                for landmark_idx, landmark in enumerate(pose_landmarks):
                    x, y, z, = landmark.x, landmark.y, landmark.z
                    pose_coordinates.append((x, y, z))
                
                landmark_coordinates = pose_coordinates
            
            return np.array(landmark_coordinates)
        
        else:
            return np.array([])
    
    def pandas_add_detections_from_image(self, df, image_column_name="image", detections_column_name="pose"):
        
        images = df[image_column_name]
        pose = []

        counter = 0
        for image in images:
            start = time.time()
            image_formated = self.format_image_from_PIL(image)
            pose.append(self.predict_pose_from_image(image_formated))
            print(f"predicted image {counter} from {len(images)}, time: {time.time() - start}")
            counter+= 1


        df['pose'] = pose

        return df
    def visualise_from_pose(self, pose):
        x, y, z = pose[:, 0], pose[:, 1], pose[:, 2]

        # Create 3D plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c='b', marker='o')

        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Scatter Plot of Given Points')

        # Show plot
        plt.show()
    
    def draw_landmarks_on_image(self, image, detection_result):
        # Make a copy of the image to draw on
        annotated_image = image.copy()
        
        # Get image dimensions
        image_height, image_width = image.shape[:2]
        
        # Define landmark connection lines (simplified version)
        POSE_CONNECTIONS = [
            # Torso
            (11, 12), (12, 24), (24, 23), (23, 11), 
            # Left arm
            (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
            # Right arm
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
            # Left leg
            (23, 25), (25, 27), (27, 29), (27, 31),
            # Right leg
            (24, 26), (26, 28), (28, 30), (28, 32),
            # Face
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            # Face to torso
            (0, 9), (9, 10), (10, 11), (10, 12)
        ]
        
        # Draw landmarks and connections for each detected pose
        if detection_result.pose_landmarks:
            for pose_landmarks in detection_result.pose_landmarks:
                # Draw landmarks
                for idx, landmark in enumerate(pose_landmarks):
                    # Convert normalized coordinates to pixel values
                    x = int(landmark.x * image_width)
                    y = int(landmark.y * image_height)
                    
                    # Draw a circle at each landmark
                    cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)  # Green circle
                    
                    # Add landmark index
                    cv2.putText(annotated_image, str(idx), (x + 5, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                    
                # Draw connections
                for connection in POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    
                    if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
                        # Get start and end points
                        start_point = (
                            int(pose_landmarks[start_idx].x * image_width),
                            int(pose_landmarks[start_idx].y * image_height)
                        )
                        end_point = (
                            int(pose_landmarks[end_idx].x * image_width),
                            int(pose_landmarks[end_idx].y * image_height)
                        )
                        
                        # Draw a line connecting the points
                        cv2.line(annotated_image, start_point, end_point, (245, 117, 66), 2)
        
        return annotated_image
