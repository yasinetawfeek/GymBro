import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple
import os
from PIL import Image
try:
    from .mediapipe_handler import MediaPipeHandler
except ImportError:
    # Fallback for direct script or notebook execution
    import sys, os
    handler_dir = os.path.dirname(os.path.abspath(__file__))
    if handler_dir not in sys.path:
        sys.path.append(handler_dir)
    from mediapipe_handler import MediaPipeHandler

class SequenceProcessor:
    def __init__(self, sequence_length: int = 90, stride: int = 15, target_fps: int = 30):  # 3 seconds at 30fps = 90 frames, stride = 0.5 seconds
        self.mp_handler = MediaPipeHandler()
        self.sequence_length = sequence_length
        self.stride = stride  # Number of frames to move forward for each sequence
        self.target_fps = target_fps  # Target frame rate for normalization
    
    def normalize_frame_rate(self, frames: List[np.ndarray], original_fps: float) -> List[np.ndarray]:
        """
        Normalize frame rate to target_fps by duplicating or removing frames.
        
        Args:
            frames: List of video frames
            original_fps: Original frame rate of the video
            
        Returns:
            List of frames normalized to target_fps
        """
        if abs(original_fps - self.target_fps) < 0.1:  # Already at target fps
            return frames
        
        fps_ratio = original_fps / self.target_fps
        
        if fps_ratio > 1.0:  # Original fps is higher - need to remove frames
            # Calculate which frames to keep (evenly distributed)
            total_frames = len(frames)
            keep_indices = np.linspace(0, total_frames - 1, int(total_frames / fps_ratio), dtype=int)
            normalized_frames = [frames[i] for i in keep_indices]
            print(f"  Normalized {total_frames} frames at {original_fps:.1f}fps to {len(normalized_frames)} frames at {self.target_fps}fps (removed {total_frames - len(normalized_frames)} frames)")
            
        else:  # Original fps is lower - need to duplicate frames
            # Calculate how many times to repeat each frame
            repeat_factor = 1 / fps_ratio
            normalized_frames = []
            
            for i, frame in enumerate(frames):
                # Calculate how many times to repeat this frame
                if i == len(frames) - 1:  # Last frame
                    # For the last frame, add remaining frames needed
                    remaining_frames = int(len(frames) * repeat_factor) - len(normalized_frames)
                    normalized_frames.extend([frame] * remaining_frames)
                else:
                    # For other frames, calculate based on position
                    start_pos = i * repeat_factor
                    end_pos = (i + 1) * repeat_factor
                    num_repeats = int(np.ceil(end_pos) - np.floor(start_pos))
                    normalized_frames.extend([frame] * num_repeats)
            
            print(f"  Normalized {len(frames)} frames at {original_fps:.1f}fps to {len(normalized_frames)} frames at {self.target_fps}fps (duplicated frames)")
        
        return normalized_frames
    
    def save_normalized_video(self, input_path: str, output_path: str) -> None:
        """
        Save a video with normalized frame rate to a new file.
        
        Args:
            input_path: Path to input video
            output_path: Path to save normalized video
        """
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Read all frames
        frames = []
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frames.append(frame)
        cap.release()
        
        # Normalize frame rate
        normalized_frames = self.normalize_frame_rate(frames, original_fps)
        
        # Save normalized video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.target_fps, (width, height))
        
        for frame in normalized_frames:
            out.write(frame)
        
        out.release()
        print(f"Saved normalized video: {output_path}")
    
    def process_video(self, video_path: str) -> List[np.ndarray]:
        """Process a video file and return sequences of pose landmarks."""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing {os.path.basename(video_path)}: {original_fps:.1f}fps, {total_frames} frames")
        
        frames = []
        landmarks_sequence = []
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            frames.append(frame)
            
        cap.release()
        
        # Normalize frame rate to target_fps
        frames = self.normalize_frame_rate(frames, original_fps)
        
        # Process frames to get landmarks
        for frame in frames:
            # Convert frame to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # Format image for MediaPipe
            mp_image = self.mp_handler.format_image_from_PIL(pil_image)
            # Get landmarks
            landmarks = self.mp_handler.predict_pose_from_image(mp_image)
            if landmarks.size > 0:  # Check if landmarks were detected
                landmarks_sequence.append(landmarks.flatten())
                
        # Create overlapping sequences using stride
        sequences = []
        for i in range(0, len(landmarks_sequence) - self.sequence_length + 1, self.stride):
            sequence = np.stack(landmarks_sequence[i:i + self.sequence_length])
            if len(sequence) == self.sequence_length:  # Only add complete sequences
                sequences.append(sequence)
            
        return sequences
    
    def process_video_directory(self, directory_path: str) -> Tuple[List[np.ndarray], List[str]]:
        """Process all videos in a directory and return sequences with their labels."""
        all_sequences = []
        all_labels = []
        label_counter = 0
        for label in os.listdir(directory_path):
            label_path = os.path.join(directory_path, label)
            if not os.path.isdir(label_path):
                continue

            counter = 0    
            for video_file in os.listdir(label_path):
                if not video_file.endswith(('.mp4', '.avi', '.mov')):
                    print(f"Skipped non-video file: {video_file}")
                    counter += 1
                    continue
                    
                video_path = os.path.join(label_path, video_file)
                sequences = self.process_video(video_path)
                
                all_sequences.extend(sequences)
                all_labels.extend([label] * len(sequences))
                counter += 1
                print(f"Processed {counter}/{len(os.listdir(label_path))} videos in label '{label}'({label_counter}/{len(os.listdir(directory_path))}) (sequences: {len(sequences)}, video: {video_file})")
            label_counter += 1    
        
        return np.array(all_sequences), np.array(all_labels)