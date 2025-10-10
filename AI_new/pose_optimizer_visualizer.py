#!/usr/bin/env python3
"""
Pose Optimizer Performance Visualizer

This script specifically evaluates and visualizes the performance of pose optimizer models:
- LSTM Model
- GRU Model  
- Transformer Model

It takes a random video, extracts a 90-frame sequence, and evaluates all pose optimizer models
to provide detailed performance metrics and visualizations for pose correction.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import pickle
import random
from PIL import Image
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from mediapipe_handler.sequence_processor import SequenceProcessor
from mediapipe_handler.mediapipe_handler import MediaPipeHandler

class PoseOptimizerVisualizer:
    def __init__(self, dataset_path=None):
        """
        Initialize the pose optimizer visualizer
        
        Args:
            dataset_path: Path to the workout fitness video dataset
        """
        self.dataset_path = dataset_path or "/Users/yasinetawfeek/.cache/kagglehub/datasets/hasyimabdillah/workoutfitness-video/versions/5"
        self.models_path = os.path.join(os.path.dirname(__file__), 'models', 'pose_optimiser')
        
        # Initialize components
        self.mp_handler = MediaPipeHandler()
        self.sequence_processor = SequenceProcessor(sequence_length=90, stride=15)
        
        # Model storage
        self.models = {}
        
        # Pose landmark indices for visualization
        self.landmark_names = [
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Load models
        self.load_models()
        
    def load_models(self):
        """Load all trained pose optimizer models"""
        print("Loading pose optimizer models...")
        
        if not os.path.exists(self.models_path):
            print(f"Models path does not exist: {self.models_path}")
            return
        
        # Load models
        for model_name in ['lstm_model.keras', 'gru_model.keras', 'transformer_model.keras']:
            model_path = os.path.join(self.models_path, model_name)
            if os.path.exists(model_path):
                try:
                    self.models[model_name.replace('.keras', '')] = tf.keras.models.load_model(model_path)
                    print(f"✓ Loaded {model_name}")
                except Exception as e:
                    print(f"✗ Failed to load {model_name}: {e}")
        
        print(f"Loaded {len(self.models)} pose optimizer models")
    
    def select_random_video(self, workout_type="squat"):
        """
        Select a random video from the dataset
        
        Args:
            workout_type: Type of workout to select (default: squat)
            
        Returns:
            Path to the selected video file and workout type
        """
        # Handle None workout_type
        if workout_type is None:
            workout_type = "squat"
        
        workout_dir = os.path.join(self.dataset_path, workout_type)
        if not os.path.exists(workout_dir):
            # If specific workout type doesn't exist, pick a random one
            workout_types = [d for d in os.listdir(self.dataset_path) 
                           if os.path.isdir(os.path.join(self.dataset_path, d))]
            workout_type = random.choice(workout_types)
            workout_dir = os.path.join(self.dataset_path, workout_type)
            print(f"Selected random workout type: {workout_type}")
        
        video_files = [f for f in os.listdir(workout_dir) 
                      if f.endswith(('.mp4', '.avi', '.mov', '.MOV'))]
        
        if not video_files:
            raise ValueError(f"No video files found in {workout_dir}")
        
        selected_video = random.choice(video_files)
        video_path = os.path.join(workout_dir, selected_video)
        
        print(f"Selected video: {selected_video} from {workout_type}")
        return video_path, workout_type
    
    def extract_sequence_from_video(self, video_path, sequence_length=90):
        """
        Extract a 90-frame sequence from a video
        
        Args:
            video_path: Path to the video file
            sequence_length: Length of sequence to extract (default: 90)
            
        Returns:
            numpy array of shape (sequence_length, 36) containing pose landmarks
        """
        print(f"Extracting {sequence_length}-frame sequence from {os.path.basename(video_path)}...")
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        landmarks_sequence = []
        
        # Read all frames
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frames.append(frame)
        
        cap.release()
        
        if len(frames) < sequence_length:
            print(f"Warning: Video has only {len(frames)} frames, padding with last frame")
            # Pad with the last frame if video is too short
            while len(frames) < sequence_length:
                frames.append(frames[-1])
        
        # Process frames to get landmarks
        for i, frame in enumerate(frames[:sequence_length]):
            # Convert frame to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # Format image for MediaPipe
            mp_image = self.mp_handler.format_image_from_PIL(pil_image)
            # Get landmarks
            landmarks = self.mp_handler.predict_pose_from_image(mp_image)
            
            if landmarks.size > 0:
                landmarks_sequence.append(landmarks.flatten())
            else:
                # If no landmarks detected, use zeros
                landmarks_sequence.append(np.zeros(36))
        
        sequence = np.array(landmarks_sequence)
        print(f"Extracted sequence shape: {sequence.shape}")
        return sequence
    
    def evaluate_models(self, sequence):
        """
        Evaluate all pose optimizer models on the sequence
        
        Args:
            sequence: Input sequence of shape (90, 36)
            
        Returns:
            Dictionary with predictions and metrics
        """
        print("\n=== POSE OPTIMIZER EVALUATION ===")
        
        results = {}
        
        for model_name, model in self.models.items():
            try:
                # Prepare input
                input_sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
                
                # Make prediction (sequence-to-sequence)
                prediction = model.predict(input_sequence, verbose=0)
                corrected_sequence = prediction[0]
                
                # Calculate metrics
                mse = mean_squared_error(sequence.flatten(), corrected_sequence.flatten())
                mae = mean_absolute_error(sequence.flatten(), corrected_sequence.flatten())
                
                # Calculate correction magnitude (average distance between original and corrected)
                correction_magnitude = np.mean(np.sqrt(np.sum((sequence - corrected_sequence) ** 2, axis=1)))
                
                # Calculate per-frame metrics
                frame_mse = np.mean((sequence - corrected_sequence) ** 2, axis=1)
                frame_mae = np.mean(np.abs(sequence - corrected_sequence), axis=1)
                
                # Calculate per-landmark metrics
                landmark_mse = np.mean((sequence - corrected_sequence) ** 2, axis=0)
                landmark_mae = np.mean(np.abs(sequence - corrected_sequence), axis=0)
                
                # Calculate correction direction (positive means increase, negative means decrease)
                correction_direction = np.mean(corrected_sequence - sequence, axis=0)
                
                results[model_name] = {
                    'original_sequence': sequence,
                    'corrected_sequence': corrected_sequence,
                    'mse': mse,
                    'mae': mae,
                    'correction_magnitude': correction_magnitude,
                    'frame_mse': frame_mse,
                    'frame_mae': frame_mae,
                    'landmark_mse': landmark_mse,
                    'landmark_mae': landmark_mae,
                    'correction_direction': correction_direction,
                    'correction_vector': corrected_sequence - sequence
                }
                
                print(f"{model_name.upper()}: MSE={mse:.6f}, MAE={mae:.6f}, Correction Magnitude={correction_magnitude:.6f}")
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def visualize_predictions(self, results):
        """Create comprehensive visualizations for pose optimizer predictions"""
        if not results:
            print("No results to visualize")
            return
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(24, 18))
        fig.suptitle('Pose Optimizer Performance Analysis', fontsize=20, fontweight='bold')
        
        model_names = list(results.keys())
        model_names = [name for name in model_names if 'error' not in results[name]]
        
        if not model_names:
            print("No valid results to visualize")
            return
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # 1. Overall metrics comparison (top-left)
        ax1 = plt.subplot(4, 4, 1)
        mse_values = [results[name]['mse'] for name in model_names]
        mae_values = [results[name]['mae'] for name in model_names]
        correction_mags = [results[name]['correction_magnitude'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.25
        
        ax1.bar(x - width, mse_values, width, label='MSE', alpha=0.8, color=colors[0])
        ax1.bar(x, mae_values, width, label='MAE', alpha=0.8, color=colors[1])
        ax1.bar(x + width, correction_mags, width, label='Correction Magnitude', alpha=0.8, color=colors[2])
        
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Value')
        ax1.set_title('Overall Performance Metrics', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Frame-by-frame MSE (top-center-left)
        ax2 = plt.subplot(4, 4, 2)
        for i, (model_name, result) in enumerate(results.items()):
            if 'frame_mse' in result:
                ax2.plot(result['frame_mse'], label=model_name, alpha=0.8, color=colors[i % len(colors)])
        
        ax2.set_title('Frame-by-Frame MSE', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Frame Index')
        ax2.set_ylabel('MSE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Frame-by-frame MAE (top-center-right)
        ax3 = plt.subplot(4, 4, 3)
        for i, (model_name, result) in enumerate(results.items()):
            if 'frame_mae' in result:
                ax3.plot(result['frame_mae'], label=model_name, alpha=0.8, color=colors[i % len(colors)])
        
        ax3.set_title('Frame-by-Frame MAE', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Frame Index')
        ax3.set_ylabel('MAE')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Per-landmark MSE heatmap (top-right)
        ax4 = plt.subplot(4, 4, 4)
        landmark_mse_matrix = np.array([results[name]['landmark_mse'] for name in model_names])
        
        # Reshape to show landmarks properly (12 landmarks, 3 coordinates each)
        landmark_mse_reshaped = landmark_mse_matrix.reshape(len(model_names), 12, 3)
        landmark_mse_avg = np.mean(landmark_mse_reshaped, axis=2)  # Average across x,y,z
        
        sns.heatmap(landmark_mse_avg, annot=True, xticklabels=self.landmark_names, 
                   yticklabels=model_names, ax=ax4, cmap='viridis')
        ax4.set_title('Per-Landmark MSE Heatmap', fontsize=14, fontweight='bold')
        
        # 5. Original vs Corrected pose comparison (middle-left)
        ax5 = plt.subplot(4, 4, 5)
        
        # Plot first model's first frame as example
        first_model = model_names[0]
        result = results[first_model]
        
        # Plot first frame landmarks
        original_frame = result['original_sequence'][0]
        corrected_frame = result['corrected_sequence'][0]
        
        # Reshape to landmarks (12 landmarks, 3 coordinates each)
        original_landmarks = original_frame.reshape(12, 3)
        corrected_landmarks = corrected_frame.reshape(12, 3)
        
        # Plot x,y coordinates (ignore z for 2D visualization)
        ax5.scatter(original_landmarks[:, 0], original_landmarks[:, 1], 
                   c='blue', label='Original', s=50, alpha=0.7)
        ax5.scatter(corrected_landmarks[:, 0], corrected_landmarks[:, 1], 
                   c='red', label='Corrected', s=50, alpha=0.7)
        
        # Draw connections
        connections = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), 
                      (6, 7), (7, 8), (8, 9), (9, 10), (10, 11)]
        
        for start, end in connections:
            if start < len(original_landmarks) and end < len(original_landmarks):
                ax5.plot([original_landmarks[start, 0], original_landmarks[end, 0]], 
                        [original_landmarks[start, 1], original_landmarks[end, 1]], 
                        'b-', alpha=0.5)
                ax5.plot([corrected_landmarks[start, 0], corrected_landmarks[end, 0]], 
                        [corrected_landmarks[start, 1], corrected_landmarks[end, 1]], 
                        'r-', alpha=0.5)
        
        ax5.set_title(f'Pose Comparison ({first_model})', fontsize=14, fontweight='bold')
        ax5.set_xlabel('X Coordinate')
        ax5.set_ylabel('Y Coordinate')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_aspect('equal')
        
        # 6. Correction magnitude over time (middle-center-left)
        ax6 = plt.subplot(4, 4, 6)
        for i, (model_name, result) in enumerate(results.items()):
            if 'correction_vector' in result:
                correction_magnitude_per_frame = np.sqrt(np.sum(result['correction_vector'] ** 2, axis=1))
                ax6.plot(correction_magnitude_per_frame, label=model_name, alpha=0.8, color=colors[i % len(colors)])
        
        ax6.set_title('Correction Magnitude Over Time', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Frame Index')
        ax6.set_ylabel('Correction Magnitude')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Correction direction analysis (middle-center-right)
        ax7 = plt.subplot(4, 4, 7)
        
        # Calculate average correction direction per landmark
        avg_corrections = []
        for model_name in model_names:
            result = results[model_name]
            correction_direction = result['correction_direction']
            # Reshape to landmarks and calculate magnitude
            landmark_corrections = correction_direction.reshape(12, 3)
            landmark_magnitudes = np.sqrt(np.sum(landmark_corrections ** 2, axis=1))
            avg_corrections.append(landmark_magnitudes)
        
        avg_corrections = np.array(avg_corrections)
        
        # Plot as heatmap
        sns.heatmap(avg_corrections, annot=True, xticklabels=self.landmark_names, 
                   yticklabels=model_names, ax=ax7, cmap='RdBu_r', center=0)
        ax7.set_title('Average Correction Direction', fontsize=14, fontweight='bold')
        
        # 8. Model performance radar chart (middle-right)
        ax8 = plt.subplot(4, 4, 8, projection='polar')
        
        # Normalize metrics for radar chart
        metrics = ['MSE', 'MAE', 'Correction Magnitude']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for i, model_name in enumerate(model_names):
            result = results[model_name]
            values = [result['mse'], result['mae'], result['correction_magnitude']]
            
            # Normalize values (invert MSE and MAE so lower is better)
            normalized_values = []
            for j, val in enumerate(values):
                if j < 2:  # MSE and MAE - invert
                    normalized_values.append(1.0 / (1.0 + val))
                else:  # Correction magnitude - keep as is
                    normalized_values.append(val)
            
            normalized_values += normalized_values[:1]  # Complete the circle
            
            ax8.plot(angles, normalized_values, 'o-', linewidth=2, label=model_name, color=colors[i % len(colors)])
            ax8.fill(angles, normalized_values, alpha=0.25, color=colors[i % len(colors)])
        
        ax8.set_xticks(angles[:-1])
        ax8.set_xticklabels(metrics)
        ax8.set_title('Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
        ax8.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 9. Sequence comparison (bottom-left)
        ax9 = plt.subplot(4, 4, 9)
        
        # Plot first few frames of first model
        first_model = model_names[0]
        result = results[first_model]
        
        # Plot first 10 frames
        for frame_idx in range(min(10, result['original_sequence'].shape[0])):
            original_frame = result['original_sequence'][frame_idx]
            corrected_frame = result['corrected_sequence'][frame_idx]
            
            # Plot x coordinates over time
            ax9.plot(frame_idx, np.mean(original_frame[::3]), 'bo', alpha=0.5)  # x coordinates
            ax9.plot(frame_idx, np.mean(corrected_frame[::3]), 'ro', alpha=0.5)
        
        ax9.set_title('Sequence Evolution (X-coords)', fontsize=14, fontweight='bold')
        ax9.set_xlabel('Frame Index')
        ax9.set_ylabel('Average X Coordinate')
        ax9.grid(True, alpha=0.3)
        
        # 10. Error distribution (bottom-center-left)
        ax10 = plt.subplot(4, 4, 10)
        
        all_errors = []
        for model_name in model_names:
            result = results[model_name]
            errors = np.abs(result['original_sequence'] - result['corrected_sequence']).flatten()
            all_errors.extend(errors)
        
        ax10.hist(all_errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax10.set_title('Error Distribution', fontsize=14, fontweight='bold')
        ax10.set_xlabel('Absolute Error')
        ax10.set_ylabel('Frequency')
        ax10.grid(True, alpha=0.3)
        
        # 11. Model comparison table (bottom-center-right)
        ax11 = plt.subplot(4, 4, 11)
        ax11.axis('off')
        
        table_data = []
        for model_name in model_names:
            result = results[model_name]
            table_data.append([
                model_name.upper(),
                f"{result['mse']:.6f}",
                f"{result['mae']:.6f}",
                f"{result['correction_magnitude']:.6f}"
            ])
        
        table = ax11.table(cellText=table_data,
                         colLabels=['Model', 'MSE', 'MAE', 'Correction Mag'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax11.set_title('Detailed Metrics', fontsize=14, fontweight='bold')
        
        # 12. Performance summary (bottom-right)
        ax12 = plt.subplot(4, 4, 12)
        ax12.axis('off')
        
        # Calculate summary statistics
        avg_mse = np.mean(mse_values)
        avg_mae = np.mean(mae_values)
        avg_correction = np.mean(correction_mags)
        
        best_mse_model = model_names[np.argmin(mse_values)]
        best_mae_model = model_names[np.argmin(mae_values)]
        
        summary_text = f"""
        PERFORMANCE SUMMARY
        
        Average MSE: {avg_mse:.6f}
        Average MAE: {avg_mae:.6f}
        Average Correction: {avg_correction:.6f}
        
        Best MSE: {best_mse_model}
        Best MAE: {best_mae_model}
        
        Total Models: {len(model_names)}
        """
        
        ax12.text(0.1, 0.5, summary_text, transform=ax12.transAxes, fontsize=12,
                 verticalalignment='center', fontfamily='monospace')
        ax12.set_title('Summary Statistics', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def run_evaluation(self, video_path=None, workout_type=None):
        """
        Run complete pose optimizer evaluation
        
        Args:
            video_path: Path to video file (if None, selects random)
            workout_type: Type of workout (if None, selects random)
        """
        print("=" * 60)
        print("POSE OPTIMIZER PERFORMANCE EVALUATION")
        print("=" * 60)
        
        # Select video if not provided
        if video_path is None:
            video_path, workout_type = self.select_random_video(workout_type)
        
        # Extract sequence
        sequence = self.extract_sequence_from_video(video_path)
        
        # Evaluate models
        results = self.evaluate_models(sequence)
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        self.visualize_predictions(results)
        
        # Summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Video: {os.path.basename(video_path)}")
        print(f"Workout Type: {workout_type}")
        print(f"Sequence Shape: {sequence.shape}")
        
        print("\nModel Results:")
        for model_name, result in results.items():
            if 'error' not in result:
                print(f"  {model_name}: MSE={result['mse']:.6f}, MAE={result['mae']:.6f}, Correction Magnitude={result['correction_magnitude']:.6f}")
        
        return {
            'results': results,
            'sequence': sequence,
            'video_path': video_path,
            'workout_type': workout_type
        }

def main():
    """Main function to run the pose optimizer evaluation"""
    # Initialize visualizer
    visualizer = PoseOptimizerVisualizer()
    
    # Run evaluation
    results = visualizer.run_evaluation()
    
    print("\nPose optimizer evaluation completed successfully!")
    return results

if __name__ == "__main__":
    main()
