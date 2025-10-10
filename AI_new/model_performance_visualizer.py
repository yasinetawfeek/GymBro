#!/usr/bin/env python3
"""
Model Performance Visualizer for GymBro AI Models

This script evaluates and visualizes the performance of three AI models:
1. Workout Classifier (LSTM, GRU, Transformer)
2. Muscle Group Classifier (LSTM, GRU, Transformer) 
3. Pose Optimizer (LSTM, GRU, Transformer)

It takes a random video, extracts a 90-frame sequence, and evaluates all models
to provide comprehensive performance metrics and visualizations.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import pickle
import random
from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from mediapipe_handler.sequence_processor import SequenceProcessor
from mediapipe_handler.mediapipe_handler import MediaPipeHandler
from muscle_group_mapping import MUSCLE_GROUPS_7, MUSCLE_GROUP_LABELS_7

class ModelPerformanceVisualizer:
    def __init__(self, dataset_path=None):
        """
        Initialize the visualizer with dataset path and model paths
        
        Args:
            dataset_path: Path to the workout fitness video dataset
        """
        self.dataset_path = dataset_path or "/Users/yasinetawfeek/.cache/kagglehub/datasets/hasyimabdillah/workoutfitness-video/versions/5"
        self.models_path = os.path.join(os.path.dirname(__file__), 'models')
        
        # Initialize components
        self.mp_handler = MediaPipeHandler()
        self.sequence_processor = SequenceProcessor(sequence_length=90, stride=15)
        
        # Model storage
        self.workout_models = {}
        self.muscle_group_models = {}
        self.pose_optimizer_models = {}
        
        # Supporting data
        self.label_encoder = None
        self.feature_scaler = None
        self.muscle_group_labels = MUSCLE_GROUP_LABELS_7
        
        # Workout mapping
        self.workout_map = {
            0: "barbell bicep curl", 1: "bench press", 2: "chest fly machine", 
            3: "deadlift", 4: "decline bench press", 5: "hammer curl", 
            6: "hip thrust", 7: "incline bench press", 8: "lat pulldown", 
            9: "lateral raises", 10: "leg extensions", 11: "leg raises", 
            12: "plank", 13: "pull up", 14: "push ups", 15: "romanian deadlift", 
            16: "russian twist", 17: "shoulder press", 18: "squat", 
            19: "t bar row", 20: "tricep dips", 21: "tricep pushdown"
        }
        
        # Load all models and supporting data
        self.load_models()
        
    def load_models(self):
        """Load all trained models and supporting data"""
        print("Loading models and supporting data...")
        
        # Load workout classifier models
        workout_models_path = os.path.join(self.models_path, 'workout_classifier')
        if os.path.exists(workout_models_path):
            for model_name in ['lstm_model.keras', 'gru_model.keras', 'transformer_model.keras']:
                model_path = os.path.join(workout_models_path, model_name)
                if os.path.exists(model_path):
                    try:
                        self.workout_models[model_name.replace('.keras', '')] = tf.keras.models.load_model(model_path)
                        print(f"✓ Loaded {model_name}")
                    except Exception as e:
                        print(f"✗ Failed to load {model_name}: {e}")
        
        # Load label encoder for workout classifier
        label_encoder_path = os.path.join(workout_models_path, 'label_encoder.pkl')
        if os.path.exists(label_encoder_path):
            with open(label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print("✓ Loaded workout label encoder")
        
        # Load muscle group classifier models
        muscle_models_path = os.path.join(self.models_path, 'muscle_group_classifier')
        if os.path.exists(muscle_models_path):
            for model_name in ['lstm_model.keras', 'gru_model.keras', 'transformer_model.keras']:
                model_path = os.path.join(muscle_models_path, model_name)
                if os.path.exists(model_path):
                    try:
                        self.muscle_group_models[model_name.replace('.keras', '')] = tf.keras.models.load_model(model_path)
                        print(f"✓ Loaded muscle group {model_name}")
                    except Exception as e:
                        print(f"✗ Failed to load muscle group {model_name}: {e}")
        
        # Load pose optimizer models
        pose_models_path = os.path.join(self.models_path, 'pose_optimiser')
        if os.path.exists(pose_models_path):
            for model_name in ['lstm_model.keras', 'gru_model.keras', 'transformer_model.keras']:
                model_path = os.path.join(pose_models_path, model_name)
                if os.path.exists(model_path):
                    try:
                        self.pose_optimizer_models[model_name.replace('.keras', '')] = tf.keras.models.load_model(model_path)
                        print(f"✓ Loaded pose optimizer {model_name}")
                    except Exception as e:
                        print(f"✗ Failed to load pose optimizer {model_name}: {e}")
        
        print(f"Loaded {len(self.workout_models)} workout models, {len(self.muscle_group_models)} muscle group models, {len(self.pose_optimizer_models)} pose optimizer models")
    
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
    
    def evaluate_workout_classifier(self, sequence, true_label=None):
        """
        Evaluate workout classifier models on the sequence
        
        Args:
            sequence: Input sequence of shape (90, 36)
            true_label: True workout label for comparison
            
        Returns:
            Dictionary with predictions and metrics
        """
        print("\n=== WORKOUT CLASSIFIER EVALUATION ===")
        
        results = {}
        
        for model_name, model in self.workout_models.items():
            try:
                # Prepare input
                input_sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
                
                # Make prediction
                prediction = model.predict(input_sequence, verbose=0)
                predicted_class = np.argmax(prediction[0])
                confidence = np.max(prediction[0])
                
                # Get workout name
                workout_name = self.workout_map.get(predicted_class, f"Unknown ({predicted_class})")
                
                results[model_name] = {
                    'predicted_class': predicted_class,
                    'workout_name': workout_name,
                    'confidence': confidence,
                    'all_probabilities': prediction[0]
                }
                
                print(f"{model_name.upper()}: {workout_name} (confidence: {confidence:.3f})")
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def evaluate_muscle_group_classifier(self, sequence, workout_type=None):
        """
        Evaluate muscle group classifier models on the sequence
        
        Args:
            sequence: Input sequence of shape (90, 36)
            workout_type: Workout type for context
            
        Returns:
            Dictionary with predictions and metrics
        """
        print("\n=== MUSCLE GROUP CLASSIFIER EVALUATION ===")
        
        results = {}
        
        for model_name, model in self.muscle_group_models.items():
            try:
                # Prepare input
                input_sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
                
                # Make prediction
                prediction = model.predict(input_sequence, verbose=0)
                predicted_class = np.argmax(prediction[0])
                confidence = np.max(prediction[0])
                
                # Get muscle group name
                muscle_group_name = self.muscle_group_labels[predicted_class]
                
                results[model_name] = {
                    'predicted_class': predicted_class,
                    'muscle_group_name': muscle_group_name,
                    'confidence': confidence,
                    'all_probabilities': prediction[0]
                }
                
                print(f"{model_name.upper()}: {muscle_group_name} (confidence: {confidence:.3f})")
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def evaluate_pose_optimizer(self, sequence):
        """
        Evaluate pose optimizer models on the sequence
        
        Args:
            sequence: Input sequence of shape (90, 36)
            
        Returns:
            Dictionary with predictions and metrics
        """
        print("\n=== POSE OPTIMIZER EVALUATION ===")
        
        results = {}
        
        for model_name, model in self.pose_optimizer_models.items():
            try:
                # Prepare input
                input_sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
                
                # Make prediction (sequence-to-sequence)
                prediction = model.predict(input_sequence, verbose=0)
                
                # Calculate metrics
                mse = np.mean((sequence - prediction[0]) ** 2)
                mae = np.mean(np.abs(sequence - prediction[0]))
                
                # Calculate correction magnitude
                correction_magnitude = np.mean(np.sqrt(np.sum((sequence - prediction[0]) ** 2, axis=1)))
                
                results[model_name] = {
                    'corrected_sequence': prediction[0],
                    'mse': mse,
                    'mae': mae,
                    'correction_magnitude': correction_magnitude,
                    'original_sequence': sequence
                }
                
                print(f"{model_name.upper()}: MSE={mse:.6f}, MAE={mae:.6f}, Correction Magnitude={correction_magnitude:.6f}")
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def visualize_workout_predictions(self, workout_results, true_label=None):
        """Visualize workout classifier predictions"""
        if not workout_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Workout Classifier Performance', fontsize=16)
        
        # 1. Confidence comparison
        model_names = list(workout_results.keys())
        confidences = [workout_results[name].get('confidence', 0) for name in model_names]
        
        axes[0, 0].bar(model_names, confidences, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0, 0].set_title('Prediction Confidence by Model')
        axes[0, 0].set_ylabel('Confidence')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Probability distributions
        for i, (model_name, result) in enumerate(workout_results.items()):
            if 'all_probabilities' in result:
                axes[0, 1].plot(result['all_probabilities'], label=model_name, alpha=0.7)
        
        axes[0, 1].set_title('Probability Distribution Across Classes')
        axes[0, 1].set_xlabel('Workout Class')
        axes[0, 1].set_ylabel('Probability')
        axes[0, 1].legend()
        axes[0, 1].set_xticks(range(0, len(self.workout_map), 3))
        
        # 3. Prediction comparison
        predictions = [workout_results[name].get('workout_name', 'Error') for name in model_names]
        axes[1, 0].text(0.1, 0.5, '\n'.join([f"{name}: {pred}" for name, pred in zip(model_names, predictions)]), 
                        transform=axes[1, 0].transAxes, fontsize=12, verticalalignment='center')
        axes[1, 0].set_title('Model Predictions')
        axes[1, 0].axis('off')
        
        # 4. Confidence heatmap
        if len(model_names) > 1:
            conf_matrix = np.array([[confidences[i] if i == j else 0 for j in range(len(model_names))] 
                                   for i in range(len(model_names))])
            sns.heatmap(conf_matrix, annot=True, xticklabels=model_names, yticklabels=model_names, 
                       ax=axes[1, 1], cmap='Blues')
            axes[1, 1].set_title('Confidence Comparison')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_muscle_group_predictions(self, muscle_results):
        """Visualize muscle group classifier predictions"""
        if not muscle_results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Muscle Group Classifier Performance', fontsize=16)
        
        # 1. Confidence comparison
        model_names = list(muscle_results.keys())
        confidences = [muscle_results[name].get('confidence', 0) for name in model_names]
        
        axes[0, 0].bar(model_names, confidences, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0, 0].set_title('Prediction Confidence by Model')
        axes[0, 0].set_ylabel('Confidence')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Probability distributions
        for i, (model_name, result) in enumerate(muscle_results.items()):
            if 'all_probabilities' in result:
                axes[0, 1].plot(result['all_probabilities'], label=model_name, alpha=0.7)
        
        axes[0, 1].set_title('Probability Distribution Across Muscle Groups')
        axes[0, 1].set_xlabel('Muscle Group')
        axes[0, 1].set_ylabel('Probability')
        axes[0, 1].legend()
        axes[0, 1].set_xticks(range(len(self.muscle_group_labels)))
        axes[0, 1].set_xticklabels(self.muscle_group_labels, rotation=45)
        
        # 3. Prediction comparison
        predictions = [muscle_results[name].get('muscle_group_name', 'Error') for name in model_names]
        axes[1, 0].text(0.1, 0.5, '\n'.join([f"{name}: {pred}" for name, pred in zip(model_names, predictions)]), 
                        transform=axes[1, 0].transAxes, fontsize=12, verticalalignment='center')
        axes[1, 0].set_title('Model Predictions')
        axes[1, 0].axis('off')
        
        # 4. Muscle group distribution
        muscle_counts = {}
        for result in muscle_results.values():
            if 'muscle_group_name' in result:
                muscle = result['muscle_group_name']
                muscle_counts[muscle] = muscle_counts.get(muscle, 0) + 1
        
        if muscle_counts:
            axes[1, 1].pie(muscle_counts.values(), labels=muscle_counts.keys(), autopct='%1.1f%%')
            axes[1, 1].set_title('Predicted Muscle Group Distribution')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_pose_optimization(self, pose_results):
        """Visualize pose optimizer predictions"""
        if not pose_results:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Pose Optimizer Performance', fontsize=16)
        
        model_names = list(pose_results.keys())
        
        # Plot for each model
        for i, (model_name, result) in enumerate(pose_results.items()):
            if 'error' in result:
                continue
                
            row = i // 3
            col = i % 3
            
            if row >= 2 or col >= 3:
                continue
            
            # Plot original vs corrected pose
            original = result['original_sequence']
            corrected = result['corrected_sequence']
            
            # Plot first frame
            axes[row, col].plot(original[0], 'b-', label='Original', alpha=0.7)
            axes[row, col].plot(corrected[0], 'r-', label='Corrected', alpha=0.7)
            axes[row, col].set_title(f'{model_name.upper()} - Frame 1')
            axes[row, col].set_xlabel('Landmark Index')
            axes[row, col].set_ylabel('Value')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(model_names), 6):
            row = i // 3
            col = i % 3
            if row < 2 and col < 3:
                axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Additional visualization: Correction magnitude comparison
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        correction_magnitudes = []
        mse_values = []
        mae_values = []
        
        for model_name, result in pose_results.items():
            if 'error' not in result:
                correction_magnitudes.append(result['correction_magnitude'])
                mse_values.append(result['mse'])
                mae_values.append(result['mae'])
        
        if correction_magnitudes:
            x = np.arange(len(model_names))
            width = 0.25
            
            ax.bar(x - width, correction_magnitudes, width, label='Correction Magnitude', alpha=0.8)
            ax.bar(x, mse_values, width, label='MSE', alpha=0.8)
            ax.bar(x + width, mae_values, width, label='MAE', alpha=0.8)
            
            ax.set_xlabel('Model')
            ax.set_ylabel('Value')
            ax.set_title('Pose Optimization Metrics Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(model_names)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_evaluation(self, video_path=None, workout_type=None):
        """
        Run complete evaluation on a video
        
        Args:
            video_path: Path to video file (if None, selects random)
            workout_type: Type of workout (if None, selects random)
        """
        print("=" * 60)
        print("GYMBRO AI MODEL PERFORMANCE EVALUATION")
        print("=" * 60)
        
        # Select video if not provided
        if video_path is None:
            video_path, workout_type = self.select_random_video(workout_type)
        
        # Extract sequence
        sequence = self.extract_sequence_from_video(video_path)
        
        # Evaluate all models
        workout_results = self.evaluate_workout_classifier(sequence, workout_type)
        muscle_results = self.evaluate_muscle_group_classifier(sequence, workout_type)
        pose_results = self.evaluate_pose_optimizer(sequence)
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        self.visualize_workout_predictions(workout_results, workout_type)
        self.visualize_muscle_group_predictions(muscle_results)
        self.visualize_pose_optimization(pose_results)
        
        # Summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Video: {os.path.basename(video_path)}")
        print(f"Workout Type: {workout_type}")
        print(f"Sequence Shape: {sequence.shape}")
        
        print("\nWorkout Classifier Results:")
        for model_name, result in workout_results.items():
            if 'error' not in result:
                print(f"  {model_name}: {result['workout_name']} (confidence: {result['confidence']:.3f})")
        
        print("\nMuscle Group Classifier Results:")
        for model_name, result in muscle_results.items():
            if 'error' not in result:
                print(f"  {model_name}: {result['muscle_group_name']} (confidence: {result['confidence']:.3f})")
        
        print("\nPose Optimizer Results:")
        for model_name, result in pose_results.items():
            if 'error' not in result:
                print(f"  {model_name}: MSE={result['mse']:.6f}, Correction Magnitude={result['correction_magnitude']:.6f}")
        
        return {
            'workout_results': workout_results,
            'muscle_results': muscle_results,
            'pose_results': pose_results,
            'sequence': sequence,
            'video_path': video_path,
            'workout_type': workout_type
        }

def main():
    """Main function to run the evaluation"""
    # Initialize visualizer
    visualizer = ModelPerformanceVisualizer()
    
    # Run evaluation
    results = visualizer.run_complete_evaluation()
    
    print("\nEvaluation completed successfully!")
    return results

if __name__ == "__main__":
    main()
