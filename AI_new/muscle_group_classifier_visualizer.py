#!/usr/bin/env python3
"""
Muscle Group Classifier Performance Visualizer

This script specifically evaluates and visualizes the performance of muscle group classifier models:
- LSTM Model
- GRU Model  
- Transformer Model

It takes a random video, extracts a 90-frame sequence, and evaluates all muscle group classifier models
to provide detailed performance metrics and visualizations.
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
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from mediapipe_handler.sequence_processor import SequenceProcessor
from mediapipe_handler.mediapipe_handler import MediaPipeHandler
from muscle_group_mapping import MUSCLE_GROUPS_7, MUSCLE_GROUP_LABELS_7

class MuscleGroupClassifierVisualizer:
    def __init__(self, dataset_path=None):
        """
        Initialize the muscle group classifier visualizer
        
        Args:
            dataset_path: Path to the workout fitness video dataset
        """
        self.dataset_path = dataset_path or "/Users/yasinetawfeek/.cache/kagglehub/datasets/hasyimabdillah/workoutfitness-video/versions/5"
        self.models_path = os.path.join(os.path.dirname(__file__), 'models', 'muscle_group_classifier')
        
        # Initialize components
        self.mp_handler = MediaPipeHandler()
        self.sequence_processor = SequenceProcessor(sequence_length=90, stride=15)
        
        # Model storage
        self.models = {}
        
        # Muscle group mapping
        self.muscle_group_labels = MUSCLE_GROUP_LABELS_7
        self.muscle_group_map = MUSCLE_GROUPS_7
        
        # Workout to muscle group mapping for reference
        self.workout_to_muscle = {
            0: 0,  # barbell bicep curl -> biceps
            1: 1,  # bench press -> chest
            2: 1,  # chest fly machine -> chest
            3: 2,  # deadlift -> back
            4: 0,  # decline bench press -> biceps
            5: 0,  # hammer curl -> biceps
            6: 3,  # hip thrust -> legs
            7: 1,  # incline bench press -> chest
            8: 2,  # lat pulldown -> back
            9: 4,  # lateral raises -> shoulders
            10: 3, # leg extensions -> legs
            11: 5, # leg raises -> core
            12: 5, # plank -> core
            13: 2, # pull up -> back
            14: 1, # push ups -> chest
            15: 3, # romanian deadlift -> legs
            16: 5, # russian twist -> core
            17: 4, # shoulder press -> shoulders
            18: 3, # squat -> legs
            19: 2, # t bar row -> back
            20: 6, # tricep dips -> triceps
            21: 6  # tricep pushdown -> triceps
        }
        
        # Load models
        self.load_models()
        
    def load_models(self):
        """Load all trained muscle group classifier models"""
        print("Loading muscle group classifier models...")
        
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
        
        print(f"Loaded {len(self.models)} muscle group classifier models")
    
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
    
    def evaluate_models(self, sequence, workout_type=None):
        """
        Evaluate all muscle group classifier models on the sequence
        
        Args:
            sequence: Input sequence of shape (90, 36)
            workout_type: Workout type for context and expected muscle group
            
        Returns:
            Dictionary with predictions and metrics
        """
        print("\n=== MUSCLE GROUP CLASSIFIER EVALUATION ===")
        
        results = {}
        
        # Get expected muscle group if workout type is provided
        expected_muscle_group = None
        if workout_type:
            # Map workout type to muscle group
            workout_to_idx = {
                'barbell bicep curl': 0, 'bench press': 1, 'chest fly machine': 2,
                'deadlift': 3, 'decline bench press': 4, 'hammer curl': 5,
                'hip thrust': 6, 'incline bench press': 7, 'lat pulldown': 8,
                'lateral raises': 9, 'leg extensions': 10, 'leg raises': 11,
                'plank': 12, 'pull up': 13, 'push ups': 14, 'romanian deadlift': 15,
                'russian twist': 16, 'shoulder press': 17, 'squat': 18,
                't bar row': 19, 'tricep dips': 20, 'tricep pushdown': 21
            }
            
            workout_idx = workout_to_idx.get(workout_type.lower(), None)
            if workout_idx is not None:
                expected_muscle_group = self.workout_to_muscle.get(workout_idx, None)
                if expected_muscle_group is not None:
                    print(f"Expected muscle group for '{workout_type}': {self.muscle_group_labels[expected_muscle_group]}")
        
        for model_name, model in self.models.items():
            try:
                # Prepare input
                input_sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
                
                # Make prediction
                prediction = model.predict(input_sequence, verbose=0)
                predicted_class = np.argmax(prediction[0])
                confidence = np.max(prediction[0])
                
                # Get muscle group name
                muscle_group_name = self.muscle_group_labels[predicted_class]
                
                # Check if prediction matches expected
                is_correct = (expected_muscle_group == predicted_class) if expected_muscle_group is not None else None
                
                results[model_name] = {
                    'predicted_class': predicted_class,
                    'muscle_group_name': muscle_group_name,
                    'confidence': confidence,
                    'all_probabilities': prediction[0],
                    'prediction_vector': prediction[0],
                    'is_correct': is_correct,
                    'expected_muscle_group': expected_muscle_group
                }
                
                status = ""
                if is_correct is True:
                    status = " ✓ CORRECT"
                elif is_correct is False:
                    status = " ✗ INCORRECT"
                
                print(f"{model_name.upper()}: {muscle_group_name} (confidence: {confidence:.3f}){status}")
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def visualize_predictions(self, results, workout_type=None):
        """Create comprehensive visualizations for muscle group classifier predictions"""
        if not results:
            print("No results to visualize")
            return
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle('Muscle Group Classifier Performance Analysis', fontsize=20, fontweight='bold')
        
        model_names = list(results.keys())
        model_names = [name for name in model_names if 'error' not in results[name]]
        
        if not model_names:
            print("No valid results to visualize")
            return
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # 1. Confidence comparison (top-left)
        ax1 = plt.subplot(3, 4, 1)
        confidences = [results[name].get('confidence', 0) for name in model_names]
        bars = ax1.bar(model_names, confidences, color=colors[:len(model_names)])
        ax1.set_title('Prediction Confidence by Model', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Confidence Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, conf in zip(bars, confidences):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{conf:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Probability distributions across muscle groups (top-right)
        ax2 = plt.subplot(3, 4, 2)
        for i, (model_name, result) in enumerate(results.items()):
            if 'all_probabilities' in result:
                ax2.plot(result['all_probabilities'], label=model_name, 
                        linewidth=2, alpha=0.8, color=colors[i % len(colors)])
        
        ax2.set_title('Probability Distribution Across Muscle Groups', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Muscle Group Index')
        ax2.set_ylabel('Probability')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(len(self.muscle_group_labels)))
        ax2.set_xticklabels(self.muscle_group_labels, rotation=45)
        
        # 3. Muscle group predictions comparison (top-center-left)
        ax3 = plt.subplot(3, 4, 3)
        predictions = [results[name].get('muscle_group_name', 'Error') for name in model_names]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(model_names))
        muscle_colors = {}
        for i, muscle in enumerate(self.muscle_group_labels):
            muscle_colors[muscle] = colors[i % len(colors)]
        
        bar_colors = [muscle_colors.get(pred, 'gray') for pred in predictions]
        bars = ax3.barh(y_pos, confidences, color=bar_colors, alpha=0.7)
        
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(model_names)
        ax3.set_xlabel('Confidence')
        ax3.set_title('Muscle Group Predictions', fontsize=14, fontweight='bold')
        
        # Add muscle group labels
        for i, (bar, pred) in enumerate(zip(bars, predictions)):
            ax3.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    pred, va='center', fontsize=10)
        
        # 4. Prediction agreement pie chart (top-center-right)
        ax4 = plt.subplot(3, 4, 4)
        unique_preds, counts = np.unique(predictions, return_counts=True)
        
        if len(unique_preds) > 1:
            wedges, texts, autotexts = ax4.pie(counts, labels=unique_preds, autopct='%1.1f%%', 
                                             colors=[muscle_colors.get(pred, 'gray') for pred in unique_preds])
            ax4.set_title('Prediction Agreement', fontsize=14, fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'All models agree!', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=16, fontweight='bold')
            ax4.set_title('Prediction Agreement', fontsize=14, fontweight='bold')
        
        # 5. Muscle group distribution histogram (middle-left)
        ax5 = plt.subplot(3, 4, 5)
        muscle_counts = {}
        for pred in predictions:
            muscle_counts[pred] = muscle_counts.get(pred, 0) + 1
        
        muscles = list(muscle_counts.keys())
        counts = list(muscle_counts.values())
        muscle_colors_list = [muscle_colors.get(muscle, 'gray') for muscle in muscles]
        
        bars = ax5.bar(muscles, counts, color=muscle_colors_list, alpha=0.7)
        ax5.set_title('Muscle Group Distribution', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Number of Predictions')
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. Accuracy analysis (middle-center-left)
        ax6 = plt.subplot(3, 4, 6)
        correct_predictions = [results[name].get('is_correct', None) for name in model_names]
        correct_count = sum(1 for c in correct_predictions if c is True)
        incorrect_count = sum(1 for c in correct_predictions if c is False)
        unknown_count = sum(1 for c in correct_predictions if c is None)
        
        if correct_count + incorrect_count > 0:
            accuracy_data = [correct_count, incorrect_count, unknown_count]
            accuracy_labels = ['Correct', 'Incorrect', 'Unknown']
            accuracy_colors = ['green', 'red', 'gray']
            
            wedges, texts, autotexts = ax6.pie(accuracy_data, labels=accuracy_labels, 
                                             colors=accuracy_colors, autopct='%1.1f%%')
            ax6.set_title('Prediction Accuracy', fontsize=14, fontweight='bold')
        else:
            ax6.text(0.5, 0.5, 'No Ground Truth\nAvailable', ha='center', va='center', 
                    transform=ax6.transAxes, fontsize=14, fontweight='bold')
            ax6.set_title('Prediction Accuracy', fontsize=14, fontweight='bold')
        
        # 7. Detailed results table (middle-center-right)
        ax7 = plt.subplot(3, 4, 7)
        ax7.axis('off')
        
        table_data = []
        for model_name in model_names:
            result = results[model_name]
            status = ""
            if result.get('is_correct') is True:
                status = "✓"
            elif result.get('is_correct') is False:
                status = "✗"
            else:
                status = "?"
            
            table_data.append([
                model_name.upper(),
                result['muscle_group_name'],
                f"{result['confidence']:.3f}",
                status
            ])
        
        table = ax7.table(cellText=table_data,
                         colLabels=['Model', 'Muscle Group', 'Confidence', 'Correct'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax7.set_title('Detailed Results', fontsize=14, fontweight='bold')
        
        # 8. Probability heatmap (middle-right)
        ax8 = plt.subplot(3, 4, 8)
        prob_matrix = np.array([results[name]['all_probabilities'] for name in model_names])
        sns.heatmap(prob_matrix, annot=False, xticklabels=self.muscle_group_labels, 
                   yticklabels=model_names, ax=ax8, cmap='viridis')
        ax8.set_title('Probability Heatmap', fontsize=14, fontweight='bold')
        ax8.set_xlabel('Muscle Groups')
        
        # 9. Confidence vs Model scatter (bottom-left)
        ax9 = plt.subplot(3, 4, 9)
        x_pos = range(len(model_names))
        ax9.scatter(x_pos, confidences, s=100, c=colors[:len(model_names)], alpha=0.7)
        ax9.plot(x_pos, confidences, '--', alpha=0.5)
        ax9.set_xticks(x_pos)
        ax9.set_xticklabels(model_names, rotation=45)
        ax9.set_ylabel('Confidence')
        ax9.set_title('Confidence Trend', fontsize=14, fontweight='bold')
        ax9.grid(True, alpha=0.3)
        
        # 10. Model performance summary (bottom-center-left)
        ax10 = plt.subplot(3, 4, 10)
        ax10.axis('off')
        
        # Calculate summary statistics
        avg_confidence = np.mean(confidences)
        max_confidence = np.max(confidences)
        min_confidence = np.min(confidences)
        std_confidence = np.std(confidences)
        
        summary_text = f"""
        SUMMARY STATISTICS
        
        Average Confidence: {avg_confidence:.3f}
        Max Confidence: {max_confidence:.3f}
        Min Confidence: {min_confidence:.3f}
        Std Deviation: {std_confidence:.3f}
        
        Best Model: {model_names[np.argmax(confidences)]}
        Worst Model: {model_names[np.argmin(confidences)]}
        
        Total Models: {len(model_names)}
        """
        
        ax10.text(0.1, 0.5, summary_text, transform=ax10.transAxes, fontsize=12,
                 verticalalignment='center', fontfamily='monospace')
        ax10.set_title('Performance Summary', fontsize=14, fontweight='bold')
        
        # 11. Muscle group confidence analysis (bottom-center-right)
        ax11 = plt.subplot(3, 4, 11)
        
        # Group confidences by muscle group
        muscle_confidences = {}
        for model_name in model_names:
            result = results[model_name]
            muscle = result['muscle_group_name']
            conf = result['confidence']
            if muscle not in muscle_confidences:
                muscle_confidences[muscle] = []
            muscle_confidences[muscle].append(conf)
        
        # Calculate average confidence per muscle group
        avg_confidences = {muscle: np.mean(confs) for muscle, confs in muscle_confidences.items()}
        
        muscles = list(avg_confidences.keys())
        avg_confs = list(avg_confidences.values())
        muscle_colors_list = [muscle_colors.get(muscle, 'gray') for muscle in muscles]
        
        bars = ax11.bar(muscles, avg_confs, color=muscle_colors_list, alpha=0.7)
        ax11.set_title('Avg Confidence by Muscle Group', fontsize=14, fontweight='bold')
        ax11.set_ylabel('Average Confidence')
        ax11.tick_params(axis='x', rotation=45)
        
        # 12. Overall assessment (bottom-right)
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')
        
        # Determine overall assessment
        if max_confidence > 0.8:
            assessment = "EXCELLENT"
            color = 'green'
        elif max_confidence > 0.6:
            assessment = "GOOD"
            color = 'orange'
        else:
            assessment = "NEEDS IMPROVEMENT"
            color = 'red'
        
        # Add accuracy info if available
        accuracy_info = ""
        if correct_count + incorrect_count > 0:
            accuracy = correct_count / (correct_count + incorrect_count) * 100
            accuracy_info = f"\nAccuracy: {accuracy:.1f}%"
        
        ax12.text(0.5, 0.5, f"OVERALL\nASSESSMENT\n\n{assessment}{accuracy_info}", 
                 ha='center', va='center', transform=ax12.transAxes,
                 fontsize=16, fontweight='bold', color=color)
        
        plt.tight_layout()
        plt.show()
    
    def run_evaluation(self, video_path=None, workout_type=None):
        """
        Run complete muscle group classifier evaluation
        
        Args:
            video_path: Path to video file (if None, selects random)
            workout_type: Type of workout (if None, selects random)
        """
        print("=" * 60)
        print("MUSCLE GROUP CLASSIFIER PERFORMANCE EVALUATION")
        print("=" * 60)
        
        # Select video if not provided
        if video_path is None:
            video_path, workout_type = self.select_random_video(workout_type)
        
        # Extract sequence
        sequence = self.extract_sequence_from_video(video_path)
        
        # Evaluate models
        results = self.evaluate_models(sequence, workout_type)
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        self.visualize_predictions(results, workout_type)
        
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
                status = ""
                if result.get('is_correct') is True:
                    status = " ✓ CORRECT"
                elif result.get('is_correct') is False:
                    status = " ✗ INCORRECT"
                print(f"  {model_name}: {result['muscle_group_name']} (confidence: {result['confidence']:.3f}){status}")
        
        return {
            'results': results,
            'sequence': sequence,
            'video_path': video_path,
            'workout_type': workout_type
        }

def main():
    """Main function to run the muscle group classifier evaluation"""
    # Initialize visualizer
    visualizer = MuscleGroupClassifierVisualizer()
    
    # Run evaluation
    results = visualizer.run_evaluation()
    
    print("\nMuscle group classifier evaluation completed successfully!")
    return results

if __name__ == "__main__":
    main()
