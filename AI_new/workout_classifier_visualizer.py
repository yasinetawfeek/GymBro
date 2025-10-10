#!/usr/bin/env python3
"""
Workout Classifier Performance Visualizer

This script specifically evaluates and visualizes the performance of workout classifier models:
- LSTM Model
- GRU Model  
- Transformer Model

It takes a random video, extracts a 90-frame sequence, and evaluates all workout classifier models
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

class WorkoutClassifierVisualizer:
    def __init__(self, dataset_path=None):
        """
        Initialize the workout classifier visualizer
        
        Args:
            dataset_path: Path to the workout fitness video dataset
        """
        self.dataset_path = dataset_path or "/Users/yasinetawfeek/.cache/kagglehub/datasets/hasyimabdillah/workoutfitness-video/versions/5"
        self.models_path = os.path.join(os.path.dirname(__file__), 'models', 'workout_classifier')
        
        # Initialize components
        self.mp_handler = MediaPipeHandler()
        self.sequence_processor = SequenceProcessor(sequence_length=90, stride=15)
        
        # Model storage
        self.models = {}
        self.label_encoder = None
        
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
        
        # Load models
        self.load_models()
        
    def load_models(self):
        """Load all trained workout classifier models"""
        print("Loading workout classifier models...")
        
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
        
        # Load label encoder
        label_encoder_path = os.path.join(self.models_path, 'label_encoder.pkl')
        if os.path.exists(label_encoder_path):
            with open(label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print("✓ Loaded workout label encoder")
        
        print(f"Loaded {len(self.models)} workout classifier models")
    
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
    
    def evaluate_models(self, sequence, true_label=None):
        """
        Evaluate all workout classifier models on the sequence
        
        Args:
            sequence: Input sequence of shape (90, 36)
            true_label: True workout label for comparison
            
        Returns:
            Dictionary with predictions and metrics
        """
        print("\n=== WORKOUT CLASSIFIER EVALUATION ===")
        
        results = {}
        
        for model_name, model in self.models.items():
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
                    'all_probabilities': prediction[0],
                    'prediction_vector': prediction[0]
                }
                
                print(f"{model_name.upper()}: {workout_name} (confidence: {confidence:.3f})")
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def visualize_predictions(self, results, true_label=None):
        """Create comprehensive visualizations for workout classifier predictions"""
        if not results:
            print("No results to visualize")
            return
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle('Workout Classifier Performance Analysis', fontsize=20, fontweight='bold')
        
        model_names = list(results.keys())
        model_names = [name for name in model_names if 'error' not in results[name]]
        
        if not model_names:
            print("No valid results to visualize")
            return
        
        # 1. Confidence comparison (top-left)
        ax1 = plt.subplot(3, 4, 1)
        confidences = [results[name].get('confidence', 0) for name in model_names]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        bars = ax1.bar(model_names, confidences, color=colors[:len(model_names)])
        ax1.set_title('Prediction Confidence by Model', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Confidence Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, conf in zip(bars, confidences):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{conf:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Probability distributions (top-right)
        ax2 = plt.subplot(3, 4, 2)
        for i, (model_name, result) in enumerate(results.items()):
            if 'all_probabilities' in result:
                ax2.plot(result['all_probabilities'], label=model_name, 
                        linewidth=2, alpha=0.8, color=colors[i % len(colors)])
        
        ax2.set_title('Probability Distribution Across Classes', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Workout Class Index')
        ax2.set_ylabel('Probability')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Top predictions comparison (top-center-left)
        ax3 = plt.subplot(3, 4, 3)
        top_predictions = {}
        for model_name, result in results.items():
            if 'all_probabilities' in result:
                probs = result['all_probabilities']
                top_indices = np.argsort(probs)[-3:][::-1]  # Top 3 predictions
                top_predictions[model_name] = [(idx, probs[idx]) for idx in top_indices]
        
        # Create horizontal bar chart for top predictions
        y_pos = np.arange(len(model_names))
        for i, model_name in enumerate(model_names):
            if model_name in top_predictions:
                top_pred = top_predictions[model_name][0]  # Best prediction
                workout_name = self.workout_map.get(top_pred[0], f"Class {top_pred[0]}")
                ax3.barh(i, top_pred[1], color=colors[i % len(colors)], alpha=0.7)
                ax3.text(top_pred[1] + 0.01, i, f'{workout_name[:15]}...', 
                        va='center', fontsize=10)
        
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(model_names)
        ax3.set_xlabel('Confidence')
        ax3.set_title('Top Predictions by Model', fontsize=14, fontweight='bold')
        
        # 4. Prediction agreement (top-center-right)
        ax4 = plt.subplot(3, 4, 4)
        predictions = [results[name].get('predicted_class', -1) for name in model_names]
        unique_preds, counts = np.unique(predictions, return_counts=True)
        
        if len(unique_preds) > 1:
            pred_labels = [self.workout_map.get(pred, f"Class {pred}") for pred in unique_preds]
            wedges, texts, autotexts = ax4.pie(counts, labels=pred_labels, autopct='%1.1f%%', 
                                             colors=colors[:len(unique_preds)])
            ax4.set_title('Prediction Agreement', fontsize=14, fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'All models agree!', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=16, fontweight='bold')
            ax4.set_title('Prediction Agreement', fontsize=14, fontweight='bold')
        
        # 5. Confidence distribution histogram (middle-left)
        ax5 = plt.subplot(3, 4, 5)
        ax5.hist(confidences, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax5.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Confidence Score')
        ax5.set_ylabel('Frequency')
        ax5.grid(True, alpha=0.3)
        
        # 6. Model comparison heatmap (middle-center-left)
        ax6 = plt.subplot(3, 4, 6)
        if len(model_names) > 1:
            # Create a comparison matrix
            comparison_matrix = np.zeros((len(model_names), len(model_names)))
            for i, name1 in enumerate(model_names):
                for j, name2 in enumerate(model_names):
                    if i == j:
                        comparison_matrix[i, j] = results[name1]['confidence']
                    else:
                        # Calculate similarity based on prediction agreement
                        pred1 = results[name1]['predicted_class']
                        pred2 = results[name2]['predicted_class']
                        comparison_matrix[i, j] = 1.0 if pred1 == pred2 else 0.0
            
            sns.heatmap(comparison_matrix, annot=True, xticklabels=model_names, 
                       yticklabels=model_names, ax=ax6, cmap='RdYlBu_r')
            ax6.set_title('Model Comparison Matrix', fontsize=14, fontweight='bold')
        
        # 7. Detailed prediction table (middle-center-right)
        ax7 = plt.subplot(3, 4, 7)
        ax7.axis('off')
        
        table_data = []
        for model_name in model_names:
            result = results[model_name]
            table_data.append([
                model_name.upper(),
                result['workout_name'],
                f"{result['confidence']:.3f}",
                f"{result['predicted_class']}"
            ])
        
        table = ax7.table(cellText=table_data,
                         colLabels=['Model', 'Prediction', 'Confidence', 'Class ID'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax7.set_title('Detailed Results', fontsize=14, fontweight='bold')
        
        # 8. Probability heatmap (middle-right)
        ax8 = plt.subplot(3, 4, 8)
        prob_matrix = np.array([results[name]['all_probabilities'] for name in model_names])
        sns.heatmap(prob_matrix, annot=False, xticklabels=False, yticklabels=model_names, 
                   ax=ax8, cmap='viridis')
        ax8.set_title('Probability Heatmap', fontsize=14, fontweight='bold')
        ax8.set_xlabel('Workout Classes')
        
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
        
        # 11. Prediction stability (bottom-center-right)
        ax11 = plt.subplot(3, 4, 11)
        # Calculate prediction stability (how much the top prediction dominates)
        stability_scores = []
        for model_name in model_names:
            probs = results[model_name]['all_probabilities']
            sorted_probs = np.sort(probs)[::-1]
            stability = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
            stability_scores.append(stability)
        
        bars = ax11.bar(model_names, stability_scores, color=colors[:len(model_names)])
        ax11.set_title('Prediction Stability', fontsize=14, fontweight='bold')
        ax11.set_ylabel('Stability Score')
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
        
        ax12.text(0.5, 0.5, f"OVERALL\nASSESSMENT\n\n{assessment}", 
                 ha='center', va='center', transform=ax12.transAxes,
                 fontsize=16, fontweight='bold', color=color)
        
        plt.tight_layout()
        plt.show()
    
    def run_evaluation(self, video_path=None, workout_type=None):
        """
        Run complete workout classifier evaluation
        
        Args:
            video_path: Path to video file (if None, selects random)
            workout_type: Type of workout (if None, selects random)
        """
        print("=" * 60)
        print("WORKOUT CLASSIFIER PERFORMANCE EVALUATION")
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
                print(f"  {model_name}: {result['workout_name']} (confidence: {result['confidence']:.3f})")
        
        return {
            'results': results,
            'sequence': sequence,
            'video_path': video_path,
            'workout_type': workout_type
        }

def main():
    """Main function to run the workout classifier evaluation"""
    # Initialize visualizer
    visualizer = WorkoutClassifierVisualizer()
    
    # Run evaluation
    results = visualizer.run_evaluation()
    
    print("\nWorkout classifier evaluation completed successfully!")
    return results

if __name__ == "__main__":
    main()
