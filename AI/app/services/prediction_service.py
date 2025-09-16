"""
Prediction services for pose correction and workout classification.
"""
import time
import numpy as np
import torch
from collections import deque
from config import (
    BODY_KEYPOINTS_INDICES, WORKOUT_MAP, MUSCLE_GROUP_MAP, 
    WORKOUT_TO_MUSCLE, INFERENCE_THROTTLE
)


class PoseCorrectionService:
    """Service for pose correction predictions."""
    
    def __init__(self, pose_model, device):
        self.pose_model = pose_model
        self.device = device
        self.last_inference_time = 0
    
    def get_pose_corrections(self, landmarks, workout_type=0):
        """Get pose corrections from the model, applying throttling."""
        if self.pose_model is None:
            print("Pose model not loaded, cannot get corrections.")
            return np.zeros(36)

        current_time = time.time()
        time_since_last = current_time - self.last_inference_time
        if time_since_last < INFERENCE_THROTTLE:
            # Throttled: return None to signal skipping the emit
            return None

        try:
            inference_start = time.time()
            
            # Make sure landmarks is a numpy array with 36 elements
            if not isinstance(landmarks, np.ndarray):
                try:
                    landmarks = np.array(landmarks, dtype=np.float32)
                except:
                    print("Error converting landmarks to numpy array")
                    return np.zeros(36)
            
            # Ensure landmarks has the right shape (36,) for the model
            if landmarks.size != 36:
                print(f"Incorrect landmarks size: {landmarks.size} (expected 36)")
                return np.zeros(36)
                
            # Prepare input data (workout type + landmarks)
            # Ensure workout_type is a number
            if not isinstance(workout_type, (int, float)):
                try:
                    workout_type = int(workout_type)
                except:
                    workout_type = 0
                    
            # Combine workout type with landmarks - workout_type is the FIRST element
            input_data = np.append(workout_type, landmarks)
            
            # Convert to tensor and reshape for model
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Perform inference
            with torch.no_grad():
                prediction = self.pose_model(input_tensor)
                corrections = prediction[0].cpu().numpy() # Get numpy array

            inference_time = time.time() - inference_start
            self.last_inference_time = current_time # Update time only when inference actually runs

            # Log if inference takes too long
            if inference_time > 0.1:
                print(f"Slow pose inference: {inference_time:.3f}s")

            # Return the raw correction values
            return corrections
        except Exception as e:
            print(f"Error during pose prediction: {e}")
            # Return zeros in case of prediction error
            return np.zeros(36)


class WorkoutClassificationService:
    """Service for workout classification predictions."""
    
    def __init__(self, workout_classifier, feature_scaler, label_encoder, device):
        self.workout_classifier = workout_classifier
        self.feature_scaler = feature_scaler
        self.label_encoder = label_encoder
        self.device = device
        self.client_pose_buffers = {}
        self.client_workout_predictions = {}
        self.prediction_smoothing_window = 5
    
    def predict_workout_from_sequence(self, client_id, current_features, sequence_length=10):
        """
        Predict workout type using sequence of frames
        
        Args:
            client_id: ID of the client
            current_features: Current frame features (1x36 or 36 values)
            sequence_length: Length of sequence to use for prediction
            
        Returns:
            predicted_workout_type: Integer workout type
            predicted_workout_name: String workout name
        """
        # Default values
        workout_type = 12  # Default to plank
        predicted_workout_name = "plank (default)"
        
        # Check if required components are loaded
        if not all([self.workout_classifier, self.feature_scaler, self.label_encoder]):
            return workout_type, predicted_workout_name
        
        try:
            # Initialize buffer for new clients
            if client_id not in self.client_pose_buffers:
                self.client_pose_buffers[client_id] = deque(maxlen=sequence_length)
                self.client_workout_predictions[client_id] = deque(maxlen=self.prediction_smoothing_window)
            
            # Ensure current_features is properly shaped for StandardScaler
            if isinstance(current_features, np.ndarray):
                # Make sure it's 2D for StandardScaler
                if current_features.ndim == 1:
                    # If it's 1D (36,), reshape to (1, 36)
                    features_to_scale = current_features.reshape(1, -1)
                else:
                    # Already 2D, use as is
                    features_to_scale = current_features
            else:
                # Try to convert to numpy array if it's a list or other type
                try:
                    features_to_scale = np.array(current_features, dtype=np.float32).reshape(1, -1)
                except:
                    print(f"Error reshaping features in predict_workout_from_sequence: {type(current_features)}")
                    # Add default to prediction history
                    self.client_workout_predictions[client_id].append(workout_type)
                    return workout_type, "plank (feature error)"
            
            # Scale the current frame features
            scaled_features = self.feature_scaler.transform(features_to_scale)[0]
            
            # Add to buffer
            self.client_pose_buffers[client_id].append(scaled_features)
            
            # If buffer is not full yet, use default workout
            if len(self.client_pose_buffers[client_id]) < sequence_length:
                # Add placeholder prediction until we have enough frames
                self.client_workout_predictions[client_id].append(workout_type)
                return workout_type, f"plank (collecting sequence: {len(self.client_pose_buffers[client_id])}/{sequence_length})"
            
            # Create sequence tensor from buffer
            sequence = np.array(list(self.client_pose_buffers[client_id]))
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)  # [1, sequence_length, features]
            
            # Make prediction
            with torch.no_grad():
                outputs = self.workout_classifier(sequence_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]  # Get probabilities
                confidence, predicted_idx = torch.max(probabilities, 0)
                predicted_idx = predicted_idx.item()
                confidence = confidence.item()
            
            # Convert index to original label
            predicted_label = self.label_encoder.inverse_transform([predicted_idx])[0]
            
            # Validate prediction
            if predicted_label in WORKOUT_MAP:
                # Add to prediction history
                self.client_workout_predictions[client_id].append(int(predicted_label))
                
                # Get most common prediction from recent history (smoothing)
                workout_counts = {}
                for pred in self.client_workout_predictions[client_id]:
                    workout_counts[pred] = workout_counts.get(pred, 0) + 1
                
                # Find most common prediction
                workout_type = max(workout_counts.items(), key=lambda x: x[1])[0]
                predicted_workout_name = f"{WORKOUT_MAP[workout_type]} (conf: {confidence:.2f})"
            else:
                # Invalid prediction, use default
                workout_type = 12
                predicted_workout_name = "plank (invalid prediction)"
                # Add default to prediction history for continuity
                self.client_workout_predictions[client_id].append(workout_type)
                
            return workout_type, predicted_workout_name
            
        except Exception as e:
            print(f"Error in sequence-based workout prediction for client {client_id}: {e}")
            # Add default to prediction history for continuity
            if client_id in self.client_workout_predictions:
                self.client_workout_predictions[client_id].append(workout_type)
            return workout_type, "plank (prediction error)"


class MuscleGroupClassificationService:
    """Service for muscle group classification predictions."""
    
    def __init__(self, muscle_group_classifier):
        self.muscle_group_classifier = muscle_group_classifier
        self.client_muscle_predictions = {}
        self.prediction_smoothing_window = 5
    
    def predict_muscle_group_from_sequence(self, client_id, current_features, workout_type):
        """
        Predict muscle group using sequence data and workout type
        
        Args:
            client_id: ID of the client
            current_features: Current frame features
            workout_type: Predicted workout type
            
        Returns:
            muscle_group: Integer muscle group
            predicted_muscle_group: String muscle group name
        """
        # Default values
        muscle_group = 0
        predicted_muscle_group = "none (default)"
        
        # If we have a direct mapping, use it
        if workout_type in WORKOUT_TO_MUSCLE:
            muscle_group = WORKOUT_TO_MUSCLE[workout_type]
            
            # Initialize buffer for new clients
            if client_id not in self.client_muscle_predictions:
                self.client_muscle_predictions[client_id] = deque(maxlen=self.prediction_smoothing_window)
            
            # Add to prediction history
            self.client_muscle_predictions[client_id].append(muscle_group)
            
            # Apply smoothing
            muscle_counts = {}
            for pred in self.client_muscle_predictions[client_id]:
                muscle_counts[pred] = muscle_counts.get(pred, 0) + 1
            
            # Find most common prediction
            muscle_group = max(muscle_counts.items(), key=lambda x: x[1])[0]
            predicted_muscle_group = MUSCLE_GROUP_MAP.get(muscle_group, "unknown")
            
            return muscle_group, predicted_muscle_group
        
        # Fall back to muscle group classifier if available
        if self.muscle_group_classifier:
            try:
                # Ensure current_features is properly shaped
                if isinstance(current_features, np.ndarray):
                    # Make sure it's 2D for classifier
                    if current_features.ndim == 1:
                        features_for_muscle = current_features.reshape(1, -1)
                    else:
                        # Already 2D, use as is
                        features_for_muscle = current_features
                else:
                    try:
                        # Try to convert to numpy array
                        features_for_muscle = np.array(current_features, dtype=np.float32).reshape(1, -1)
                    except:
                        print(f"Error reshaping features in predict_muscle_group: {type(current_features)}")
                        return 0, "none (feature error)"
                
                # Predict muscle group
                predicted_muscle_label = self.muscle_group_classifier.predict(features_for_muscle)[0]
                
                # Initialize buffer for new clients
                if client_id not in self.client_muscle_predictions:
                    self.client_muscle_predictions[client_id] = deque(maxlen=self.prediction_smoothing_window)
                
                # Validate prediction
                if predicted_muscle_label in MUSCLE_GROUP_MAP:
                    muscle_group = int(predicted_muscle_label)
                    
                    # Add to prediction history
                    self.client_muscle_predictions[client_id].append(muscle_group)
                    
                    # Apply smoothing
                    muscle_counts = {}
                    for pred in self.client_muscle_predictions[client_id]:
                        muscle_counts[pred] = muscle_counts.get(pred, 0) + 1
                    
                    # Find most common prediction
                    muscle_group = max(muscle_counts.items(), key=lambda x: x[1])[0]
                    predicted_muscle_group = MUSCLE_GROUP_MAP.get(muscle_group, "unknown")
                else:
                    muscle_group = 0
                    predicted_muscle_group = "none (invalid prediction)"
                    # Add default to prediction history
                    self.client_muscle_predictions[client_id].append(muscle_group)
                    
            except Exception as e:
                print(f"Error in muscle group prediction for client {client_id}: {e}")
                # Add default to prediction history
                if client_id in self.client_muscle_predictions:
                    self.client_muscle_predictions[client_id].append(muscle_group)
        
        return muscle_group, predicted_muscle_group