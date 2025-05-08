# --- Eventlet Monkey Patching (MUST be first) ---
# Attempt to import and patch *before* other imports like Flask, SocketIO, etc.
try:
    import eventlet
    eventlet.monkey_patch()
    print("Eventlet monkey patching applied successfully.")
    USING_EVENTLET = True
except ImportError:
    print("Eventlet not found, monkey patching skipped.")
    USING_EVENTLET = False
except Exception as e:
    print(f"An exception occurred during eventlet monkey patching: {e}")
    USING_EVENTLET = False # Assume patching failed

# --- Standard Imports ---
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import os
import pickle # Added for loading the classifier
from flask_socketio import SocketIO, emit
import json
import time
from datetime import datetime # Import datetime for formatted timestamp
from sklearn.preprocessing import StandardScaler # Added for feature scaling
from collections import deque # For sequence buffer management
import requests
import uuid

# --- Constants ---
# Body keypoints indices from MediaPipe BlazePose (the key joints we're tracking)
BODY_KEYPOINTS_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
LANDMARK_DIM = 3  # x, y, z
FLAT_LANDMARK_SIZE = len(BODY_KEYPOINTS_INDICES) * LANDMARK_DIM  # Should be 36

# --- Config for sequence models ---
# This will be populated when models are loaded
workout_classifier_config = {
    'sequence_length': 10  # Default value, may be overridden when model loads
}

# --- DNN model class definition (Unchanged) ---
class EnhancedPoseModel(nn.Module):
    def __init__(self, input_dim=37, hidden_dim=512, output_dim=36):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Model outputs values in range [-0.1, 0.1]
        return self.net(x) * 0.1

# --- LSTM Workout Classifier model definition (UPDATED for sequences) ---
class LSTMWorkoutClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super(LSTMWorkoutClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers with increased dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
        
        # Additional dropout after LSTM
        self.lstm_dropout = nn.Dropout(dropout + 0.1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 128)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Ensure x has the right shape for sequences
        # x should be [batch_size, sequence_length, features]
        if len(x.shape) == 2:
            # If single frame, reshape to [batch, 1, features]
            x = x.unsqueeze(1)
        
        batch_size, seq_len, _ = x.size()
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))  # lstm_out: [batch_size, seq_len, hidden_size]
        
        # Apply additional dropout to LSTM output
        lstm_out = self.lstm_dropout(lstm_out)
        
        # Apply attention mechanism if sequence length > 1
        if seq_len > 1:
            attention_weights = self.attention(lstm_out)  # [batch_size, seq_len, 1]
            context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # [batch_size, hidden_size]
        else:
            # If single frame, just use the LSTM output directly
            context_vector = lstm_out.squeeze(1)
        
        # Dense layers
        out = self.fc1(context_vector)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

# --- Workout type mapping (Unchanged) ---
workout_map = { 0: "barbell bicep curl", 1: "bench press", 2: "chest fly machine", 3: "deadlift", 4: "decline bench press", 5: "hammer curl", 6: "hip thrust", 7: "incline bench press", 8: "lat pulldown", 9: "lateral raises", 10: "leg extensions", 11: "leg raises", 12: "plank", 13: "pull up", 14: "push ups", 15: "romanian deadlift", 16: "russian twist", 17: "shoulder press", 18: "squat", 19: "t bar row", 20: "tricep dips", 21: "tricep pushdown" }

# --- Initialize Flask app and SocketIO ---
app = Flask(__name__)
# Pass async_mode explicitly based on whether eventlet was successfully patched
async_mode = 'eventlet' if USING_EVENTLET else None
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=10, ping_interval=5, async_mode=async_mode)
print(f"SocketIO initialized with async_mode: {socketio.async_mode}")


# --- Global variables ---
pose_model = None
workout_classifier = None # Global variable for the LSTM workout classifier
feature_scaler = None # Global variable for the feature scaler
label_encoder = None # Global variable for the label encoder
muscle_group_classifier = None # Global variable for the muscle group classifier
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
last_inference_time = 0
INFERENCE_THROTTLE = 0.05 # 50ms throttle

# --- Sequence-related global variables (NEW) ---
SEQUENCE_LENGTH = 10  # Number of frames to use for sequence
client_pose_buffers = {}  # Dictionary to store pose buffers for each client
client_workout_predictions = {}  # Store recent workout predictions for smoothing
client_muscle_predictions = {}  # Store recent muscle group predictions for smoothing
prediction_smoothing_window = 5  # Number of predictions to average for smoothing

# Global tracking dictionaries
client_sessions = {}
performance_metrics = {}
model_version = "1.0.0"  # Update this when you deploy new models

# Initialize API endpoint URLs
BACKEND_BASE_URL = 'http://localhost:8000'
USAGE_ENDPOINT = f"{BACKEND_BASE_URL}/api/usage/"
PERFORMANCE_ENDPOINT = f"{BACKEND_BASE_URL}/api/model-performance/"

# Helper function to verify user token
def verify_token(token):
    """Verify user token with Django backend"""
    try:
        response = requests.post(
            f"{BACKEND_BASE_URL}/api/token/verify/", 
            json={"token": token},
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"Error verifying token: {str(e)}")
        return None

# Initialize session tracking for a new client
def initialize_session_tracking(client_id, user_id=None, token=None, workout_type=0):
    """Initialize session tracking for a new client"""
    client_sessions[client_id] = {
        'user_id': user_id,
        'token': token,
        'session_id': str(uuid.uuid4()),
        'start_time': time.time(),
        'is_authenticated': user_id is not None,
        'frames_processed': 0,
        'corrections_sent': 0,
        'last_activity': time.time(),
        'workout_type': workout_type,
        'session_recorded': False
    }
    
    # Initialize performance metrics
    performance_metrics[client_id] = {
        'confidence_values': [],
        'correction_magnitudes': [],
        'response_times': [],
        'processing_times': [],
        'first_correction_time': None,
        'frames_per_second': [],
        'prediction_changes': 0,
        'last_prediction': None,
        'prediction_counts': {},
        'frame_count': 0,
        'start_time': time.time()
    }
    
    # If user is authenticated, start a session in the backend
    if user_id and token:
        try:
            response = requests.post(
                f"{USAGE_ENDPOINT}start_session/",
                json={"workout_type": workout_type, "platform": "web"},
                headers={"Authorization": f"Bearer {token}"},
                timeout=5
            )
            if response.status_code == 201:
                data = response.json()
                client_sessions[client_id]['session_id'] = data['session_id']
                print(f"Session started in backend: {data['session_id']}")
            else:
                print(f"Failed to start session: {response.text}")
        except Exception as e:
            print(f"Error starting session: {str(e)}")

# Update client session data
def update_session_data(client_id, frames=1, corrections=1, workout_type=None):
    """Update session tracking data"""
    if client_id in client_sessions:
        client_sessions[client_id]['frames_processed'] += frames
        client_sessions[client_id]['corrections_sent'] += corrections
        client_sessions[client_id]['last_activity'] = time.time()
        
        if workout_type is not None:
            client_sessions[client_id]['workout_type'] = workout_type
        
        # Periodically update metrics in the backend
        if (client_sessions[client_id]['corrections_sent'] % 50 == 0 and 
            client_sessions[client_id]['is_authenticated']):
            report_session_metrics(client_id)

# Update performance metrics
def update_performance_metrics(client_id, confidence=None, correction_magnitude=None, 
                              response_time=None, processing_time=None, predicted_type=None):
    """Update performance metrics for the client session"""
    if client_id not in performance_metrics:
        return
    
    metrics = performance_metrics[client_id]
    
    # Update frame count
    metrics['frame_count'] += 1
    
    # Calculate frames per second
    elapsed = time.time() - metrics['start_time']
    if elapsed > 0:
        fps = metrics['frame_count'] / elapsed
        metrics['frames_per_second'].append(fps)
    
    # Track confidence
    if confidence is not None:
        metrics['confidence_values'].append(confidence)
    
    # Track correction magnitude
    if correction_magnitude is not None:
        metrics['correction_magnitudes'].append(correction_magnitude)
    
    # Track response time
    if response_time is not None:
        metrics['response_times'].append(response_time)
    
    # Track processing time
    if processing_time is not None:
        metrics['processing_times'].append(processing_time)
    
    # Track time to first correction
    if metrics['first_correction_time'] is None and correction_magnitude is not None:
        metrics['first_correction_time'] = time.time() - metrics['start_time']
    
    # Track prediction stability
    if predicted_type is not None:
        # Count occurrences of each prediction
        metrics['prediction_counts'][predicted_type] = metrics['prediction_counts'].get(predicted_type, 0) + 1
        
        # Check for prediction changes
        if metrics['last_prediction'] is not None and metrics['last_prediction'] != predicted_type:
            metrics['prediction_changes'] += 1
        
        metrics['last_prediction'] = predicted_type
    
    # Periodically report performance metrics
    if metrics['frame_count'] % 1000 == 0:
        report_performance_metrics(client_id)

# Report session metrics to the backend
def report_session_metrics(client_id):
    """Report current session metrics to Django backend"""
    if client_id not in client_sessions or not client_sessions[client_id]['is_authenticated']:
        return
    
    session = client_sessions[client_id]
    token = session['token']
    
    try:
        response = requests.post(
            f"{USAGE_ENDPOINT}update_metrics/",
            json={
                "session_id": session['session_id'],
                "frames_processed": session['frames_processed'],
                "corrections_sent": session['corrections_sent']
            },
            headers={"Authorization": f"Bearer {token}"},
            timeout=5
        )
        
        if response.status_code != 200:
            print(f"Failed to update metrics: {response.text}")
    except Exception as e:
        print(f"Error updating metrics: {str(e)}")

# End client session and report final metrics
def end_client_session(client_id):
    """End client session and report final metrics"""
    if client_id not in client_sessions:
        return
    
    session = client_sessions[client_id]
    
    # Only report to backend if authenticated
    if session['is_authenticated'] and not session['session_recorded']:
        token = session['token']
        
        try:
            response = requests.post(
                f"{USAGE_ENDPOINT}end_session/",
                json={
                    "session_id": session['session_id'],
                    "frames_processed": session['frames_processed'],
                    "corrections_sent": session['corrections_sent']
                },
                headers={"Authorization": f"Bearer {token}"},
                timeout=5
            )
            
            if response.status_code == 200:
                session['session_recorded'] = True
                print(f"Session ended and recorded: {session['session_id']}")
            else:
                print(f"Failed to end session: {response.text}")
        except Exception as e:
            print(f"Error ending session: {str(e)}")
    
    # Report performance metrics before cleanup
    if client_id in performance_metrics:
        report_performance_metrics(client_id, is_final=True)
        del performance_metrics[client_id]
    
    # Clean up session data
    del client_sessions[client_id]

# Calculate and report performance metrics
def report_performance_metrics(client_id, is_final=False):
    """Calculate and report performance metrics to the backend"""
    if client_id not in performance_metrics:
        return
    
    metrics = performance_metrics[client_id]
    
    # Skip if not enough data
    if len(metrics['confidence_values']) < 10:
        return
    
    # Calculate average metrics
    avg_confidence = sum(metrics['confidence_values']) / len(metrics['confidence_values']) if metrics['confidence_values'] else 0
    min_confidence = min(metrics['confidence_values']) if metrics['confidence_values'] else 0
    max_confidence = max(metrics['confidence_values']) if metrics['confidence_values'] else 0
    
    avg_correction = sum(metrics['correction_magnitudes']) / len(metrics['correction_magnitudes']) if metrics['correction_magnitudes'] else 0
    
    avg_response = sum(metrics['response_times']) / len(metrics['response_times']) if metrics['response_times'] else 0
    avg_processing = sum(metrics['processing_times']) / len(metrics['processing_times']) if metrics['processing_times'] else 0
    
    # Calculate frames per second
    avg_fps = sum(metrics['frames_per_second']) / len(metrics['frames_per_second']) if metrics['frames_per_second'] else 0
    
    # Calculate stability rate
    total_predictions = sum(metrics['prediction_counts'].values())
    stability_rate = 0
    if total_predictions > 0:
        # Calculate what percentage of frames didn't cause a prediction change
        stability_rate = 1.0 - (metrics['prediction_changes'] / total_predictions)
    
    # Get the workout type from the session
    workout_type = client_sessions.get(client_id, {}).get('workout_type', 0)
    
    # Prepare metric data
    metric_data = {
        "model_version": model_version,
        "workout_type": workout_type,
        "avg_prediction_confidence": avg_confidence,
        "min_prediction_confidence": min_confidence,
        "max_prediction_confidence": max_confidence,
        "correction_magnitude_avg": avg_correction,
        "stable_prediction_rate": stability_rate,
        "avg_response_latency": int(avg_response),
        "processing_time_per_frame": int(avg_processing),
        "time_to_first_correction": int(metrics['first_correction_time'] * 1000) if metrics['first_correction_time'] else 0,
        "frame_processing_rate": avg_fps
    }
    
    # Only send to backend if we have a user ID and token
    user_id = client_sessions.get(client_id, {}).get('user_id')
    token = client_sessions.get(client_id, {}).get('token')
    
    if user_id and token:
        try:
            # First check if the user has admin or ML expert role before trying to send metrics
            # This avoids users getting 403 errors which could affect their experience
            role_check_response = requests.get(
                "http://localhost:8000/api/role-info/",
                headers={"Authorization": f"Bearer {token}"},
                timeout=5
            )
            
            if role_check_response.status_code == 200:
                role_data = role_check_response.json()
                is_admin = role_data.get('is_admin', False)
                is_ai_engineer = role_data.get('is_ai_engineer', False)
                
                # Only proceed if user is admin or AI engineer
                if is_admin or is_ai_engineer:
                    response = requests.post(
                        f"{PERFORMANCE_ENDPOINT}record_metrics/",
                        json=metric_data,
                        headers={"Authorization": f"Bearer {token}"},
                        timeout=5
                    )
                    
                    if response.status_code == 201:
                        print(f"Performance metrics recorded for workout type {workout_type}")
                    else:
                        print(f"Failed to record performance metrics: {response.text}")
                else:
                    # Debug output but don't attempt to record for non-admin/non-ML users
                    if is_final:
                        print(f"Skipping performance metrics record - user {user_id} is not admin or AI engineer")
            else:
                print(f"Couldn't check user role: {role_check_response.text}")
        except Exception as e:
            print(f"Error recording performance metrics: {str(e)}")
    
    # If this is the final report, clear metrics
    if is_final:
        metrics.clear()

# --- Model Loading ---
def find_file(filename, search_paths):
    """Helper function to find a file in a list of paths."""
    # Check absolute paths first (like /data)
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename
    # Check search paths relative to CWD
    for path in search_paths:
        # Check relative path directly
        full_path = os.path.join(path, filename)
        if os.path.exists(full_path):
            print(f"Found '{filename}' at: {full_path}")
            return full_path
        # Check path relative to script's directory (if different from CWD)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_relative_path = os.path.join(script_dir, path, filename)
        if os.path.exists(script_relative_path):
             print(f"Found '{filename}' at: {script_relative_path}")
             return script_relative_path
    print(f"Could not find '{filename}' in search paths: {search_paths}")
    return None


def load_pose_model():
    """Load the DNN model at startup"""
    global pose_model
    try:
        # Define potential base directories relative to the script or common structures
        possible_base_dirs = ['.', '..', '../..', 'AI']
        # Construct full search paths
        search_paths = [os.path.join(base, 'data') for base in possible_base_dirs] + ['data', '/data'] # Add /data for container environments

        model_path = find_file('best_model.pth', search_paths)

        if model_path is None:
            print(f"Error: Could not find pose model file 'best_model.pth'")
            print(f"Current working directory: {os.getcwd()}")
            return False

        pose_model = EnhancedPoseModel(input_dim=37, hidden_dim=512, output_dim=36).to(device)
        pose_model.load_state_dict(torch.load(model_path, map_location=device))
        pose_model.eval()
        print(f"DNN pose model loaded successfully from {model_path}")
        if torch.cuda.is_available():
            print("Running warmup inference for CUDA initialization...")
            dummy_input = torch.zeros(1, 37, device=device)
            with torch.no_grad():
                _ = pose_model(dummy_input) # Assign to dummy variable
        return True
    except Exception as e:
        print(f"Error loading DNN pose model: {e}")
        return False

def load_workout_classifier():
    """Load the LSTM workout classifier model and supporting files"""
    global workout_classifier, feature_scaler, label_encoder
    try:
        # Define potential base directories relative to the script or common structures
        possible_base_dirs = ['.', '..', '../..', 'AI']
        # Construct full search paths
        search_paths = [os.path.join(base, 'data', 'models') for base in possible_base_dirs] + \
                       [os.path.join(base, 'data') for base in possible_base_dirs] + \
                       ['data/models', 'data', '/data/models', '/data']  # Add /data for container environments

        # 1. Find and load the LSTM model file - first check for the sequential version
        model_path = find_file('lstm_workout_classifier_sequential_v2.pth', search_paths)
        if model_path is None:
            # Try to load the original model as fallback
            model_path = find_file('lstm_workout_classifier.pth', search_paths)
            
        if model_path is None:
            # Try to load best checkpoint file if main model not found
            model_path = find_file('lstm_workout_best_checkpoint.pth', search_paths)
            
        if model_path is None:
            print(f"Error: Could not find LSTM workout classifier model files")
            print(f"Current working directory: {os.getcwd()}")
            return False

        # 2. Find and load the feature scaler
        scaler_path = find_file('feature_scaler.pkl', search_paths)
        if scaler_path is None:
            print(f"Error: Could not find feature scaler file")
            return False

        # 3. Find and load the label encoder
        encoder_path = find_file('label_encoder.pkl', search_paths)
        if encoder_path is None:
            print(f"Error: Could not find label encoder file")
            return False

        # Load the model checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Check if this is a complete model or just state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # This is a checkpoint with metadata
            input_size = checkpoint.get('input_size', 36)
            hidden_size = checkpoint.get('hidden_size', 128)
            num_layers = checkpoint.get('num_layers', 2)
            num_classes = checkpoint.get('num_classes', 22)
            dropout_rate = checkpoint.get('dropout_rate', 0.3)
            
            # Initialize model with parameters from checkpoint
            workout_classifier = LSTMWorkoutClassifier(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout=dropout_rate
            ).to(device)
            
            # Load the state dict
            workout_classifier.load_state_dict(checkpoint['model_state_dict'])
            
            # Check if the model has sequence_length parameter
            if 'sequence_length' in checkpoint:
                global SEQUENCE_LENGTH
                SEQUENCE_LENGTH = checkpoint['sequence_length']
                print(f"Using sequence length from model: {SEQUENCE_LENGTH}")
        else:
            # Assume it's just a state dict with default parameters
            workout_classifier = LSTMWorkoutClassifier(
                input_size=36,  # Default input size
                hidden_size=128,
                num_layers=2,
                num_classes=22,
                dropout=0.3
            ).to(device)
            workout_classifier.load_state_dict(checkpoint)
        
        # Set model to evaluation mode
        workout_classifier.eval()
        print(f"LSTM workout classifier loaded successfully from {model_path}")

        # Load feature scaler
        with open(scaler_path, 'rb') as f:
            feature_scaler = pickle.load(f)
        print(f"Feature scaler loaded successfully from {scaler_path}")

        # Load label encoder
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        print(f"Label encoder loaded successfully from {encoder_path}")

        # Run warmup inference with sequence
        if torch.cuda.is_available():
            print("Running warmup inference for LSTM classifier...")
            # Create a dummy sequence for warmup
            dummy_sequence = torch.zeros(1, SEQUENCE_LENGTH, 36, device=device)
            with torch.no_grad():
                _ = workout_classifier(dummy_sequence)

        return True
    except FileNotFoundError as e:
        print(f"Error: File not found when loading LSTM workout classifier: {e}")
        return False
    except Exception as e:
        print(f"Error loading LSTM workout classifier: {e}")
        return False

# --- Load muscle group classifier ---
def load_muscle_group_classifier():
    """Load the muscle group classifier model (.pkl) at startup"""
    global muscle_group_classifier
    try:
        # Define potential base directories relative to the script or common structures
        possible_base_dirs = ['.', '..', '../..', 'AI']
         # Construct full search paths, including a 'models' subdirectory as seen in training script
        search_paths = [os.path.join(base, 'data', 'models') for base in possible_base_dirs] + \
                       [os.path.join(base, 'data') for base in possible_base_dirs] + \
                       ['data/models', 'data', '/data/models', '/data'] # Add /data for container environments

        classifier_path = find_file('rfc_muscle_group_classifier.pkl', search_paths)

        if classifier_path is None:
            print(f"Error: Could not find muscle group classifier file 'rfc_muscle_group_classifier.pkl'")
            print(f"Current working directory: {os.getcwd()}")
            return False

        with open(classifier_path, 'rb') as f:
            muscle_group_classifier = pickle.load(f)
        print(f"Muscle group classifier loaded successfully from {classifier_path}")
        return True
    except FileNotFoundError:
        print(f"Error: Muscle group classifier file not found at expected paths.")
        return False
    except pickle.UnpicklingError as e:
         print(f"Error unpickling muscle group classifier model: {e}")
         return False
    except Exception as e:
        print(f"Error loading muscle group classifier model: {e}")
        return False

# --- Pose Correction Logic (Unchanged) ---
def get_pose_corrections(landmarks, workout_type=0):
    """Get pose corrections from the model, applying throttling."""
    global pose_model, last_inference_time
    if pose_model is None:
        print("Pose model not loaded, cannot get corrections.")
        # Return zeros if model cannot be loaded
        return np.zeros(36)

    current_time = time.time()
    time_since_last = current_time - last_inference_time
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
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            prediction = pose_model(input_tensor)
            corrections = prediction[0].cpu().numpy() # Get numpy array

        inference_time = time.time() - inference_start
        last_inference_time = current_time # Update time only when inference actually runs

        # Log if inference takes too long
        if inference_time > 0.1:
            print(f"Slow pose inference: {inference_time:.3f}s")

        # Return the raw correction values
        return corrections
    except Exception as e:
        print(f"Error during pose prediction: {e}")
        # Return zeros in case of prediction error
        return np.zeros(36)

# --- NEW: Sequence-based workout prediction function ---
def predict_workout_from_sequence(client_id, current_features):
    """
    Predict workout type using sequence of frames
    
    Args:
        client_id: ID of the client
        current_features: Current frame features (1x36 or 36 values)
        
    Returns:
        predicted_workout_type: Integer workout type
        predicted_workout_name: String workout name
    """
    # Default values
    workout_type = 12  # Default to plank
    predicted_workout_name = "plank (default)"
    
    # Check if required components are loaded
    if not all([workout_classifier, feature_scaler, label_encoder]):
        return workout_type, predicted_workout_name
    
    try:
        # Initialize buffer for new clients
        if client_id not in client_pose_buffers:
            client_pose_buffers[client_id] = deque(maxlen=SEQUENCE_LENGTH)
            client_workout_predictions[client_id] = deque(maxlen=prediction_smoothing_window)
        
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
                client_workout_predictions[client_id].append(workout_type)
                return workout_type, "plank (feature error)"
        
        # Scale the current frame features
        scaled_features = feature_scaler.transform(features_to_scale)[0]
        
        # Add to buffer
        client_pose_buffers[client_id].append(scaled_features)
        
        # If buffer is not full yet, use default workout
        if len(client_pose_buffers[client_id]) < SEQUENCE_LENGTH:
            # Add placeholder prediction until we have enough frames
            client_workout_predictions[client_id].append(workout_type)
            return workout_type, f"plank (collecting sequence: {len(client_pose_buffers[client_id])}/{SEQUENCE_LENGTH})"
        
        # Create sequence tensor from buffer
        sequence = np.array(list(client_pose_buffers[client_id]))
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)  # [1, sequence_length, features]
        
        # Make prediction
        with torch.no_grad():
            outputs = workout_classifier(sequence_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]  # Get probabilities
            confidence, predicted_idx = torch.max(probabilities, 0)
            predicted_idx = predicted_idx.item()
            confidence = confidence.item()
        
        # Convert index to original label
        predicted_label = label_encoder.inverse_transform([predicted_idx])[0]
        
        # Validate prediction
        if predicted_label in workout_map:
            # Add to prediction history
            client_workout_predictions[client_id].append(int(predicted_label))
            
            # Get most common prediction from recent history (smoothing)
            workout_counts = {}
            for pred in client_workout_predictions[client_id]:
                workout_counts[pred] = workout_counts.get(pred, 0) + 1
            
            # Find most common prediction
            workout_type = max(workout_counts.items(), key=lambda x: x[1])[0]
            predicted_workout_name = f"{workout_map[workout_type]} (conf: {confidence:.2f})"
        else:
            # Invalid prediction, use default
            workout_type = 12
            predicted_workout_name = "plank (invalid prediction)"
            # Add default to prediction history for continuity
            client_workout_predictions[client_id].append(workout_type)
            
        return workout_type, predicted_workout_name
        
    except Exception as e:
        print(f"Error in sequence-based workout prediction for client {client_id}: {e}")
        # Add default to prediction history for continuity
        if client_id in client_workout_predictions:
            client_workout_predictions[client_id].append(workout_type)
        return workout_type, "plank (prediction error)"

# --- NEW: Sequence-based muscle group prediction ---
def predict_muscle_group_from_sequence(client_id, current_features, workout_type):
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
    # Define muscle group mapping
    muscle_group_map = {
        1: "shoulders",
        2: "chest",
        3: "biceps",
        4: "core",
        5: "triceps",
        6: "legs",
        7: "back"
    }
    
    # Default values
    muscle_group = 0
    predicted_muscle_group = "none (default)"
    
    # Fast path for common workout types - directly map to muscle groups
    # This provides consistency for well-known workout types
    workout_to_muscle = {
        0: 3,  # barbell bicep curl -> biceps
        1: 2,  # bench press -> chest
        2: 2,  # chest fly machine -> chest
        3: 7,  # deadlift -> back
        4: 2,  # decline bench press -> chest
        5: 3,  # hammer curl -> biceps
        6: 6,  # hip thrust -> legs
        7: 2,  # incline bench press -> chest
        8: 7,  # lat pulldown -> back
        9: 1,  # lateral raises -> shoulders
        10: 6, # leg extensions -> legs
        11: 4, # leg raises -> core
        12: 4, # plank -> core
        13: 7, # pull up -> back
        14: 2, # push ups -> chest
        15: 7, # romanian deadlift -> back
        16: 4, # russian twist -> core
        17: 1, # shoulder press -> shoulders
        18: 6, # squat -> legs
        19: 7, # t bar row -> back
        20: 5, # tricep dips -> triceps
        21: 5  # tricep pushdown -> triceps
    }
    
    # If we have a direct mapping, use it
    if workout_type in workout_to_muscle:
        muscle_group = workout_to_muscle[workout_type]
        
        # Initialize buffer for new clients
        if client_id not in client_muscle_predictions:
            client_muscle_predictions[client_id] = deque(maxlen=prediction_smoothing_window)
        
        # Add to prediction history
        client_muscle_predictions[client_id].append(muscle_group)
        
        # Apply smoothing
        muscle_counts = {}
        for pred in client_muscle_predictions[client_id]:
            muscle_counts[pred] = muscle_counts.get(pred, 0) + 1
        
        # Find most common prediction
        muscle_group = max(muscle_counts.items(), key=lambda x: x[1])[0]
        predicted_muscle_group = muscle_group_map.get(muscle_group, "unknown")
        
        return muscle_group, predicted_muscle_group
    
    # Fall back to muscle group classifier if available
    if muscle_group_classifier:
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
            predicted_muscle_label = muscle_group_classifier.predict(features_for_muscle)[0]
            
            # Initialize buffer for new clients
            if client_id not in client_muscle_predictions:
                client_muscle_predictions[client_id] = deque(maxlen=prediction_smoothing_window)
            
            # Validate prediction
            if predicted_muscle_label in muscle_group_map:
                muscle_group = int(predicted_muscle_label)
                
                # Add to prediction history
                client_muscle_predictions[client_id].append(muscle_group)
                
                # Apply smoothing
                muscle_counts = {}
                for pred in client_muscle_predictions[client_id]:
                    muscle_counts[pred] = muscle_counts.get(pred, 0) + 1
                
                # Find most common prediction
                muscle_group = max(muscle_counts.items(), key=lambda x: x[1])[0]
                predicted_muscle_group = muscle_group_map.get(muscle_group, "unknown")
            else:
                muscle_group = 0
                predicted_muscle_group = "none (invalid prediction)"
                # Add default to prediction history
                client_muscle_predictions[client_id].append(muscle_group)
                
        except Exception as e:
            print(f"Error in muscle group prediction for client {client_id}: {e}")
            # Add default to prediction history
            if client_id in client_muscle_predictions:
                client_muscle_predictions[client_id].append(muscle_group)
    
    return muscle_group, predicted_muscle_group

# --- Connection Tracking ---
connected_clients = 0
last_processed_time = {} # Store last *processed* request time per client

# --- WebSocket Event Handlers ---
@socketio.on('connect')
def handle_connect():
    """Handle new client connection"""
    client_id = request.sid
    print(f"Client connected: {client_id}")
    
    # Check if token was provided
    token = request.args.get('token')
    user_id = None
    
    if token:
        # Verify token with backend
        user_data = verify_token(token)
        if user_data:
            user_id = user_data.get('user_id')
            print(f"Authenticated connection from user {user_id}")
    
    # Initialize session tracking
    initialize_session_tracking(client_id, user_id, token)
    
    # Send confirmation
    emit('connected', {'client_id': client_id, 'authenticated': user_id is not None})
    
    return client_id

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    client_id = request.sid
    print(f"Client disconnected: {client_id}")
    
    # End tracking session
    end_client_session(client_id)

@socketio.on('pose_data')
def handle_pose_data(data):
    """Handle incoming pose data and calculate corrections"""
    start_time = time.time()
    client_id = request.sid
    
    try:
        # Extract landmarks from request
        landmarks = data.get('landmarks', [])
        timestamp = data.get('timestamp', int(time.time() * 1000))
        
        # Get selected workout type from request
        selected_workout = data.get('selected_workout', 0)
        
        # Update workout type in session tracking
        if client_id in client_sessions:
            client_sessions[client_id]['workout_type'] = selected_workout
        
        # Convert landmarks to numpy array for processing
        landmarks_flat = []
        if landmarks and isinstance(landmarks, list):
            # Extract only the key landmarks we need (filter out irrelevant ones)
            try:
                # Convert landmarks to flat array format that the model expects
                for idx in BODY_KEYPOINTS_INDICES:
                    if idx < len(landmarks):
                        landmark = landmarks[idx]
                        if isinstance(landmark, dict):
                            landmarks_flat.extend([
                                landmark.get('x', 0), 
                                landmark.get('y', 0), 
                                landmark.get('z', 0)
                            ])
                    else:
                        # Add zeros if landmark is missing
                        landmarks_flat.extend([0, 0, 0])
                
                # Ensure we have exactly 36 values
                if len(landmarks_flat) != 36:
                    # Pad or truncate to 36 values
                    if len(landmarks_flat) < 36:
                        landmarks_flat.extend([0] * (36 - len(landmarks_flat)))
                    else:
                        landmarks_flat = landmarks_flat[:36]
                
                # Convert to numpy array for processing
                landmarks_array = np.array(landmarks_flat, dtype=np.float32)
            except Exception as e:
                print(f"Error processing landmarks: {e}")
                landmarks_array = np.zeros(36, dtype=np.float32)
        else:
            # Default to zeros if landmarks are invalid
            landmarks_array = np.zeros(36, dtype=np.float32)
        
        # Calculate average confidence for this frame
        avg_confidence = 0
        if landmarks:
            confidences = [lm.get('visibility', 0) for lm in landmarks if isinstance(lm, dict) and 'visibility' in lm]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
        
        # Get predicted workout type and muscle group from model
        try:
            predicted_workout, predicted_workout_name = predict_workout_from_sequence(client_id, landmarks_array.reshape(1, -1))
            predicted_muscle_group, predicted_muscle_group_name = predict_muscle_group_from_sequence(client_id, landmarks_array.reshape(1, -1), selected_workout)
        except Exception as e:
            print(f"Error during workout/muscle prediction: {e}")
            predicted_workout, predicted_workout_name = 12, "plank (error)"
            predicted_muscle_group, predicted_muscle_group_name = 0, "none (error)"
        
        # Get corrections
        try:
            corrections = get_pose_corrections(landmarks_array, selected_workout)
            
            # Process corrections into the expected format
            correction_dict = {}
            if corrections is not None:
                # Check if corrections is a NumPy array
                if isinstance(corrections, np.ndarray) and corrections.size > 0:
                    # Convert to a dictionary format for each joint
                    for i, idx in enumerate(BODY_KEYPOINTS_INDICES):
                        # Each joint has x, y, z corrections
                        base_idx = i * 3
                        correction_dict[str(idx)] = {
                            'x': float(corrections[base_idx]),
                            'y': float(corrections[base_idx + 1]),
                            'z': float(corrections[base_idx + 2]) if len(corrections) > base_idx + 2 else 0.0
                        }
            
            # Calculate average correction magnitude
            correction_magnitude = 0
            if correction_dict:
                magnitudes = []
                for joint_idx, correction in correction_dict.items():
                    if isinstance(correction, dict) and 'x' in correction and 'y' in correction:
                        magnitude = (correction['x']**2 + correction['y']**2)**0.5
                        magnitudes.append(magnitude)
                
                if magnitudes:
                    correction_magnitude = sum(magnitudes) / len(magnitudes)
        except Exception as e:
            print(f"Error processing corrections: {e}")
            correction_dict = {}
            correction_magnitude = 0
        
        # Add predictions to the response
        response_data = {
            'corrections': correction_dict,
            'predicted_workout_type': predicted_workout,
            'predicted_muscle_group': predicted_muscle_group
        }
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Send the response
        emit('pose_corrections', response_data)
        
        # Calculate round-trip time
        response_time = int(time.time() * 1000) - timestamp
        
        # Update tracking
        update_session_data(client_id, frames=1, corrections=1, workout_type=selected_workout)
        update_performance_metrics(
            client_id,
            confidence=avg_confidence,
            correction_magnitude=correction_magnitude,
            response_time=response_time,
            processing_time=processing_time,
            predicted_type=predicted_workout
        )
    except Exception as e:
        print(f"Error handling pose data: {e}")
        # Send an error response to the client
        emit('error', {'message': f'Server error processing pose data: {str(e)}'})

@app.route('/')
def index():
    """Basic route to confirm the server is running."""
    return "Pose Correction WebSocket Server with Sequential LSTM Workout Classification is running."

# --- Main Execution ---
if __name__ == '__main__':
    print("Current working directory:", os.getcwd())

    # Load models at startup
    print("Loading DNN pose model...")
    pose_model_loaded = load_pose_model()

    print("Loading LSTM workout classifier model...")
    classifier_loaded = load_workout_classifier()
    
    print("Loading muscle group classifier model...")
    muscle_group_loaded = load_muscle_group_classifier()

    if not pose_model_loaded:
        print("CRITICAL: DNN Pose Model failed to load. Corrections will be zeros.")
        # Allow server to start but corrections won't work properly
    if not classifier_loaded:
        print("WARNING: LSTM Workout Classifier failed to load. Workout type will default to plank (12).")
        # Allow server to start but classification won't work
    if not muscle_group_loaded:
        print("WARNING: Muscle Group Classifier failed to load. Muscle group predictions will not be available.")
        # Allow server to start but muscle group detection won't work

    print(f"Starting WebSocket server on http://0.0.0.0:8001 (using {device})")
    print(f"Using sequence length of {SEQUENCE_LENGTH} frames for workout classification")

    # Run the server using the determined async_mode
    # The host/port arguments are passed directly to socketio.run
    print(f"Attempting to run with async_mode='{socketio.async_mode}'...")
    try:
         socketio.run(app, host='0.0.0.0', port=8001, debug=False, use_reloader=False)
    except Exception as run_error:
         print(f"Error running socketio.run: {run_error}")
         if not USING_EVENTLET:
             print("Falling back to standard Flask development server without explicit async_mode.")
             # Fallback if async_mode=None causes issues or if eventlet wasn't used initially
             app.run(host='0.0.0.0', port=8001, debug=False) # Use app.run as a last resort