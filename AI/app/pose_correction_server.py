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
        # Prepare input data (workout type + landmarks)
        input_data = np.append(workout_type, landmarks) # workout_type is the FIRST element
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
        current_features: Current frame features (36 values)
        
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
        
        # Scale the current frame features
        scaled_features = feature_scaler.transform(current_features.reshape(1, -1))[0]
        
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
            # Use current features for prediction
            features_for_muscle = current_features.reshape(1, -1)
            
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
    global connected_clients
    connected_clients += 1
    client_id = request.sid
    last_processed_time[client_id] = 0 # Initialize last processed time
    print(f'Client connected ({client_id}). Total clients: {connected_clients}')
    emit('connected', {'status': 'connected'}) # Confirm connection

@socketio.on('disconnect')
def handle_disconnect():
    global connected_clients
    client_id = request.sid
    connected_clients -= 1
    
    # Clean up client-specific data
    if client_id in last_processed_time:
        del last_processed_time[client_id]
    if client_id in client_pose_buffers:
        del client_pose_buffers[client_id]
    if client_id in client_workout_predictions:
        del client_workout_predictions[client_id]
    if client_id in client_muscle_predictions:
        del client_muscle_predictions[client_id]
        
    print(f'Client disconnected ({client_id}). Total clients: {connected_clients}')

@socketio.on('pose_data')
def handle_pose_data(data):
    """Handles incoming pose data, predicts workout, performs inference, and emits corrections."""
    client_id = request.sid
    now = time.time()
    # receive_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] # Optional: for detailed logging

    # Check time since last *processed* request for logging long gaps
    # if client_id in last_processed_time and last_processed_time[client_id] != 0:
    #     time_since_last_processed = now - last_processed_time[client_id]
    #     if time_since_last_processed > 1.0:
    #         print(f"Long gap between *processed* requests for client {client_id}: {time_since_last_processed:.2f}s")

    try:
        process_start = time.time()
        landmarks = data.get('landmarks', [])
        # Get selected workout from frontend if provided, otherwise default to None
        selected_workout = data.get('selected_workout')
        
        if not landmarks:
            print(f"Warning: Received empty landmarks list from {client_id}.")
            return

        # --- Landmark Processing (Extract relevant coordinates) ---
        flat_landmarks = []
        for idx in BODY_KEYPOINTS_INDICES:
            if idx < len(landmarks) and landmarks[idx] is not None and isinstance(landmarks[idx], dict):
                landmark = landmarks[idx]
                flat_landmarks.extend([
                    landmark.get('x', 0.0), landmark.get('y', 0.0), landmark.get('z', 0.0)
                ])
            else:
                # Handle missing, None, or incorrect type landmarks
                flat_landmarks.extend([0.0, 0.0, 0.0])

        # Ensure exactly 36 values
        if len(flat_landmarks) != FLAT_LANDMARK_SIZE:
            print(f"Warning: Landmark processing for client {client_id} resulted in {len(flat_landmarks)} values, expected {FLAT_LANDMARK_SIZE}. Padding/truncating.")
            flat_landmarks = (flat_landmarks + [0.0]*FLAT_LANDMARK_SIZE)[:FLAT_LANDMARK_SIZE]

        flat_landmarks_np = np.array(flat_landmarks, dtype=float)

        # --- Predict Workout Type (using workout buffer/scaler) ---
        workout_type_index, predicted_workout_name = predict_workout_from_sequence(client_id, flat_landmarks_np)

        # --- Predict Muscle Group ---
        muscle_group_index, predicted_muscle_group_name = predict_muscle_group_from_sequence(
            client_id, flat_landmarks_np, workout_type_index)

        # --- Get Pose Corrections (using correction buffer/scaler) ---
        # Use the selected workout if provided, otherwise use the predicted one
        correction_workout_type = selected_workout if selected_workout is not None else workout_type_index
        corrections_array = get_pose_corrections(flat_landmarks_np, correction_workout_type)

        # If throttled or buffer not ready (returns None or Zeros), stop processing this message
        if corrections_array is None:
            # print(f"Correction throttled for {client_id}") # Optional debug
            return
        if np.all(corrections_array == 0):
             # Buffer might not be full, or model isn't loaded - don't emit zero corrections unless intended
             # print(f"Zero corrections returned for {client_id}, likely buffer not full or model issue.") # Optional debug
             # We might still want to emit workout/muscle group info even if corrections are zero
             pass # Continue to emit, but corrections will be zero

        # Update last processed time *only* when inference was attempted (even if buffer wasn't full)
        last_processed_time[client_id] = now

        # --- Prepare Correction Data for Frontend ---
        correction_data = {}
        for i, original_landmark_idx in enumerate(BODY_KEYPOINTS_INDICES):
            base_idx = i * LANDMARK_DIM
            if (base_idx + LANDMARK_DIM -1) < len(corrections_array):
                x_corr = float(corrections_array[base_idx]) if np.isfinite(corrections_array[base_idx]) else 0.0
                y_corr = float(corrections_array[base_idx+1]) if np.isfinite(corrections_array[base_idx+1]) else 0.0
                z_corr = float(corrections_array[base_idx+2]) if np.isfinite(corrections_array[base_idx+2]) else 0.0
                correction_data[str(original_landmark_idx)] = {'x': x_corr, 'y': y_corr, 'z': z_corr}
            else:
                print(f"Warning: Index out of bounds when formatting corrections for joint {original_landmark_idx}")
                correction_data[str(original_landmark_idx)] = {'x': 0.0, 'y': 0.0, 'z': 0.0}

        # Add predicted workout type and muscle group to correction data
        correction_data['predicted_workout_type'] = workout_type_index
        correction_data['predicted_muscle_group'] = muscle_group_index
        # Optionally add names for debugging/display
        correction_data['predicted_workout_name'] = predicted_workout_name
        correction_data['predicted_muscle_group_name'] = predicted_muscle_group_name

        # Add sequence info for debugging
        correction_data['correction_sequence_len'] = len(client_pose_buffers.get(client_id, []))
        correction_data['target_correction_sequence_len'] = SEQUENCE_LENGTH
        correction_data['workout_sequence_len'] = len(client_pose_buffers.get(client_id, []))
        correction_data['target_workout_sequence_len'] = workout_classifier_config.get('sequence_length', SEQUENCE_LENGTH)

        # --- Emitting ---
        # emit_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] # Optional
        emit('pose_corrections', correction_data)

        # Log processing time if slow
        process_time = time.time() - process_start
        if process_time > 0.2:
            print(f"Slow processing cycle for client {client_id}: {process_time:.3f}s (Workout: {predicted_workout_name}, Muscle: {predicted_muscle_group_name})")

    except Exception as e:
        print(f"Error processing pose data for client {client_id}: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        try:
            emit('error', {'message': f'Backend error processing pose data: {str(e)}'})
        except Exception as emit_error:
            print(f"Error emitting error message to client {client_id}: {emit_error}")


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