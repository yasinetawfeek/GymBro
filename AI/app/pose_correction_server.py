from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import os
from flask_socketio import SocketIO, emit
import json
import time
from datetime import datetime # Import datetime for formatted timestamp

# DNN model class definition
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
        return self.net(x) * 0.1

# Workout type mapping (Unchanged)
workout_map = { 0: "barbell bicep curl", 1: "bench press", 2: "chest fly machine", 3: "deadlift", 4: "decline bench press", 5: "hammer curl", 6: "hip thrust", 7: "incline bench press", 8: "lat pulldown", 9: "lateral raises", 10: "leg extensions", 11: "leg raises", 12: "plank", 13: "pull up", 14: "push ups", 15: "romanian deadlift", 16: "russian twist", 17: "shoulder press", 18: "squat", 19: "t bar row", 20: "tricep dips", 21: "tricep pushdown" }

# Initialize Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=10, ping_interval=5)

# Global model variable and settings
pose_model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
last_inference_time = 0
INFERENCE_THROTTLE = 0.05 # 50ms throttle

def load_model():
    """Load the DNN model at startup"""
    global pose_model
    try:
        model_paths = [ 'AI/data/best_model.pth', 'data/best_model.pth', '../data/best_model.pth', '../../data/best_model.pth' ]
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        if model_path is None:
            print("Error: Could not find model file in any of the expected locations")
            return False
        pose_model = EnhancedPoseModel(input_dim=37, hidden_dim=512, output_dim=36).to(device)
        pose_model.load_state_dict(torch.load(model_path, map_location=device))
        pose_model.eval()
        print(f"DNN model loaded successfully from {model_path}")
        if torch.cuda.is_available():
            print("Running warmup inference for CUDA initialization...")
            dummy_input = torch.zeros(1, 37, device=device)
            with torch.no_grad():
                _ = pose_model(dummy_input) # Assign to dummy variable
        return True
    except Exception as e:
        print(f"Error loading DNN model: {e}")
        return False

def get_pose_corrections(landmarks, workout_type=0):
    """Get pose corrections from the model, applying throttling."""
    global pose_model, last_inference_time
    if pose_model is None:
        print("Model not loaded, attempting to load")
        if not load_model():
            return np.zeros(36)

    current_time = time.time()
    time_since_last = current_time - last_inference_time
    if time_since_last < INFERENCE_THROTTLE:
        # print(f"Backend throttled: {time_since_last*1000:.1f}ms < {INFERENCE_THROTTLE*1000}ms") # Optional detailed throttle log
        return None # Signal to skip this frame

    try:
        inference_start = time.time()
        input_data = np.append(landmarks, workout_type)
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = pose_model(input_tensor)
            corrections = prediction[0].cpu().numpy()
        inference_time = time.time() - inference_start
        last_inference_time = current_time # Update time only when inference runs
        if inference_time > 0.1:
            print(f"Slow inference: {inference_time:.3f}s")
        return corrections
    except Exception as e:
        print(f"Error during prediction: {e}")
        return np.zeros(36)

# Connection tracking
connected_clients = 0
last_processed_time = {} # Renamed for clarity: Store last *processed* request time per client

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    global connected_clients
    connected_clients += 1
    client_id = request.sid
    last_processed_time[client_id] = 0 # Initialize last processed time for new client
    print(f'Client connected ({client_id}). Total clients: {connected_clients}')
    emit('connected', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    global connected_clients
    client_id = request.sid
    connected_clients -= 1
    if client_id in last_processed_time:
        del last_processed_time[client_id] # Clean up tracking
    print(f'Client disconnected ({client_id}). Total clients: {connected_clients}')

@socketio.on('pose_data')
def handle_pose_data(data):
    """Handles incoming pose data, performs inference, and emits corrections."""
    client_id = request.sid
    now = time.time()
    receive_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] # Milliseconds

    # --- ADDED LOG LINE ---
    print(f"Received pose_data from client ({client_id}) at {receive_timestamp}")
    # --- END ADDED LOG LINE ---


    # Check time since last *processed* request for this client
    if client_id in last_processed_time and last_processed_time[client_id] != 0:
        time_since_last_processed = now - last_processed_time[client_id]
        # Log only if the gap since the last *processed* request is long
        if time_since_last_processed > 1.0:
            # This log now clearly indicates the gap between successfully processed requests
            print(f"Long gap between *processed* requests for client {client_id}: {time_since_last_processed:.2f}s")
    # Note: We update last_processed_time *only* when we successfully process below

    try:
        process_start = time.time()
        landmarks = data.get('landmarks', [])
        if not landmarks:
            print(f"Warning: Received empty landmarks list from {client_id}.")
            return # Don't process if no landmarks

        # Convert landmarks (ensure correct format)
        flat_landmarks = []
        selected_indices = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        for idx in selected_indices:
            # Check index bounds and if the landmark exists and is not None
            if idx < len(landmarks) and landmarks[idx] is not None:
                landmark = landmarks[idx]
                # Ensure landmark is a dict-like object before accessing keys
                if isinstance(landmark, dict):
                    flat_landmarks.extend([landmark.get('x', 0), landmark.get('y', 0), landmark.get('z', 0)])
                else:
                     # Log if the format is unexpected
                     print(f"Warning: Landmark at index {idx} from client {client_id} is not a dictionary: {landmark}")
                     flat_landmarks.extend([0, 0, 0]) # Add zeros if format is wrong
            else:
                # Add zeros if index is out of bounds or landmark is None
                flat_landmarks.extend([0, 0, 0])

        # Check if the final list has the correct length
        if len(flat_landmarks) != 36:
            print(f"Warning: Landmark processing for client {client_id} resulted in {len(flat_landmarks)} values, expected 36. Padding/truncating.")
            if len(flat_landmarks) < 36:
                flat_landmarks.extend([0] * (36 - len(flat_landmarks)))
            else:
                flat_landmarks = flat_landmarks[:36]

        workout_type = 0 # Hardcoded workout type

        # Get corrections from model (handles internal throttling)
        corrections = get_pose_corrections(flat_landmarks, workout_type)

        # If throttled by get_pose_corrections, skip sending emit
        if corrections is None:
            # print(f"Skipping emit for client {client_id} due to backend throttle.") # Optional detailed log
            return

        # Update last processed time *only* when inference was successful and emit is happening
        last_processed_time[client_id] = now

        # Convert to dict with joint indices
        correction_data = {}
        for i, idx in enumerate(selected_indices):
            # Basic check for NaN or infinity, replace with 0 if found
            x_corr = float(corrections[i*3]) if np.isfinite(corrections[i*3]) else 0.0
            y_corr = float(corrections[i*3+1]) if np.isfinite(corrections[i*3+1]) else 0.0
            z_corr = float(corrections[i*3+2]) if np.isfinite(corrections[i*3+2]) else 0.0
            correction_data[str(idx)] = {'x': x_corr, 'y': y_corr, 'z': z_corr}


        # Log before emitting
        emit_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] # Milliseconds
        print(f"Streaming displacement value to frontend ({client_id}) at {emit_timestamp}")

        # Send corrections back to client
        emit('pose_corrections', correction_data)

        process_time = time.time() - process_start
        if process_time > 0.2: # Log if the processing part takes longer than 200ms
            print(f"Slow processing cycle for client {client_id}: {process_time:.3f}s")

    except Exception as e:
        # Log any exceptions during processing
        print(f"Error processing pose data for client {client_id}: {e}")
        # Optionally send an error back to the specific client
        try:
            emit('error', {'message': f'Backend error processing pose data: {str(e)}'})
        except Exception as emit_error:
            print(f"Error emitting error message to client {client_id}: {emit_error}")


@app.route('/')
def index():
    """Basic route to confirm the server is running."""
    return "Pose Correction WebSocket Server is running."

# Load model when app starts
print("Loading DNN model...")
model_loaded = load_model()

if __name__ == '__main__':
    if not model_loaded:
        print("Exiting: Model failed to load.")
    else:
        print(f"Starting WebSocket server on http://localhost:8001 (using {device})")
        print("Current working directory:", os.getcwd())
        # Consider removing debug=True for production
        # Use eventlet or gevent for better performance if installed
        try:
            import eventlet
            print("Attempting to use eventlet...")
            # eventlet.monkey_patch() # Might be needed depending on other libraries
            # Use default Flask-SocketIO runner with eventlet if available
            socketio.run(app, host='0.0.0.0', port=8001)
        except ImportError:
            print("Eventlet not found, using default Flask development server (threading).")
            print("For production, consider installing eventlet or gevent and using a production WSGI server.")
            # Fallback to Werkzeug's development server (less performant for websockets)
            socketio.run(app, host='0.0.0.0', port=8001, debug=True, allow_unsafe_werkzeug=True)

