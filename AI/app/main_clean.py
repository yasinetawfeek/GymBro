"""
Clean, organized main application for the AI service.

This file provides a well-structured Flask application with proper separation of concerns.
"""
import os
import time
import uuid
import requests
import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit

# Eventlet monkey patching (MUST be first)
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
    USING_EVENTLET = False

# Import our organized modules
from config import (
    BODY_KEYPOINTS_INDICES, WORKOUT_MAP, MUSCLE_GROUP_MAP, 
    WORKOUT_TO_MUSCLE, BACKEND_BASE_URL, USAGE_ENDPOINT, 
    PERFORMANCE_ENDPOINT, MODEL_VERSION, SEQUENCE_LENGTH
)
from services import (
    load_pose_model, load_workout_classifier, load_muscle_group_classifier,
    PoseCorrectionService, WorkoutClassificationService, MuscleGroupClassificationService
)


class AIService:
    """Main AI service class that orchestrates all components."""
    
    def __init__(self):
        self.app = Flask(__name__)
        self.device = self._setup_device()
        self.models = {}
        self.services = {}
        self.client_sessions = {}
        self.performance_metrics = {}
        self._setup_socketio()
        self._load_models()
        self._setup_routes()
        self._setup_socket_handlers()
    
    def _setup_device(self):
        """Setup PyTorch device."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        return device
    
    def _setup_socketio(self):
        """Setup SocketIO with proper configuration."""
        async_mode = 'eventlet' if USING_EVENTLET else None
        self.socketio = SocketIO(
            self.app, 
            cors_allowed_origins="*", 
            ping_timeout=10, 
            ping_interval=5, 
            async_mode=async_mode
        )
        print(f"SocketIO initialized with async_mode: {self.socketio.async_mode}")
    
    def _load_models(self):
        """Load all required models."""
        print("Loading models...")
        
        # Load pose correction model
        pose_model = load_pose_model(self.device)
        if pose_model:
            self.models['pose'] = pose_model
            self.services['pose'] = PoseCorrectionService(pose_model, self.device)
            print("✓ Pose correction model loaded")
        else:
            print("✗ Failed to load pose correction model")
        
        # Load workout classifier
        workout_classifier, feature_scaler, label_encoder = load_workout_classifier(self.device)
        if all([workout_classifier, feature_scaler, label_encoder]):
            self.models['workout'] = workout_classifier
            self.models['feature_scaler'] = feature_scaler
            self.models['label_encoder'] = label_encoder
            self.services['workout'] = WorkoutClassificationService(
                workout_classifier, feature_scaler, label_encoder, self.device
            )
            print("✓ Workout classifier loaded")
        else:
            print("✗ Failed to load workout classifier")
        
        # Load muscle group classifier
        muscle_classifier = load_muscle_group_classifier()
        if muscle_classifier:
            self.models['muscle'] = muscle_classifier
            self.services['muscle'] = MuscleGroupClassificationService(muscle_classifier)
            print("✓ Muscle group classifier loaded")
        else:
            print("✗ Failed to load muscle group classifier")
    
    def _setup_routes(self):
        """Setup Flask routes."""
        @self.app.route('/')
        def index():
            """Basic route to confirm the server is running."""
            return "Pose Correction WebSocket Server with Sequential LSTM Workout Classification is running."
        
        @self.app.route('/health')
        def health():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'models_loaded': {
                    'pose': 'pose' in self.models,
                    'workout': 'workout' in self.models,
                    'muscle': 'muscle' in self.models
                },
                'version': MODEL_VERSION
            })
    
    def _setup_socket_handlers(self):
        """Setup SocketIO event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle new client connection."""
            client_id = request.sid
            print(f"Client connected: {client_id}")
            
            # Check if token was provided
            token = request.args.get('token')
            user_id = None
            
            if token:
                # Verify token with backend
                user_data = self._verify_token(token)
                if user_data:
                    user_id = user_data.get('user_id')
                    print(f"Authenticated connection from user {user_id}")
            
            # Initialize session tracking
            self._initialize_session_tracking(client_id, user_id, token)
            
            # Send confirmation
            emit('connected', {'client_id': client_id, 'authenticated': user_id is not None})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            client_id = request.sid
            print(f"Client disconnected: {client_id}")
            
            # End tracking session
            self._end_client_session(client_id)
        
        @self.socketio.on('pose_data')
        def handle_pose_data(data):
            """Handle incoming pose data and calculate corrections."""
            start_time = time.time()
            client_id = request.sid
            
            try:
                # Extract landmarks from request
                landmarks = data.get('landmarks', [])
                timestamp = data.get('timestamp', int(time.time() * 1000))
                selected_workout = data.get('selected_workout', 0)
                
                # Update workout type in session tracking
                if client_id in self.client_sessions:
                    self.client_sessions[client_id]['workout_type'] = selected_workout
                
                # Process landmarks
                landmarks_array = self._process_landmarks(landmarks)
                
                # Calculate average confidence for this frame
                avg_confidence = self._calculate_confidence(landmarks)
                
                # Get predictions
                predicted_workout, predicted_workout_name = self._predict_workout(client_id, landmarks_array)
                predicted_muscle_group, predicted_muscle_group_name = self._predict_muscle_group(
                    client_id, landmarks_array, selected_workout
                )
                
                # Get pose corrections
                corrections = self._get_pose_corrections(landmarks_array, selected_workout)
                correction_dict, correction_magnitude = self._process_corrections(corrections)
                
                # Prepare response
                response_data = {
                    'corrections': correction_dict,
                    'predicted_workout_type': predicted_workout,
                    'predicted_muscle_group': predicted_muscle_group
                }
                
                # Send response
                emit('pose_corrections', response_data)
                
                # Update tracking
                self._update_session_data(client_id, frames=1, corrections=1, workout_type=selected_workout)
                self._update_performance_metrics(
                    client_id, avg_confidence, correction_magnitude, 
                    int(time.time() * 1000) - timestamp, (time.time() - start_time) * 1000, predicted_workout
                )
                
            except Exception as e:
                print(f"Error handling pose data: {e}")
                emit('error', {'message': f'Server error processing pose data: {str(e)}'})
    
    def _verify_token(self, token):
        """Verify user token with Django backend."""
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
    
    def _process_landmarks(self, landmarks):
        """Process landmarks into the format expected by models."""
        landmarks_flat = []
        if landmarks and isinstance(landmarks, list):
            try:
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
                        landmarks_flat.extend([0, 0, 0])
                
                # Ensure we have exactly 36 values
                if len(landmarks_flat) != 36:
                    if len(landmarks_flat) < 36:
                        landmarks_flat.extend([0] * (36 - len(landmarks_flat)))
                    else:
                        landmarks_flat = landmarks_flat[:36]
                
                return np.array(landmarks_flat, dtype=np.float32)
            except Exception as e:
                print(f"Error processing landmarks: {e}")
                return np.zeros(36, dtype=np.float32)
        else:
            return np.zeros(36, dtype=np.float32)
    
    def _calculate_confidence(self, landmarks):
        """Calculate average confidence from landmarks."""
        if landmarks:
            confidences = [lm.get('visibility', 0) for lm in landmarks if isinstance(lm, dict) and 'visibility' in lm]
            if confidences:
                return sum(confidences) / len(confidences)
        return 0
    
    def _predict_workout(self, client_id, landmarks_array):
        """Predict workout type from landmarks."""
        if 'workout' in self.services:
            return self.services['workout'].predict_workout_from_sequence(
                client_id, landmarks_array.reshape(1, -1), SEQUENCE_LENGTH
            )
        return 12, "plank (model not loaded)"
    
    def _predict_muscle_group(self, client_id, landmarks_array, workout_type):
        """Predict muscle group from landmarks and workout type."""
        if 'muscle' in self.services:
            return self.services['muscle'].predict_muscle_group_from_sequence(
                client_id, landmarks_array.reshape(1, -1), workout_type
            )
        return 0, "none (model not loaded)"
    
    def _get_pose_corrections(self, landmarks_array, workout_type):
        """Get pose corrections from the pose model."""
        if 'pose' in self.services:
            return self.services['pose'].get_pose_corrections(landmarks_array, workout_type)
        return np.zeros(36)
    
    def _process_corrections(self, corrections):
        """Process corrections into the expected format."""
        correction_dict = {}
        correction_magnitude = 0
        
        if corrections is not None and isinstance(corrections, np.ndarray) and corrections.size > 0:
            # Convert to a dictionary format for each joint
            for i, idx in enumerate(BODY_KEYPOINTS_INDICES):
                base_idx = i * 3
                correction_dict[str(idx)] = {
                    'x': float(corrections[base_idx]),
                    'y': float(corrections[base_idx + 1]),
                    'z': float(corrections[base_idx + 2]) if len(corrections) > base_idx + 2 else 0.0
                }
            
            # Calculate average correction magnitude
            magnitudes = []
            for joint_idx, correction in correction_dict.items():
                if isinstance(correction, dict) and 'x' in correction and 'y' in correction:
                    magnitude = (correction['x']**2 + correction['y']**2)**0.5
                    magnitudes.append(magnitude)
            
            if magnitudes:
                correction_magnitude = sum(magnitudes) / len(magnitudes)
        
        return correction_dict, correction_magnitude
    
    def _initialize_session_tracking(self, client_id, user_id=None, token=None, workout_type=0):
        """Initialize session tracking for a new client."""
        self.client_sessions[client_id] = {
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
        self.performance_metrics[client_id] = {
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
                    self.client_sessions[client_id]['session_id'] = data['session_id']
                    print(f"Session started in backend: {data['session_id']}")
                else:
                    print(f"Failed to start session: {response.text}")
            except Exception as e:
                print(f"Error starting session: {str(e)}")
    
    def _update_session_data(self, client_id, frames=1, corrections=1, workout_type=None):
        """Update session tracking data."""
        if client_id in self.client_sessions:
            self.client_sessions[client_id]['frames_processed'] += frames
            self.client_sessions[client_id]['corrections_sent'] += corrections
            self.client_sessions[client_id]['last_activity'] = time.time()
            
            if workout_type is not None:
                self.client_sessions[client_id]['workout_type'] = workout_type
    
    def _update_performance_metrics(self, client_id, confidence=None, correction_magnitude=None, 
                                  response_time=None, processing_time=None, predicted_type=None):
        """Update performance metrics for the client session."""
        if client_id not in self.performance_metrics:
            return
        
        metrics = self.performance_metrics[client_id]
        
        # Update frame count
        metrics['frame_count'] += 1
        
        # Calculate frames per second
        elapsed = time.time() - metrics['start_time']
        if elapsed > 0:
            fps = metrics['frame_count'] / elapsed
            metrics['frames_per_second'].append(fps)
        
        # Track various metrics
        if confidence is not None:
            metrics['confidence_values'].append(confidence)
        if correction_magnitude is not None:
            metrics['correction_magnitudes'].append(correction_magnitude)
        if response_time is not None:
            metrics['response_times'].append(response_time)
        if processing_time is not None:
            metrics['processing_times'].append(processing_time)
        
        # Track time to first correction
        if metrics['first_correction_time'] is None and correction_magnitude is not None:
            metrics['first_correction_time'] = time.time() - metrics['start_time']
        
        # Track prediction stability
        if predicted_type is not None:
            metrics['prediction_counts'][predicted_type] = metrics['prediction_counts'].get(predicted_type, 0) + 1
            
            if metrics['last_prediction'] is not None and metrics['last_prediction'] != predicted_type:
                metrics['prediction_changes'] += 1
            
            metrics['last_prediction'] = predicted_type
    
    def _end_client_session(self, client_id):
        """End client session and report final metrics."""
        if client_id not in self.client_sessions:
            return
        
        session = self.client_sessions[client_id]
        
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
        
        # Clean up session data
        if client_id in self.performance_metrics:
            del self.performance_metrics[client_id]
        del self.client_sessions[client_id]
    
    def run(self, host='0.0.0.0', port=8001, debug=False):
        """Run the AI service."""
        print(f"Starting AI service on http://{host}:{port}")
        print(f"Using sequence length of {SEQUENCE_LENGTH} frames for workout classification")
        
        try:
            self.socketio.run(self.app, host=host, port=port, debug=debug, use_reloader=False, allow_unsafe_werkzeug=True)
        except Exception as run_error:
            print(f"Error running socketio.run: {run_error}")
            if not USING_EVENTLET:
                print("Falling back to standard Flask development server without explicit async_mode.")
                self.app.run(host=host, port=port, debug=debug)


def main():
    """Main entry point for the AI service."""
    print("Current working directory:", os.getcwd())
    
    # Create and run the AI service
    ai_service = AIService()
    ai_service.run()


if __name__ == '__main__':
    main()