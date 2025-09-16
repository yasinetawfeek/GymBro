"""
Model loading utilities for the AI service.
"""
import os
import torch
import pickle
from models import EnhancedPoseModel, LSTMWorkoutClassifier
from config import SEQUENCE_LENGTH


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


def load_pose_model(device):
    """Load the DNN pose correction model."""
    try:
        # Define potential base directories relative to the script or common structures
        possible_base_dirs = ['.', '..', '../..', 'AI', '../models']
        # Construct full search paths
        search_paths = [os.path.join(base, 'pose_correctors') for base in possible_base_dirs] + ['data', '/data']

        model_path = find_file('best_model.pth', search_paths)

        if model_path is None:
            print(f"Error: Could not find pose model file 'best_model.pth'")
            print(f"Current working directory: {os.getcwd()}")
            return None

        pose_model = EnhancedPoseModel(input_dim=37, hidden_dim=512, output_dim=36).to(device)
        pose_model.load_state_dict(torch.load(model_path, map_location=device))
        pose_model.eval()
        print(f"DNN pose model loaded successfully from {model_path}")
        
        if torch.cuda.is_available():
            print("Running warmup inference for CUDA initialization...")
            dummy_input = torch.zeros(1, 37, device=device)
            with torch.no_grad():
                _ = pose_model(dummy_input)
        
        return pose_model
    except Exception as e:
        print(f"Error loading DNN pose model: {e}")
        return None


def load_workout_classifier(device):
    """Load the LSTM workout classifier model and supporting files."""
    try:
        # Define potential base directories relative to the script or common structures
        possible_base_dirs = ['.', '..', '../..', 'AI', '../models']
        # Construct full search paths
        search_paths = [os.path.join(base, 'workout_classifiers') for base in possible_base_dirs] +\
                       ['data/models', 'data', '/data/models', '/data', '../data']

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
            return None, None, None

        # 2. Find and load the feature scaler
        scaler_path = find_file('feature_scaler.pkl', search_paths)
        if scaler_path is None:
            print(f"Error: Could not find feature scaler file")
            return None, None, None

        # 3. Find and load the label encoder
        encoder_path = find_file('label_encoder.pkl', search_paths)
        if encoder_path is None:
            print(f"Error: Could not find label encoder file")
            return None, None, None

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

        return workout_classifier, feature_scaler, label_encoder
    except FileNotFoundError as e:
        print(f"Error: File not found when loading LSTM workout classifier: {e}")
        return None, None, None
    except Exception as e:
        print(f"Error loading LSTM workout classifier: {e}")
        return None, None, None


def load_muscle_group_classifier():
    """Load the muscle group classifier model (.pkl) at startup."""
    try:
        # Define potential base directories relative to the script or common structures
        possible_base_dirs = ['.', '..', '../..', 'AI', '../models']
        # Construct full search paths, including a 'models' subdirectory as seen in training script
        search_paths = [os.path.join(base, 'data', 'models') for base in possible_base_dirs] + \
                       [os.path.join(base, 'data') for base in possible_base_dirs] + \
                       [os.path.join(base, 'muscle_group_classifiers') for base in possible_base_dirs] +\
                       ['data/models', 'data', '/data/models', '/data'] # Add /data for container environments

        classifier_path = find_file('rfc_muscle_group_classifier.pkl', search_paths)

        if classifier_path is None:
            print(f"Error: Could not find muscle group classifier file 'rfc_muscle_group_classifier.pkl'")
            print(f"Current working directory: {os.getcwd()}")
            return None

        with open(classifier_path, 'rb') as f:
            muscle_group_classifier = pickle.load(f)
        print(f"Muscle group classifier loaded successfully from {classifier_path}")
        return muscle_group_classifier
    except FileNotFoundError:
        print(f"Error: Muscle group classifier file not found at expected paths.")
        return None
    except pickle.UnpicklingError as e:
        print(f"Error unpickling muscle group classifier model: {e}")
        return None
    except Exception as e:
        print(f"Error loading muscle group classifier model: {e}")
        return None