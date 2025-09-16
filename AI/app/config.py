"""
Configuration settings for the AI service.
"""
import os

# Model configuration
BODY_KEYPOINTS_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
LANDMARK_DIM = 3  # x, y, z
FLAT_LANDMARK_SIZE = len(BODY_KEYPOINTS_INDICES) * LANDMARK_DIM  # Should be 36

# Sequence configuration
SEQUENCE_LENGTH = 10
PREDICTION_SMOOTHING_WINDOW = 5

# Performance settings
INFERENCE_THROTTLE = 0.05  # 50ms throttle

# Workout type mapping
WORKOUT_MAP = {
    0: "barbell bicep curl", 1: "bench press", 2: "chest fly machine", 
    3: "deadlift", 4: "decline bench press", 5: "hammer curl", 
    6: "hip thrust", 7: "incline bench press", 8: "lat pulldown", 
    9: "lateral raises", 10: "leg extensions", 11: "leg raises", 
    12: "plank", 13: "pull up", 14: "push ups", 15: "romanian deadlift", 
    16: "russian twist", 17: "shoulder press", 18: "squat", 
    19: "t bar row", 20: "tricep dips", 21: "tricep pushdown"
}

# Muscle group mapping
MUSCLE_GROUP_MAP = {
    1: "shoulders", 2: "chest", 3: "biceps", 4: "core",
    5: "triceps", 6: "legs", 7: "back"
}

# Workout to muscle group mapping
WORKOUT_TO_MUSCLE = {
    0: 3, 1: 2, 2: 2, 3: 7, 4: 2, 5: 3, 6: 6, 7: 2, 8: 7, 9: 1,
    10: 6, 11: 4, 12: 4, 13: 7, 14: 2, 15: 7, 16: 4, 17: 1, 18: 6, 19: 7,
    20: 5, 21: 5
}

# API configuration
BACKEND_BASE_URL = os.environ.get('BACKEND_URL', 'http://localhost:8000')
USAGE_ENDPOINT = f"{BACKEND_BASE_URL}/api/usage/"
PERFORMANCE_ENDPOINT = f"{BACKEND_BASE_URL}/api/model-performance/"

# Model version
MODEL_VERSION = "1.0.0"