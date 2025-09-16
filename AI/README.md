# GymBro AI Service - Real-time Pose Analysis

The AI service provides real-time pose correction and workout classification using PyTorch models. It processes pose landmarks from MediaPipe and provides corrections and predictions via WebSocket connections.

## 🏗️ Architecture

The AI service is built with Flask and SocketIO, featuring a clean, modular architecture:

```
AI/
├── app/
│   ├── models/              # Neural network model definitions
│   │   ├── pose_model.py    # Pose correction model
│   │   └── workout_classifier.py # Workout classification model
│   ├── services/            # Service classes for predictions
│   │   ├── model_loader.py  # Model loading utilities
│   │   └── prediction_service.py # Prediction services
│   ├── config.py           # Configuration constants
│   ├── main.py            # Original main application
│   └── main_clean.py      # Clean, organized main application
├── models/                 # Pre-trained model files
│   ├── pose_correctors/
│   ├── workout_classifiers/
│   └── muscle_group_classifiers/
├── data/                   # Training data and processed files
├── Dockerfile.clean       # Production-ready Dockerfile
└── requirements.txt       # Python dependencies
```

## 🚀 Quick Start

### Development Setup

```bash
# Start development environment
make dev

# Or manually with Docker
docker-compose -f docker-compose.clean.yml --profile development up ai-dev
```

### Production Setup

```bash
# Start production environment
make prod

# Or manually with Docker
docker-compose -f docker-compose.clean.yml up ai
```

## 🧠 AI Models

### Pose Correction Model
- **Type**: Deep Neural Network (DNN)
- **Input**: 37 features (workout type + 36 pose landmarks)
- **Output**: 36 correction values (x, y, z for 12 keypoints)
- **Architecture**: 3-layer MLP with LeakyReLU activation
- **Purpose**: Provides real-time pose corrections

### Workout Classification Model
- **Type**: LSTM with Attention Mechanism
- **Input**: Sequence of 36 pose features (10 frames)
- **Output**: 22 workout types
- **Architecture**: LSTM + Attention + Dense layers
- **Purpose**: Classifies workout type from pose sequences

### Muscle Group Classification Model
- **Type**: Random Forest Classifier
- **Input**: 36 pose features
- **Output**: 7 muscle groups
- **Purpose**: Identifies activated muscle groups

## 🔌 WebSocket API

### Connection
```javascript
const socket = io('http://localhost:8001', {
  transports: ['websocket'],
  query: { token: 'your-jwt-token' }
});
```

### Events

#### Client → Server

**`pose_data`**
```javascript
socket.emit('pose_data', {
  landmarks: [
    { x: 0.5, y: 0.3, z: 0.1, visibility: 0.9 },
    // ... 33 more landmarks
  ],
  selected_workout: 12, // workout type
  timestamp: Date.now()
});
```

#### Server → Client

**`connected`**
```javascript
socket.on('connected', (data) => {
  console.log('Connected:', data);
  // { client_id: 'abc123', authenticated: true }
});
```

**`pose_corrections`**
```javascript
socket.on('pose_corrections', (data) => {
  console.log('Corrections:', data);
  // {
  //   corrections: {
  //     '11': { x: 0.01, y: -0.02, z: 0.0 },
  //     '12': { x: -0.01, y: 0.01, z: 0.0 },
  //     // ... more joint corrections
  //   },
  //   predicted_workout_type: 12,
  //   predicted_muscle_group: 4
  // }
});
```

**`error`**
```javascript
socket.on('error', (error) => {
  console.error('AI Service Error:', error);
});
```

## 📊 Configuration

### Model Configuration
```python
# config.py
BODY_KEYPOINTS_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
LANDMARK_DIM = 3  # x, y, z
FLAT_LANDMARK_SIZE = 36  # 12 keypoints * 3 dimensions

SEQUENCE_LENGTH = 10  # Frames for LSTM
PREDICTION_SMOOTHING_WINDOW = 5
INFERENCE_THROTTLE = 0.05  # 50ms throttle
```

### Workout Types
```python
WORKOUT_MAP = {
    0: "barbell bicep curl", 1: "bench press", 2: "chest fly machine",
    3: "deadlift", 4: "decline bench press", 5: "hammer curl",
    6: "hip thrust", 7: "incline bench press", 8: "lat pulldown",
    9: "lateral raises", 10: "leg extensions", 11: "leg raises",
    12: "plank", 13: "pull up", 14: "push ups", 15: "romanian deadlift",
    16: "russian twist", 17: "shoulder press", 18: "squat",
    19: "t bar row", 20: "tricep dips", 21: "tricep pushdown"
}
```

### Muscle Groups
```python
MUSCLE_GROUP_MAP = {
    1: "shoulders", 2: "chest", 3: "biceps", 4: "core",
    5: "triceps", 6: "legs", 7: "back"
}
```

## 🔧 Services

### PoseCorrectionService
Handles pose correction predictions with throttling.

```python
class PoseCorrectionService:
    def __init__(self, pose_model, device):
        self.pose_model = pose_model
        self.device = device
        self.last_inference_time = 0
    
    def get_pose_corrections(self, landmarks, workout_type=0):
        # Returns pose corrections or None if throttled
        pass
```

### WorkoutClassificationService
Manages workout classification with sequence buffering.

```python
class WorkoutClassificationService:
    def __init__(self, workout_classifier, feature_scaler, label_encoder, device):
        self.workout_classifier = workout_classifier
        self.feature_scaler = feature_scaler
        self.label_encoder = label_encoder
        self.device = device
        self.client_pose_buffers = {}
        self.client_workout_predictions = {}
    
    def predict_workout_from_sequence(self, client_id, current_features, sequence_length=10):
        # Returns predicted workout type and name
        pass
```

### MuscleGroupClassificationService
Handles muscle group classification.

```python
class MuscleGroupClassificationService:
    def __init__(self, muscle_group_classifier):
        self.muscle_group_classifier = muscle_group_classifier
        self.client_muscle_predictions = {}
    
    def predict_muscle_group_from_sequence(self, client_id, current_features, workout_type):
        # Returns predicted muscle group and name
        pass
```

## 📈 Performance Optimization

### Inference Throttling
- 50ms minimum interval between pose corrections
- Prevents excessive GPU/CPU usage
- Maintains smooth user experience

### Sequence Buffering
- Buffers 10 frames for LSTM predictions
- Smooths predictions over time
- Reduces prediction noise

### Model Loading
- Models loaded once at startup
- Warmup inference for GPU initialization
- Efficient memory management

## 🔍 Monitoring & Analytics

### Health Check
- Endpoint: `/health`
- Returns model loading status
- Service version information

### Performance Metrics
- Response time tracking
- Confidence score monitoring
- Frame processing rate
- Model accuracy metrics

### Session Tracking
- Client session management
- Usage analytics
- Performance reporting to backend

## 🧪 Testing

### Model Testing
```bash
# Test model loading
python -c "from app.services.model_loader import load_pose_model; print(load_pose_model('cpu'))"

# Test predictions
python -c "from app.services.prediction_service import PoseCorrectionService; print('OK')"
```

### WebSocket Testing
```javascript
// Test WebSocket connection
const socket = io('http://localhost:8001');
socket.on('connect', () => console.log('Connected!'));
socket.emit('pose_data', { landmarks: [], selected_workout: 0 });
```

## 🚀 Deployment

### Docker Deployment
```bash
# Build image
docker build -f Dockerfile.clean -t gymbro-ai .

# Run container
docker run -p 8001:8001 gymbro-ai
```

### Production Considerations
- Use CPU-only PyTorch for smaller image size
- Enable health checks
- Monitor memory usage
- Set up logging
- Configure proper timeouts

## 🔒 Security

### Authentication
- JWT token verification for WebSocket connections
- Backend integration for user validation
- Session-based client tracking

### Input Validation
- Landmark data validation
- Workout type validation
- Error handling and logging

## 📚 Model Training

### Data Preparation
```python
# Process MediaPipe landmarks
landmarks = extract_keypoints(mediapipe_results)
features = normalize_landmarks(landmarks)
```

### Training Scripts
- `workout_classifer_model.py`: LSTM training
- `muscle_group_classifer.py`: Random Forest training
- `displacement.py`: Pose correction training

### Model Evaluation
- Cross-validation metrics
- Confusion matrices
- Performance plots

## 🤝 Contributing

### Development Workflow
1. Create feature branch
2. Make changes to models/services
3. Test with sample data
4. Update documentation
5. Submit pull request

### Code Standards
- Follow PEP 8
- Add type hints
- Write docstrings
- Test new features

## 🆘 Troubleshooting

### Common Issues

**Model Loading Errors**
```bash
# Check model files exist
ls -la models/pose_correctors/
ls -la models/workout_classifiers/

# Check file permissions
chmod 644 models/*/*.pth
```

**WebSocket Connection Issues**
```bash
# Check service status
curl http://localhost:8001/health

# Check logs
make logs-ai
```

**Performance Issues**
```bash
# Monitor resource usage
docker stats gymbro-ai

# Check inference times in logs
grep "inference" logs/ai.log
```

### Debug Mode
```bash
# Enable debug logging
export FLASK_ENV=development
make restart
```

---

**AI Service** - The brain of the GymBro platform 🧠🤖