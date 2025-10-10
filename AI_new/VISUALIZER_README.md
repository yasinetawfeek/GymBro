# Model Performance Visualizers

This directory contains comprehensive visualizers for evaluating the performance of the GymBro AI models. The visualizers analyze three different types of models:

1. **Workout Classifier** - Classifies workout types from pose sequences
2. **Muscle Group Classifier** - Identifies muscle groups being targeted
3. **Pose Optimizer** - Provides pose corrections and optimizations

## Files Overview

### Main Visualizers
- `model_performance_visualizer.py` - Combined visualizer for all three model types
- `workout_classifier_visualizer.py` - Dedicated workout classifier visualizer
- `muscle_group_classifier_visualizer.py` - Dedicated muscle group classifier visualizer
- `pose_optimizer_visualizer.py` - Dedicated pose optimizer visualizer

### Testing and Documentation
- `test_visualizers.py` - Test script to verify all visualizers work correctly
- `README.md` - This documentation file

## Features

### Workout Classifier Visualizer
- Evaluates LSTM, GRU, and Transformer models
- Shows prediction confidence and probability distributions
- Displays model agreement and performance comparison
- Provides detailed metrics and assessment

### Muscle Group Classifier Visualizer
- Evaluates muscle group classification models
- Shows accuracy against expected muscle groups
- Displays probability distributions across muscle groups
- Provides performance summary and assessment

### Pose Optimizer Visualizer
- Evaluates pose correction models
- Shows original vs corrected pose comparisons
- Displays frame-by-frame correction metrics
- Provides comprehensive error analysis

### Combined Visualizer
- Runs all three model types on the same video
- Provides comprehensive performance overview
- Generates multiple visualization types
- Offers complete evaluation summary

## Usage

### Prerequisites
Make sure you have the required dependencies installed:
```bash
pip install tensorflow numpy matplotlib seaborn opencv-python pillow scikit-learn
```

### Running Individual Visualizers

#### Workout Classifier Visualizer
```python
from workout_classifier_visualizer import WorkoutClassifierVisualizer

visualizer = WorkoutClassifierVisualizer()
results = visualizer.run_evaluation()
```

#### Muscle Group Classifier Visualizer
```python
from muscle_group_classifier_visualizer import MuscleGroupClassifierVisualizer

visualizer = MuscleGroupClassifierVisualizer()
results = visualizer.run_evaluation()
```

#### Pose Optimizer Visualizer
```python
from pose_optimizer_visualizer import PoseOptimizerVisualizer

visualizer = PoseOptimizerVisualizer()
results = visualizer.run_evaluation()
```

### Running the Combined Visualizer
```python
from model_performance_visualizer import ModelPerformanceVisualizer

visualizer = ModelPerformanceVisualizer()
results = visualizer.run_complete_evaluation()
```

### Running Tests
```bash
python test_visualizers.py
```

## How It Works

### Video Selection
Each visualizer automatically selects a random video from the dataset. You can specify a particular workout type:

```python
# Select a specific workout type
results = visualizer.run_evaluation(workout_type="squat")

# Or provide a specific video path
results = visualizer.run_evaluation(video_path="/path/to/video.mp4")
```

### Sequence Extraction
The visualizers extract a 90-frame sequence (3 seconds at 30fps) from the selected video:
- Uses MediaPipe to extract pose landmarks
- Handles videos shorter than 90 frames by padding
- Processes each frame to get 36-dimensional pose vectors

### Model Evaluation
Each visualizer loads the appropriate trained models and evaluates them:
- **Workout Classifier**: Predicts workout type with confidence scores
- **Muscle Group Classifier**: Predicts muscle groups with accuracy assessment
- **Pose Optimizer**: Generates pose corrections and calculates error metrics

### Visualization Generation
The visualizers create comprehensive plots including:
- Confidence comparisons
- Probability distributions
- Performance metrics
- Error analysis
- Model comparisons
- Summary statistics

## Model Requirements

### Expected Model Files
The visualizers expect the following model files to be present:

#### Workout Classifier Models
- `models/workout_classifier/lstm_model.keras`
- `models/workout_classifier/gru_model.keras`
- `models/workout_classifier/transformer_model.keras`
- `models/workout_classifier/label_encoder.pkl`

#### Muscle Group Classifier Models
- `models/muscle_group_classifier/lstm_model.keras`
- `models/muscle_group_classifier/gru_model.keras`
- `models/muscle_group_classifier/transformer_model.keras`

#### Pose Optimizer Models
- `models/pose_optimiser/lstm_model.keras`
- `models/pose_optimiser/gru_model.keras`
- `models/pose_optimiser/transformer_model.keras`

### Dataset Requirements
The visualizers expect the workout fitness video dataset to be available at:
```
/Users/yasinetawfeek/.cache/kagglehub/datasets/hasyimabdillah/workoutfitness-video/versions/5
```

## Customization

### Changing Dataset Path
You can specify a different dataset path when initializing visualizers:

```python
visualizer = WorkoutClassifierVisualizer(dataset_path="/path/to/your/dataset")
```

### Modifying Sequence Length
The default sequence length is 90 frames (3 seconds). You can modify this in the `extract_sequence_from_video` method.

### Adding New Metrics
You can extend the visualizers by adding new evaluation metrics in the `evaluate_models` methods.

## Output

Each visualizer generates:
1. **Console Output**: Detailed evaluation results and metrics
2. **Visualizations**: Multiple matplotlib plots showing performance analysis
3. **Return Values**: Structured data containing all results for further analysis

## Troubleshooting

### Common Issues

1. **Model Loading Errors**: Ensure all model files are present and accessible
2. **Dataset Not Found**: Check that the dataset path is correct
3. **Import Errors**: Make sure all dependencies are installed
4. **Memory Issues**: For large videos, consider reducing sequence length

### Debug Mode
Enable debug output by setting verbose flags in the visualizer classes.

## Contributing

To add new visualizations or metrics:
1. Extend the appropriate visualizer class
2. Add new methods for evaluation or visualization
3. Update the test script to include new functionality
4. Update this documentation

## License

This code is part of the GymBro project and follows the same licensing terms.
