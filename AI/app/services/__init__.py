"""
Services package for the AI service.

This module contains all service classes for model loading and predictions.
"""

from .model_loader import load_pose_model, load_workout_classifier, load_muscle_group_classifier
from .prediction_service import PoseCorrectionService, WorkoutClassificationService, MuscleGroupClassificationService

__all__ = [
    'load_pose_model',
    'load_workout_classifier', 
    'load_muscle_group_classifier',
    'PoseCorrectionService',
    'WorkoutClassificationService',
    'MuscleGroupClassificationService',
]