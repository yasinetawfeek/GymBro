"""
Models package for the AI service.

This module contains all neural network model definitions.
"""

from .pose_model import EnhancedPoseModel
from .workout_classifier import LSTMWorkoutClassifier

__all__ = [
    'EnhancedPoseModel',
    'LSTMWorkoutClassifier',
]