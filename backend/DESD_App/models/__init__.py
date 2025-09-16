"""
Models package for the GymBro application.

This module organizes all models into logical groups for better maintainability.
"""

# Import all models to make them available at the package level
from .user import UserProfile
from .billing import Subscription, Invoice, BillingRecord
from .analytics import UsageRecord, ModelPerformanceMetric, UserLastViewedExercise
from .ml_models import MLModel

# Define __all__ to control what gets imported with "from models import *"
__all__ = [
    'UserProfile',
    'Subscription', 
    'Invoice', 
    'BillingRecord',
    'UsageRecord',
    'ModelPerformanceMetric',
    'UserLastViewedExercise',
    'MLModel',
]