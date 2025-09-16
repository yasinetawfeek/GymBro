"""
ViewSets package for the GymBro application.

This module organizes all viewsets into logical groups for better maintainability.
"""

# Import all viewsets to make them available at the package level
from .user_management import UserViewSet, AccountManagementViewSet, ApprovalViewSet, RoleInfoViewSet, RegisterView
from .billing import SubscriptionViewSet, InvoiceViewSet, BillingViewSet
from .analytics import UsageTrackingViewSet, ModelPerformanceViewSet, UserLastViewedExerciseViewSet
from .ml_models import MLModelViewSet, TrainWorkoutClassiferViewSet, PredictWorkoutClassiferViewSet
from .streaming import StreamViewSet

# Define __all__ to control what gets imported with "from viewsets import *"
__all__ = [
    'UserViewSet',
    'AccountManagementViewSet', 
    'ApprovalViewSet',
    'RoleInfoViewSet',
    'RegisterView',
    'SubscriptionViewSet',
    'InvoiceViewSet',
    'BillingViewSet',
    'UsageTrackingViewSet',
    'ModelPerformanceViewSet',
    'UserLastViewedExerciseViewSet',
    'MLModelViewSet',
    'TrainWorkoutClassiferViewSet',
    'PredictWorkoutClassiferViewSet',
    'StreamViewSet',
]