from django.contrib import admin
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .viewsets import *

router = DefaultRouter()
# router.register(r'users', UserViewSet.as_view(), basename='user')
router.register(r'manage_accounts',AccountManagementViewSet, basename='manage_account')
router.register(r'approvals', ApprovalViewSet, basename='approval')

# urlpatterns = router.urls

urlpatterns = [
    path('', include(router.urls)),
    path('my_account/', UserViewSet.as_view(), name='my_account'),
    path('register/', RegisterView.as_view(), name='register'),
    path('train_model', TrainWorkoutClassiferViewSet.as_view({'post':'post'}),name='train_model'),
    path('predict_workout_classifer', PredictWorkoutClassiferViewSet.as_view({'post':'post'}),name='predict_workout_classifer'),
    path('stream-info/', StreamViewSet.as_view({'get': 'stream_info'}), name='stream-info'),
    path('get-token/', StreamViewSet.as_view({'get': 'get_token'}), name='get-token'),
]

"""
3 CRUD endpoints for users

GET /my_account/current id --> Returns current user who is logged in information

PUT /my_account/current id/ --> Updates the user with current id [Updates all user information]

PATCH /my_account/current id/ --> Updates the user with current id [Updates specific user information]
--------------------------------------------------------------------------------------------------------

6 CRUD endpoints for manage_accounts

GET /manage_accounts/ --> Returns a list of users

GET /manage_accounts/1/ --> Returns a single user with id 1

POST /manage_accounts/ --> Creates a new user

PUT /manage_accounts/1/ --> Updates the user with id 1 [Updates all user information]

PATCH /manage_accounts/1/ --> Updates the user with id 1 [Updates specific user information]

DELETE /manage_accounts/1/ --> Deletes the user with id 1
--------------------------------------------------------------------------------------------------------


to check more about all url in this app write python manage.py show_urls

"""
