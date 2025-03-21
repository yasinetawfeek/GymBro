from django.contrib import admin
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .viewsets import *

# Create a router for our viewsets
router = DefaultRouter()

# Define URL patterns
urlpatterns = [
    # Stream API endpoints
    path('stream-info/', StreamViewSet.as_view({'get': 'stream_info'}), name='stream-info'),
    path('get-token/', StreamViewSet.as_view({'get': 'get_token'}), name='get-token'),
    
    # Add the router URLs
    path('', include(router.urls)),
]

