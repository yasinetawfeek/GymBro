"""
Streaming and video-related viewsets for the GymBro application.
"""
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework.decorators import action
import requests
import jwt
import time
import os
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi


class StreamViewSet(viewsets.ViewSet):
    """API endpoints for video streaming functionality."""
    
    @swagger_auto_schema(
        operation_description="Get HLS stream information",
        operation_summary="Get stream information",
        responses={200: openapi.Response(
            description="Stream information including playback URL and configuration",
            examples={
                "application/json": {
                    "playback_url": "http://localhost:8000/media/videos/stream.m3u8",
                    "hls_config": {
                        "max_buffer_length": 30,
                        "live_sync_duration": 10
                    }
                }
            }
        )}
    )
    @action(detail=False, methods=['get'], permission_classes=[AllowAny])
    def stream_info(self, request):
        """Get HLS stream information."""
        return Response({
            "playback_url": "http://localhost:8000/media/videos/stream.m3u8",
            "hls_config": {
                "max_buffer_length": 30,
                "live_sync_duration": 10
            }
        })
    
    @swagger_auto_schema(
        operation_description="Generate a VideoSDK JWT token for authentication",
        operation_summary="Get VideoSDK token",
        responses={
            200: openapi.Response(
                description="Token generated successfully",
                examples={"application/json": {"token": "eyJhbGciOiJIUzI1..."}}
            ),
            500: openapi.Response(
                description="Failed to generate token",
                examples={"application/json": {"error": "API key not found in environment variables"}}
            )
        }
    )
    @action(detail=False, methods=['get'], permission_classes=[AllowAny])
    def get_token(self, request):
        """Generate a VideoSDK JWT token for authentication."""
        try:
            # Get API key and secret from environment variables
            api_key = os.getenv('VIDEOSDK_API_KEY')
            api_secret = os.getenv('VIDEOSDK_API_SECRET')
            
            if not api_key:
                return Response(
                    {"error": "API key not found in environment variables"}, 
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            
            if not api_secret:
                # For development purposes only - use a consistent secret for testing
                # In production, this should be loaded from environment variables
                api_secret = "ZaqXsw123EdcRfvBgt567"  # Development fallback
                print("Warning: API secret not found in environment, using development fallback")
            
            # Generate JWT token according to VideoSDK requirements
            payload = {
                'api_key': api_key,
                'permissions': ['allow_join', 'allow_mod'],
                'iat': int(time.time()),
                'exp': int(time.time()) + 86400  # Token expires in 1 day
            }
            
            # Generate JWT token with proper secret
            token = jwt.encode(payload, api_secret, algorithm='HS256')
            
            return Response({"token": token})
            
        except Exception as e:
            return Response(
                {"error": f"Failed to generate token: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )