import jwt
import time
import os
from django.http import JsonResponse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def stream_info(request):
    return JsonResponse({
        "playback_url": "http://localhost:8000/media/videos/stream.m3u8",
        "hls_config": {
            "max_buffer_length": 30,
            "live_sync_duration": 10
        }
    })

def get_token(request):
    try:
        # Get API key and secret from environment variables
        api_key = os.getenv('VIDEOSDK_API_KEY')
        api_secret = os.getenv('VIDEOSDK_API_SECRET')
        
        if not api_key:
            return JsonResponse({"error": "API key not found in environment variables"}, status=500)
        
        if not api_secret:
            # For development purposes only - use a consistent secret for testing
            # In production, this should be loaded from environment variables
            api_secret = "ZaqXsw123EdcRfvBgt567" # This is just for development, replace with real secret
            print("Warning: API secret not found in environment, using development fallback")
        
        # Generate JWT token according to VideoSDK requirements
        payload = {
            'api_key': api_key,  # Changed from 'apikey' to 'api_key'
            'permissions': ['allow_join', 'allow_mod'],
            'iat': int(time.time()),
            'exp': int(time.time()) + 86400  # Token expires in 1 day
        }
        
        # Generate JWT token with proper secret
        token = jwt.encode(payload, api_secret, algorithm='HS256')
        
        # Debug output
        print(f"Generated token for API key {api_key}: {token[:15]}...")
        
        return JsonResponse({'token': token})
        
    except Exception as e:
        print(f"Error generating token: {str(e)}")
        return JsonResponse({"error": "Failed to generate token"}, status=500)