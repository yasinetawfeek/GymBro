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
        # Get API key from environment variables
        api_key = os.getenv('VIDEOSDK_API_KEY')
        
        if not api_key:
            return JsonResponse({"error": "API key not found in environment variables"}, status=500)
        
        # Generate JWT token locally - this works for development
        payload = {
            'apikey': api_key,
            'permissions': ['allow_join', 'allow_mod'],  # Add necessary permissions
            'iat': int(time.time()),
            'exp': int(time.time()) + 86400  # Token expires in 1 day
        }
        
        # Generate JWT token
        token = jwt.encode(payload, "your-dummy-secret", algorithm='HS256')
        
        return JsonResponse({'token': token})
        
    except Exception as e:
        print(f"Error generating token: {str(e)}")
        return JsonResponse({"error": "Failed to generate token"}, status=500)