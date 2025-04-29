from rest_framework.generics import CreateAPIView
from django.contrib.auth.models import User
from rest_framework import viewsets, status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes, action
from rest_framework.permissions import AllowAny
from .serializers import *
from rest_framework.permissions import IsAdminUser, IsAuthenticated
from .permissions import IsOwner, IsMachineLearningExpert, IsApprovedUser, IsAdminRole, IsAIEngineerRole, IsApprovedAIEngineer
from rest_framework import generics
from rest_framework.response import Response
from rest_framework.exceptions import PermissionDenied
import requests

import jwt
import time
import os
from dotenv import load_dotenv
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

# Load environment variables
load_dotenv()

class StreamViewSet(viewsets.ViewSet):
    """API endpoints for video streaming functionality"""
    
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
    @permission_classes([AllowAny])
    def stream_info(self, request):
        """Get HLS stream information"""
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
    @permission_classes([AllowAny])
    def get_token(self, request):
        """Generate a VideoSDK JWT token for authentication"""
        try:
            # Get API key and secret from environment variables
            api_key = os.getenv('VIDEOSDK_API_KEY')
            api_secret = os.getenv('VIDEOSDK_API_SECRET')
            
            if not api_key:
                return Response({"error": "API key not found in environment variables"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
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
            
            return Response({'token': token})
            
        except Exception as e:
            print(f"Error generating token: {str(e)}")
            return Response({"error": "Failed to generate token"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

"""
No need to put IsAuthenticated inside each permission class, since it is already in settings [on project-level by default]
meaning that you cannot access any viewset here without being authenticated first

"""

class UserViewSet(generics.RetrieveUpdateAPIView):
    """
    This is for user to change their personal account information or to view it only
    however they cannot access other users personal information
    """
    permission_classes = [IsAuthenticated] # User can update their own profile
    serializer_class = UserCreateSerializer
    #the reason why we do not set queryset to all objects, to prevent user from accessing other users' information

    def get_object(self):
        return self.request.user  # Only user can access their information even if they pass a different id
    
    def get_serializer_context(self):
        context = super().get_serializer_context()
        # Add flag indicating this is a self-update
        context['is_self_update'] = True
        return context


class AccountManagementViewSet(viewsets.ModelViewSet):
    """
    This is for admin to view and manage all users.
    Admins can view, update, delete, create users.
    """
    permission_classes = [IsAuthenticated, IsAdminRole] # Only users with Admin role can access
    queryset = User.objects.all()
    serializer_class = UserCreateSerializer

    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()

        # Get current user's role
        current_user_group = request.user.groups.first()
        current_rolename = current_user_group.name if current_user_group else None
        
        # Get target user's role
        target_user_group = instance.groups.first()
        target_rolename = target_user_group.name if target_user_group else None
        
        # Only check role conflict if both users have roles
        if current_rolename and target_rolename and current_rolename == target_rolename:
            raise PermissionDenied("User cannot delete other users with the same role")
            
        return super().destroy(request, *args, **kwargs)

class UserActiveCountViewSet(viewsets.ViewSet):
    permission_classes = [IsAuthenticated, IsAdminRole]  # Only admins can access
    def list(self, request):
        total_number_of_users = User.objects.count()
        total_number_of_active_users = User.objects.filter(is_active=True).count()
        total_number_of_inactive_users = User.objects.filter(is_active=False).count()
        data = {
            "total_number_of_users": total_number_of_users,
            "total_number_of_active_users": total_number_of_active_users,
            "total_number_of_inactive_users": total_number_of_inactive_users,
        }

        return Response(data)
    


"""
Machine Learning ViewSets below for both training and predicting
AI as service will run through python flask as API, this API will only be accessible through django
viewSets only if user has been authenticated and his group has the permission to access these api views
e.g. only Admin, AI Engineer can do that but customer shall only use AI model for prediction but not for
training it or modifying it.

"""
# FLASK_MACHINE_LEARNING_API_URL="http://10.167.143.148:8001/train_model"
FLASK_MACHINE_LEARNING_API_URL="http://ai:8001/train_model"
FLASK_MACHINE_LEARNING_API_PREDICT_URL="http://10.167.143.148:8001/predict_classify_workout"

API_KEY = "job_hunting_ai_memory_leakage"

class TrainWorkoutClassiferViewSet(viewsets.ViewSet):
    """
    ViewSet only accessible by Admin and approved AI Engineers
    """
    permission_classes = [IsAuthenticated, IsApprovedUser, IsMachineLearningExpert] 

    def post(self, request):
        user = request.user
        permissions = user.get_all_permissions()
        print("all permissions", permissions)
        
        headers = {
            "X-API-KEY": API_KEY, 
            "Content-Type": "application/json",
        }
        machine_learning_accuracy = requests.post(FLASK_MACHINE_LEARNING_API_URL, json=request.data, headers=headers)
        return Response(machine_learning_accuracy.json(), status=machine_learning_accuracy.status_code)
    

class PredictWorkoutClassiferViewSet(viewsets.ViewSet):
    """
    ViewSet is accessed by anyone who has registered and has been authenticated
    """
    permission_classes = [IsAuthenticated, IsApprovedUser]  # Must be an approved user
    def post(self, request):
        headers = {
            "X-API-KEY": API_KEY, 
            "Content-Type": "application/json",
        }
        machine_learning_prediction = requests.post(FLASK_MACHINE_LEARNING_API_PREDICT_URL, json=request.data, headers=headers)
        return Response(machine_learning_prediction.json(), status=machine_learning_prediction.status_code)

# Add this new view for registration
class RegisterView(generics.CreateAPIView):
    """
    API view for user registration with role selection.
    """
    serializer_class = UserCreateSerializer
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        # Create a copy of the data to manipulate
        data = request.data.copy()
        
        # Default role is Customer if none is provided
        if 'group' not in data:
            data['group'] = 'Customer'
        
        # Set default approval status based on role
        if data['group'] in ['Admin', 'AI Engineer']:
            data['is_approved'] = False
        else:
            data['is_approved'] = True
            
        serializer = self.get_serializer(data=data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        
        return Response(
            {"detail": "User registered successfully. Please log in."},
            status=status.HTTP_201_CREATED
        )

# Add an approval endpoint for admins
class ApprovalViewSet(viewsets.ViewSet):
    """
    API endpoints for admin to approve/reject users.
    """
    permission_classes = [IsAuthenticated, IsAdminRole]  # Only admins can approve/reject
    
    @action(detail=True, methods=['post'])
    def approve(self, request, pk=None):
        try:
            user = User.objects.get(pk=pk)
            user.profile.is_approved = True
            user.profile.save()
            return Response({"detail": f"User {user.username} has been approved."})
        except User.DoesNotExist:
            return Response(
                {"detail": "User not found."}, 
                status=status.HTTP_404_NOT_FOUND
            )
    
    @action(detail=True, methods=['post'])
    def reject(self, request, pk=None):
        try:
            user = User.objects.get(pk=pk)
            user.profile.is_approved = False
            user.profile.save()
            return Response({"detail": f"User {user.username} has been rejected."})
        except User.DoesNotExist:
            return Response(
                {"detail": "User not found."}, 
                status=status.HTTP_404_NOT_FOUND
            )
    
    @action(detail=False, methods=['get'])
    def pending(self, request):
        # Get users who are not approved
        pending_users = User.objects.filter(profile__is_approved=False)
        serializer = UserCreateSerializer(pending_users, many=True)
        return Response(serializer.data)

# New endpoint for role information
class RoleInfoViewSet(viewsets.ViewSet):
    """
    API endpoint to get role information for the current user
    """
    permission_classes = [IsAuthenticated]
    
    def list(self, request):
        user = request.user
        role = user.groups.first().name if user.groups.exists() else "No Role"
        is_approved = user.profile.is_approved if hasattr(user, 'profile') else False
        
        return Response({
            "username": user.username,
            "email": user.email,
            "role": role,
            "is_approved": is_approved,
            "is_admin": role == "Admin",
            "is_ai_engineer": role == "AI Engineer",
            "is_customer": role == "Customer"
        })
    
    @action(detail=True, methods=['get'])
    def user_approval_status(self, request, pk=None):
        """
        Get approval status for a specific user - only accessible by admins
        """
        try:
            # Only admins can check another user's approval status
            if not request.user.groups.filter(name='Admin').exists():
                return Response(
                    {"detail": "You do not have permission to check other users' approval status."},
                    status=status.HTTP_403_FORBIDDEN
                )
            
            user = User.objects.get(pk=pk)
            role = user.groups.first().name if user.groups.exists() else "No Role"
            is_approved = user.profile.is_approved if hasattr(user, 'profile') else False
            
            return Response({
                "username": user.username,
                "email": user.email,
                "role": role,
                "is_approved": is_approved,
                "is_admin": role == "Admin",
                "is_ai_engineer": role == "AI Engineer",
                "is_customer": role == "Customer"
            })
        except User.DoesNotExist:
            return Response(
                {"detail": "User not found."}, 
                status=status.HTTP_404_NOT_FOUND
            )

# Add this after the RoleInfoViewSet
class BillingViewSet(viewsets.ModelViewSet):
    """
    API endpoint for billing records with date filtering capabilities.
    Only accessible by Admin users.
    """
    permission_classes = [IsAuthenticated, IsAdminRole]
    serializer_class = BillingRecordSerializer
    queryset = BillingRecord.objects.all()
    
    def get_queryset(self):
        queryset = BillingRecord.objects.all()
        
        # Filter by date range if provided
        start_date = self.request.query_params.get('start_date', None)
        end_date = self.request.query_params.get('end_date', None)
        
        if start_date and end_date:
            queryset = queryset.filter(billing_date__gte=start_date, billing_date__lte=end_date)
        elif start_date:
            queryset = queryset.filter(billing_date__gte=start_date)
        elif end_date:
            queryset = queryset.filter(billing_date__lte=end_date)
        
        # Filter by status if provided
        status = self.request.query_params.get('status', None)
        if status:
            queryset = queryset.filter(status=status)
        
        # Filter by subscription type if provided
        subscription = self.request.query_params.get('subscription', None)
        if subscription:
            queryset = queryset.filter(subscription_type=subscription)
            
        # Filter by username if provided
        username = self.request.query_params.get('username', None)
        if username:
            queryset = queryset.filter(user__username__icontains=username)
        
        return queryset
    
    @action(detail=False, methods=['get'])
    def summary(self, request):
        """
        Get a summary of billing data: total revenue, counts by status, etc.
        """
        # Date range filtering
        start_date = self.request.query_params.get('start_date', None)
        end_date = self.request.query_params.get('end_date', None)
        
        queryset = self.get_queryset()
        
        total_amount = sum(record.amount for record in queryset)
        total_records = queryset.count()
        status_counts = {}
        subscription_counts = {}
        
        # Count records by status
        for status, _ in BillingRecord.STATUS_CHOICES:
            status_counts[status] = queryset.filter(status=status).count()
        
        # Count records by subscription type
        for sub_type, _ in BillingRecord.SUBSCRIPTION_CHOICES:
            subscription_counts[sub_type] = queryset.filter(subscription_type=sub_type).count()
        
        return Response({
            'total_amount': total_amount,
            'total_records': total_records,
            'status_counts': status_counts,
            'subscription_counts': subscription_counts,
            'date_range': {
                'start_date': start_date,
                'end_date': end_date
            }
        })




