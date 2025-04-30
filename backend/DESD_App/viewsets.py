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
import math
from datetime import datetime, timedelta
from dotenv import load_dotenv
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from django.db.models import Count

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
        
        # Create usage tracking record
        start_time = time.time()
        
        # Make API call to Flask ML service
        machine_learning_accuracy = requests.post(FLASK_MACHINE_LEARNING_API_URL, json=request.data, headers=headers)
        
        # Calculate response time
        response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Track usage for billing purposes
        UsageTracking.objects.create(
            user=request.user,
            endpoint='train_model',
            response_time=response_time,
            status_code=machine_learning_accuracy.status_code,
            request_data=request.data,
            response_data=machine_learning_accuracy.json() if machine_learning_accuracy.status_code == 200 else None,
            ip_address=request.META.get('REMOTE_ADDR')
        )
        
        # Update user quota
        try:
            quota, created = UsageQuota.objects.get_or_create(
                user=request.user,
                defaults={'reset_date': (datetime.now() + timedelta(days=30)).date()}
            )
            quota.api_calls_used += 1
            
            # Assuming each training request uses 1000MB (1GB) of data
            data_usage = 1000.0
            quota.data_usage = float(quota.data_usage) + data_usage
            quota.save()
        except Exception as e:
            print(f"Error updating quota: {str(e)}")
        
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
        
        # Create usage tracking record
        start_time = time.time()
        
        # Make API call to Flask ML service
        machine_learning_prediction = requests.post(FLASK_MACHINE_LEARNING_API_PREDICT_URL, json=request.data, headers=headers)
        
        # Calculate response time
        response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Track usage for billing purposes
        UsageTracking.objects.create(
            user=request.user,
            endpoint='predict_model',
            response_time=response_time,
            status_code=machine_learning_prediction.status_code,
            request_data=request.data,
            response_data=machine_learning_prediction.json() if machine_learning_prediction.status_code == 200 else None,
            ip_address=request.META.get('REMOTE_ADDR')
        )
        
        # Update user quota
        try:
            quota, created = UsageQuota.objects.get_or_create(
                user=request.user,
                defaults={'reset_date': (datetime.now() + timedelta(days=30)).date()}
            )
            quota.api_calls_used += 1
            
            # Assuming each prediction request uses 10MB of data
            data_usage = 10.0
            quota.data_usage = float(quota.data_usage) + data_usage
            quota.save()
            
            # Create invoice record for the usage if it's a customer
            if request.user.groups.filter(name='Customer').exists():
                # Check if user has an active subscription
                subscription = Subscription.objects.filter(
                    user=request.user, 
                    is_active=True, 
                    start_date__lte=datetime.now().date(),
                    end_date__gte=datetime.now().date()
                ).first()
                
                # If no subscription exists, create a free subscription
                if not subscription:
                    subscription = Subscription.objects.create(
                        user=request.user,
                        plan='free',
                        start_date=datetime.now().date(),
                        end_date=(datetime.now() + timedelta(days=30)).date(),
                        price=0.00,
                        max_api_calls=100,
                        max_data_usage=1000
                    )
                
                # Check if user has exceeded their quota
                if quota.api_calls_used > subscription.max_api_calls or float(quota.data_usage) > subscription.max_data_usage:
                    # Calculate overage charges
                    api_overage = max(0, quota.api_calls_used - subscription.max_api_calls)
                    data_overage = max(0, float(quota.data_usage) - subscription.max_data_usage)
                    
                    # Rate per overage based on subscription plan
                    if subscription.plan == 'free':
                        api_rate = 0.01  # $0.01 per call
                        data_rate = 0.005  # $0.005 per MB
                    elif subscription.plan == 'basic':
                        api_rate = 0.005
                        data_rate = 0.002
                    elif subscription.plan == 'premium':
                        api_rate = 0.002
                        data_rate = 0.001
                    else:  # enterprise
                        api_rate = 0.001
                        data_rate = 0.0005
                    
                    # Calculate charges
                    api_charge = api_overage * api_rate
                    data_charge = data_overage * data_rate
                    total_charge = api_charge + data_charge
                    
                    # Only create invoice if there's a charge
                    if total_charge > 0:
                        Invoice.objects.create(
                            subscription=subscription,
                            user=request.user,
                            amount=total_charge,
                            due_date=(datetime.now() + timedelta(days=15)).date(),
                            status='pending',
                            description=f"Overage charges: {api_overage} API calls, {data_overage} MB data"
                        )
                
                # Create a billing record for this usage
                BillingRecord.objects.create(
                    user=request.user,
                    amount=0.00,  # Will be updated later if needed
                    subscription_type=subscription.plan,
                    billing_date=datetime.now().date(),
                    due_date=(datetime.now() + timedelta(days=15)).date(),
                    status='pending',
                    description='ML prediction service usage',
                    api_calls=1,
                    data_usage=10.0  # 10MB per prediction
                )
            
        except Exception as e:
            print(f"Error updating quota or creating invoice: {str(e)}")
        
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
    
    @action(detail=False, methods=['get'])
    def billable_activity(self, request):
        """
        Get all billable activity on the system in a given time period.
        This combines data from BillingRecord, Invoices, and UsageTracking.
        """
        # Date range filtering is required for this endpoint
        start_date = self.request.query_params.get('start_date', None)
        end_date = self.request.query_params.get('end_date', None)
        
        if not start_date or not end_date:
            return Response(
                {"detail": "Both start_date and end_date parameters are required"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Filter by username if provided
        username = self.request.query_params.get('username', None)
        user_filter = {}
        if username:
            user_filter = {'user__username__icontains': username}
            
        # Filter by subscription type if provided
        subscription_type = self.request.query_params.get('subscription_type', None)
        sub_filter = {}
        if subscription_type:
            sub_filter = {'subscription_type': subscription_type}
            
        # Filter by status if provided
        bill_status = self.request.query_params.get('status', None)
        status_filter = {}
        if bill_status:
            status_filter = {'status': bill_status}
            
        # Get billing records in the date range
        billing_records = BillingRecord.objects.filter(
            billing_date__gte=start_date,
            billing_date__lte=end_date,
            **user_filter,
            **sub_filter,
            **status_filter
        )
        
        # Get invoices in the date range with same filters
        invoices = Invoice.objects.filter(
            invoice_date__gte=start_date,
            invoice_date__lte=end_date,
            **user_filter
        )
        if bill_status:
            invoices = invoices.filter(status=bill_status)
            
        # Get usage tracking in the date range with username filter if provided
        usage_records = UsageTracking.objects.filter(
            timestamp__date__gte=start_date,
            timestamp__date__lte=end_date,
            **user_filter
        )
        
        # Implement basic pagination
        page_size = int(request.query_params.get('page_size', 50))
        page = int(request.query_params.get('page', 1))
        
        # Apply pagination to each dataset
        start_idx = (page - 1) * page_size
        end_idx = page * page_size
        
        billing_records_paginated = billing_records[start_idx:end_idx]
        invoices_paginated = invoices[start_idx:end_idx]
        usage_records_paginated = usage_records[start_idx:end_idx]
        
        # Serialize each dataset
        billing_serializer = BillingRecordSerializer(billing_records_paginated, many=True)
        invoice_serializer = InvoiceSerializer(invoices_paginated, many=True)
        usage_serializer = UsageTrackingSerializer(usage_records_paginated, many=True)
        
        # Calculate summary metrics
        total_billed_amount = sum(record.amount for record in billing_records)
        total_invoice_amount = sum(invoice.amount for invoice in invoices)
        total_api_calls = sum(record.api_calls for record in billing_records)
        total_data_usage = sum(float(record.data_usage) for record in billing_records)
        
        # Get unique users with billing activity
        active_users = set(record.user.id for record in billing_records)
        active_users.update(invoice.user.id for invoice in invoices)
        active_users.update(record.user.id for record in usage_records)
        
        # Get activity by subscription type
        subscription_activity = {}
        for sub_type, _ in BillingRecord.SUBSCRIPTION_CHOICES:
            sub_records = billing_records.filter(subscription_type=sub_type)
            subscription_activity[sub_type] = {
                'count': sub_records.count(),
                'total_amount': sum(record.amount for record in sub_records),
                'api_calls': sum(record.api_calls for record in sub_records),
                'data_usage': sum(float(record.data_usage) for record in sub_records)
            }
        
        return Response({
            'date_range': {
                'start_date': start_date,
                'end_date': end_date
            },
            'summary': {
                'total_billed_amount': total_billed_amount,
                'total_invoice_amount': total_invoice_amount,
                'total_api_calls': total_api_calls,
                'total_data_usage': total_data_usage,
                'active_users_count': len(active_users),
                'total_records_count': billing_records.count() + invoices.count() + usage_records.count()
            },
            'pagination': {
                'page': page,
                'page_size': page_size,
                'total_pages': max(
                    1, 
                    math.ceil(max(
                        billing_records.count(), 
                        invoices.count(), 
                        usage_records.count()
                    ) / page_size)
                )
            },
            'subscription_activity': subscription_activity,
            'billing_records': billing_serializer.data,
            'invoices': invoice_serializer.data,
            'usage_records': usage_serializer.data
        })

# Model Version ViewSet
class ModelVersionViewSet(viewsets.ModelViewSet):
    """
    API for managing ML model versions.
    Only accessible by approved AI Engineers and Admin.
    """
    permission_classes = [IsAuthenticated, IsApprovedUser, IsMachineLearningExpert]
    serializer_class = ModelVersionSerializer
    queryset = ModelVersion.objects.all()
    
    @action(detail=True, methods=['post'])
    def activate(self, request, pk=None):
        """Activate a specific model version and deactivate all others of the same name"""
        try:
            model_version = self.get_object()
            
            # Deactivate all other versions of this model
            ModelVersion.objects.filter(name=model_version.name).update(is_active=False)
            
            # Activate this version
            model_version.is_active = True
            model_version.save()
            
            return Response({
                "status": "success",
                "message": f"Model {model_version.name} v{model_version.version} is now active"
            })
        except Exception as e:
            return Response({
                "status": "error",
                "message": str(e)
            }, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=False, methods=['get'])
    def active(self, request):
        """Get all currently active model versions"""
        active_models = ModelVersion.objects.filter(is_active=True)
        serializer = self.get_serializer(active_models, many=True)
        return Response(serializer.data)

# Model Performance ViewSet
class ModelPerformanceViewSet(viewsets.ModelViewSet):
    """
    API for tracking and querying ML model performance metrics.
    Only accessible by approved AI Engineers and Admin.
    """
    permission_classes = [IsAuthenticated, IsApprovedUser, IsMachineLearningExpert]
    serializer_class = ModelPerformanceSerializer
    queryset = ModelPerformance.objects.all()
    
    @action(detail=False, methods=['get'])
    def latest(self, request):
        """Get the latest performance metrics for each model version"""
        # Group by model_version and get the latest for each
        latest_performances = {}
        
        for perf in ModelPerformance.objects.all().order_by('-recorded_at'):
            model_id = perf.model_version_id
            if model_id not in latest_performances:
                latest_performances[model_id] = perf
                
        serializer = self.get_serializer(latest_performances.values(), many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def by_model(self, request):
        """Get performance metrics for a specific model by name"""
        model_name = request.query_params.get('name')
        if not model_name:
            return Response({"error": "Model name parameter is required"}, 
                          status=status.HTTP_400_BAD_REQUEST)
            
        # Get all versions for this model name, then get performance metrics
        model_versions = ModelVersion.objects.filter(name=model_name)
        performances = ModelPerformance.objects.filter(
            model_version__in=model_versions
        ).order_by('-recorded_at')
        
        serializer = self.get_serializer(performances, many=True)
        return Response(serializer.data)

# Model Update ViewSet
class ModelUpdateViewSet(viewsets.ModelViewSet):
    """
    API for managing model updates.
    Only accessible by approved AI Engineers and Admin.
    """
    permission_classes = [IsAuthenticated, IsApprovedUser, IsMachineLearningExpert]
    serializer_class = ModelUpdateSerializer
    queryset = ModelUpdate.objects.all()
    
    def perform_create(self, serializer):
        # Set the current user as the updater
        serializer.save(updated_by=self.request.user)
        
    @action(detail=True, methods=['post'])
    def complete(self, request, pk=None):
        """Mark a model update as completed"""
        update = self.get_object()
        update.status = 'completed'
        update.completed_at = timezone.now()
        update.save()
        
        return Response({
            "status": "success",
            "message": "Model update marked as completed",
            "completed_at": update.completed_at
        })
    
    @action(detail=True, methods=['post'])
    def fail(self, request, pk=None):
        """Mark a model update as failed"""
        update = self.get_object()
        update.status = 'failed'
        update.completed_at = timezone.now()
        update.save()
        
        return Response({
            "status": "error",
            "message": "Model update marked as failed",
            "completed_at": update.completed_at
        })

# Usage Tracking ViewSet
class UsageTrackingViewSet(viewsets.ReadOnlyModelViewSet):
    """
    API for viewing ML service usage data.
    Only accessible by Admin.
    """
    permission_classes = [IsAuthenticated, IsAdminRole]
    serializer_class = UsageTrackingSerializer
    queryset = UsageTracking.objects.all()
    
    def get_queryset(self):
        queryset = UsageTracking.objects.all()
        
        # Filter by user if provided
        user_id = self.request.query_params.get('user_id')
        if user_id:
            queryset = queryset.filter(user_id=user_id)
            
        # Filter by endpoint if provided
        endpoint = self.request.query_params.get('endpoint')
        if endpoint:
            queryset = queryset.filter(endpoint=endpoint)
            
        # Filter by date range if provided
        start_date = self.request.query_params.get('start_date')
        end_date = self.request.query_params.get('end_date')
        if start_date and end_date:
            queryset = queryset.filter(
                timestamp__date__gte=start_date,
                timestamp__date__lte=end_date
            )
            
        return queryset
    
    @action(detail=False, methods=['get'])
    def summary(self, request):
        """Get a summary of usage statistics"""
        queryset = self.get_queryset()
        
        # Total API calls
        total_calls = queryset.count()
        
        # Calls by endpoint
        endpoint_counts = {}
        for record in queryset:
            endpoint = record.endpoint
            endpoint_counts[endpoint] = endpoint_counts.get(endpoint, 0) + 1
            
        # Calls by user (top 10 users)
        user_counts = {}
        for record in queryset:
            user_id = record.user_id
            username = record.user.username
            if username not in user_counts:
                user_counts[username] = 0
            user_counts[username] += 1
            
        top_users = sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Average response time
        avg_response_time = queryset.exclude(response_time__isnull=True).values_list('response_time', flat=True)
        avg_response_time = sum(avg_response_time) / len(avg_response_time) if avg_response_time else 0
        
        return Response({
            'total_calls': total_calls,
            'endpoint_counts': endpoint_counts,
            'top_users': dict(top_users),
            'avg_response_time': avg_response_time
        })
    
    @action(detail=False, methods=['get'])
    def my_usage(self, request):
        """Get the current user's usage data"""
        queryset = UsageTracking.objects.filter(user=request.user)
        serializer = self.get_serializer(queryset, many=True)
        
        # Get the user's quota information
        quota, created = UsageQuota.objects.get_or_create(
            user=request.user,
            defaults={'reset_date': (datetime.now() + timedelta(days=30)).date()}
        )
        
        # Get user's active subscription
        subscription = Subscription.objects.filter(
            user=request.user,
            is_active=True
        ).first()
        
        # If subscription exists, calculate remaining quota
        quota_data = {}
        if subscription:
            quota_data = {
                'api_calls_used': quota.api_calls_used,
                'api_calls_limit': subscription.max_api_calls,
                'api_calls_remaining': max(0, subscription.max_api_calls - quota.api_calls_used),
                'data_usage': float(quota.data_usage),
                'data_usage_limit': subscription.max_data_usage,
                'data_remaining': max(0, subscription.max_data_usage - float(quota.data_usage)),
                'reset_date': quota.reset_date
            }
        
        return Response({
            'usage_records': serializer.data,
            'quota': quota_data,
            'total_calls': queryset.count()
        })

# Subscription ViewSet
class SubscriptionViewSet(viewsets.ModelViewSet):
    """
    API for managing user subscriptions.
    Admin can manage all subscriptions, users can view their own.
    """
    serializer_class = SubscriptionSerializer
    
    def get_permissions(self):
        if self.action in ['list', 'create', 'update', 'destroy']:
            permission_classes = [IsAuthenticated, IsAdminRole]
        else:
            permission_classes = [IsAuthenticated]
        return [permission() for permission in permission_classes]
    
    def get_queryset(self):
        user = self.request.user
        
        # Admin can see all subscriptions
        if user.groups.filter(name='Admin').exists():
            return Subscription.objects.all()
        
        # Other users can only see their own subscriptions
        return Subscription.objects.filter(user=user)
    
    @action(detail=False, methods=['get'])
    def my_subscription(self, request):
        """Get the current user's active subscription"""
        subscription = Subscription.objects.filter(
            user=request.user,
            is_active=True
        ).first()
        
        if not subscription:
            return Response({"detail": "No active subscription found"}, status=status.HTTP_404_NOT_FOUND)
            
        serializer = self.get_serializer(subscription)
        return Response(serializer.data)
    
    @action(detail=False, methods=['post'])
    def subscribe(self, request):
        """Create a new subscription for the current user"""
        # Check if user already has an active subscription
        active_sub = Subscription.objects.filter(
            user=request.user,
            is_active=True
        ).first()
        
        if active_sub:
            # Deactivate the current subscription
            active_sub.is_active = False
            active_sub.save()
        
        # Get subscription details
        plan = request.data.get('plan', 'free')
        
        # Set price based on plan
        if plan == 'free':
            price = 0.00
            max_api_calls = 100
            max_data_usage = 1000
        elif plan == 'basic':
            price = 9.99
            max_api_calls = 1000
            max_data_usage = 10000
        elif plan == 'premium':
            price = 29.99
            max_api_calls = 10000
            max_data_usage = 100000
        else:  # enterprise
            price = 99.99
            max_api_calls = 100000
            max_data_usage = 1000000
        
        # Create new subscription
        subscription = Subscription.objects.create(
            user=request.user,
            plan=plan,
            start_date=datetime.now().date(),
            end_date=(datetime.now() + timedelta(days=30)).date(),
            is_active=True,
            auto_renew=request.data.get('auto_renew', True),
            price=price,
            max_api_calls=max_api_calls,
            max_data_usage=max_data_usage
        )
        
        # Create invoice for the subscription
        invoice = Invoice.objects.create(
            subscription=subscription,
            user=request.user,
            amount=price,
            due_date=(datetime.now() + timedelta(days=15)).date(),
            status='pending',
            description=f"{plan.capitalize()} plan subscription"
        )
        
        # Reset the user's quota
        quota, created = UsageQuota.objects.get_or_create(
            user=request.user,
            defaults={'reset_date': (datetime.now() + timedelta(days=30)).date()}
        )
        quota.api_calls_used = 0
        quota.data_usage = 0
        quota.reset_date = (datetime.now() + timedelta(days=30)).date()
        quota.save()
        
        serializer = self.get_serializer(subscription)
        return Response({
            'subscription': serializer.data,
            'invoice_id': invoice.id,
            'invoice_amount': invoice.amount,
            'invoice_due_date': invoice.due_date
        })

# Invoice ViewSet
class InvoiceViewSet(viewsets.ModelViewSet):
    """
    API for managing invoices.
    Admin can manage all invoices, users can view their own.
    """
    serializer_class = InvoiceSerializer
    
    def get_permissions(self):
        if self.action in ['list', 'create', 'destroy']:
            permission_classes = [IsAuthenticated, IsAdminRole]
        else:
            permission_classes = [IsAuthenticated]
        return [permission() for permission in permission_classes]
    
    def get_queryset(self):
        user = self.request.user
        
        # Admin can see all invoices
        if user.groups.filter(name='Admin').exists():
            return Invoice.objects.all()
        
        # Other users can only see their own invoices
        return Invoice.objects.filter(user=user)
    
    @action(detail=True, methods=['post'])
    def pay(self, request, pk=None):
        """Mark an invoice as paid"""
        invoice = self.get_object()
        
        # Only admin or the invoice owner can pay
        if not (request.user.groups.filter(name='Admin').exists() or invoice.user == request.user):
            return Response(
                {"detail": "You do not have permission to pay this invoice."}, 
                status=status.HTTP_403_FORBIDDEN
            )
        
        invoice.status = 'paid'
        invoice.payment_date = datetime.now().date()
        invoice.save()
        
        return Response({
            "status": "success",
            "message": f"Invoice {invoice.id} has been marked as paid",
            "payment_date": invoice.payment_date
        })
    
    @action(detail=False, methods=['get'])
    def my_invoices(self, request):
        """Get the current user's invoices"""
        invoices = Invoice.objects.filter(user=request.user)
        serializer = self.get_serializer(invoices, many=True)
        
        # Calculate totals
        pending_total = sum(inv.amount for inv in invoices if inv.status == 'pending')
        paid_total = sum(inv.amount for inv in invoices if inv.status == 'paid')
        overdue_total = sum(inv.amount for inv in invoices if inv.status == 'overdue')
        
        return Response({
            'invoices': serializer.data,
            'summary': {
                'pending_total': pending_total,
                'paid_total': paid_total,
                'overdue_total': overdue_total,
                'total_invoices': invoices.count()
            }
        })

# Usage Quota ViewSet
class UsageQuotaViewSet(viewsets.ReadOnlyModelViewSet):
    """
    API for viewing usage quotas.
    Admin can view all quotas, users can view their own.
    """
    serializer_class = UsageQuotaSerializer
    
    def get_permissions(self):
        if self.action == 'list':
            permission_classes = [IsAuthenticated, IsAdminRole]
        else:
            permission_classes = [IsAuthenticated]
        return [permission() for permission in permission_classes]
    
    def get_queryset(self):
        user = self.request.user
        
        # Admin can see all quotas
        if user.groups.filter(name='Admin').exists():
            return UsageQuota.objects.all()
        
        # Other users can only see their own quota
        return UsageQuota.objects.filter(user=user)
    
    @action(detail=False, methods=['get'])
    def my_quota(self, request):
        """Get the current user's usage quota"""
        quota, created = UsageQuota.objects.get_or_create(
            user=request.user,
            defaults={'reset_date': (datetime.now() + timedelta(days=30)).date()}
        )
        
        serializer = self.get_serializer(quota)
        return Response(serializer.data)
    
    @action(detail=True, methods=['post'])
    def reset(self, request, pk=None):
        """Reset a user's quota (admin only)"""
        if not request.user.groups.filter(name='Admin').exists():
            return Response(
                {"detail": "Only admins can reset quotas."}, 
                status=status.HTTP_403_FORBIDDEN
            )
        
        quota = self.get_object()
        quota.api_calls_used = 0
        quota.data_usage = 0
        quota.reset_date = (datetime.now() + timedelta(days=30)).date()
        quota.save()
        
        serializer = self.get_serializer(quota)
        return Response({
            'message': f"Quota reset for {quota.user.username}",
            'quota': serializer.data
        })




