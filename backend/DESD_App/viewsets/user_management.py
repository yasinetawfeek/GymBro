"""
User management viewsets for the GymBro application.
"""
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, IsAdminUser, AllowAny
from django.contrib.auth.models import User
from rest_framework.decorators import action
from rest_framework.generics import CreateAPIView
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from ..models import UserProfile
from ..serializers import UserSerializer, UserProfileSerializer, UserCreateSerializer
from ..permissions import IsOwner, IsApprovedUser


class UserViewSet(viewsets.ModelViewSet):
    """ViewSet for user account management."""
    
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [IsAuthenticated, IsOwner]
    
    def get_queryset(self):
        """Return only the current user's data."""
        return User.objects.filter(id=self.request.user.id)
    
    @swagger_auto_schema(
        operation_description="Get current user information",
        operation_summary="Get my account",
        responses={200: openapi.Response(
            description="Current user information",
            examples={
                "application/json": {
                    "id": 1,
                    "username": "john_doe",
                    "email": "john@example.com",
                    "first_name": "John",
                    "last_name": "Doe",
                    "profile": {
                        "is_approved": True,
                        "fitness_level": "intermediate"
                    }
                }
            }
        )}
    )
    def list(self, request):
        """Get current user information."""
        serializer = self.get_serializer(request.user)
        return Response(serializer.data)
    
    @swagger_auto_schema(
        operation_description="Update current user information",
        operation_summary="Update my account",
        responses={
            200: openapi.Response(description="User updated successfully"),
            400: openapi.Response(description="Invalid data provided")
        }
    )
    def update(self, request, *args, **kwargs):
        """Update current user information."""
        instance = request.user
        serializer = self.get_serializer(instance, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class AccountManagementViewSet(viewsets.ModelViewSet):
    """ViewSet for admin account management."""
    
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [IsAdminUser]
    
    @swagger_auto_schema(
        operation_description="Get all users (admin only)",
        operation_summary="List all users",
        responses={200: openapi.Response(description="List of all users")}
    )
    def list(self, request):
        """Get all users."""
        users = User.objects.all()
        serializer = self.get_serializer(users, many=True)
        return Response(serializer.data)
    
    @swagger_auto_schema(
        operation_description="Create a new user (admin only)",
        operation_summary="Create user",
        responses={
            201: openapi.Response(description="User created successfully"),
            400: openapi.Response(description="Invalid data provided")
        }
    )
    def create(self, request):
        """Create a new user."""
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    @swagger_auto_schema(
        operation_description="Update user information (admin only)",
        operation_summary="Update user",
        responses={
            200: openapi.Response(description="User updated successfully"),
            400: openapi.Response(description="Invalid data provided")
        }
    )
    def update(self, request, *args, **kwargs):
        """Update user information."""
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    @swagger_auto_schema(
        operation_description="Delete a user (admin only)",
        operation_summary="Delete user",
        responses={
            204: openapi.Response(description="User deleted successfully"),
            404: openapi.Response(description="User not found")
        }
    )
    def destroy(self, request, *args, **kwargs):
        """Delete a user."""
        instance = self.get_object()
        instance.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class ApprovalViewSet(viewsets.ModelViewSet):
    """ViewSet for user approval management."""
    
    queryset = UserProfile.objects.all()
    serializer_class = UserProfileSerializer
    permission_classes = [IsAdminUser]
    
    @swagger_auto_schema(
        operation_description="Approve or reject user accounts",
        operation_summary="Manage user approvals",
        responses={
            200: openapi.Response(description="Approval status updated"),
            400: openapi.Response(description="Invalid data provided")
        }
    )
    @action(detail=True, methods=['patch'])
    def approve(self, request, pk=None):
        """Approve a user account."""
        profile = self.get_object()
        profile.is_approved = True
        profile.save()
        return Response({'status': 'User approved successfully'})
    
    @swagger_auto_schema(
        operation_description="Reject a user account",
        operation_summary="Reject user",
        responses={200: openapi.Response(description="User rejected successfully")}
    )
    @action(detail=True, methods=['patch'])
    def reject(self, request, pk=None):
        """Reject a user account."""
        profile = self.get_object()
        profile.is_approved = False
        profile.save()
        return Response({'status': 'User rejected successfully'})


class RoleInfoViewSet(viewsets.ViewSet):
    """ViewSet for role information."""
    
    permission_classes = [IsAuthenticated]
    
    @swagger_auto_schema(
        operation_description="Get current user's role information",
        operation_summary="Get role info",
        responses={200: openapi.Response(
            description="User role information",
            examples={
                "application/json": {
                    "is_admin": False,
                    "is_ai_engineer": True,
                    "is_approved": True,
                    "groups": ["AI Engineers"]
                }
            }
        )}
    )
    def list(self, request):
        """Get current user's role information."""
        user = request.user
        user_groups = user.groups.all()
        
        return Response({
            'is_admin': user.is_staff,
            'is_ai_engineer': user.groups.filter(name='AI Engineers').exists(),
            'is_approved': getattr(user.profile, 'is_approved', False),
            'groups': [group.name for group in user_groups]
        })


class RegisterView(CreateAPIView):
    """View for user registration."""
    
    queryset = User.objects.all()
    serializer_class = UserCreateSerializer
    permission_classes = [AllowAny]
    
    @swagger_auto_schema(
        operation_description="Register a new user",
        operation_summary="Register user",
        responses={
            201: openapi.Response(description="User created successfully"),
            400: openapi.Response(description="Invalid data provided")
        }
    )
    def post(self, request, *args, **kwargs):
        """Register a new user."""
        return super().post(request, *args, **kwargs)