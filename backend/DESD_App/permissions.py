from rest_framework import permissions
from django.contrib.auth.models import Group

class IsOwner(permissions.BasePermission):
    """
    Permission check for object owner
    """
    def has_object_permission(self, request, view, obj):
        return obj.owner == request.user

class IsMachineLearningExpert(permissions.BasePermission):
    """
    Permission check for machine learning experts
    """
    def has_permission(self, request, view):
        # check if user has permission to access Machine Learning features
        return request.user.has_perm('auth.machine_learning_permission')
    
class IsApprovedUser(permissions.BasePermission):
    """
    Permission check for approved users
    """
    def has_permission(self, request, view):
        # Check if user is authenticated and approved
        return request.user.is_authenticated and hasattr(request.user, 'profile') and request.user.profile.is_approved

class IsAdminRole(permissions.BasePermission):
    """
    Permission check for users with Admin role
    """
    def has_permission(self, request, view):
        # Check if user has Admin role
        if not request.user.is_authenticated:
            return False
        return request.user.groups.filter(name='Admin').exists()

class IsAIEngineerRole(permissions.BasePermission):
    """
    Permission check for users with AI Engineer role
    """
    def has_permission(self, request, view):
        # Check if user has AI Engineer role
        if not request.user.is_authenticated:
            return False
        return request.user.groups.filter(name='AI Engineer').exists()

class IsApprovedAIEngineer(permissions.BasePermission):
    """
    Permission check for approved AI Engineers
    """
    def has_permission(self, request, view):
        # Check if user is an approved AI Engineer
        if not request.user.is_authenticated:
            return False
        
        is_ai_engineer = request.user.groups.filter(name='AI Engineer').exists()
        is_approved = hasattr(request.user, 'profile') and request.user.profile.is_approved
        
        return is_ai_engineer and is_approved

    
