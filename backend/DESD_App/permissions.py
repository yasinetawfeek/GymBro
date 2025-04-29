from rest_framework import permissions

class IsOwner(permissions.BasePermission):
    """
    Permission check for object owner
    """
    def has_object_permission(self, request, view, obj):
        return obj.owner == request.user

class IsMachineLearningExpert(permissions.BasePermission):
    """
    permission check for machine learning experts
    """
    def has_permission(self, request, view):
        # check if user has permission to access a Machine Learning features
        return request.user.has_perm('auth.machine_learning_permission')
    
class IsApprovedUser(permissions.BasePermission):
    """
    Permission check for approved users
    """
    def has_permission(self, request, view):
        # Check if user is authenticated and approved
        return request.user.is_authenticated and hasattr(request.user, 'profile') and request.user.profile.is_approved

    
