from rest_framework import permissions

class IsOwner(permissions.BasePermission):
    """
    Object level permission
    Custom permission to only allow owners of an object to perform actions on it.
    """
    def has_object_permission(self, request, view, obj):
        # If the object is a User, check if it's the same as the requesting user
        return obj == request.user