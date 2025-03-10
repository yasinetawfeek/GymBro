from rest_framework.generics import CreateAPIView
from django.contrib.auth.models import User
from rest_framework import viewsets
from .serializers import *
from .models import User
from rest_framework.permissions import IsAdminUser,IsAuthenticated
from .permissions import IsOwner
from rest_framework import generics


"""
No need to put IsAuthenticated inside each permission class, since it is already in settings [on project-level by default]
meaning that you cannot access any viewset here without being authenticated first

"""

class UserViewSet(generics.RetrieveUpdateAPIView):
    """
    This is for user to change their personal account information or to view it only
    however they cannot access other users personal information
    """
    permission_classes = [IsAuthenticated, IsOwner] #only the owner of their account can access this view
    serializer_class = UserCreateSerializer
    #the reason why we do not set queryset to all objects, to prevent user from accessing other users' information

    def get_object(self):
        return self.request.user  # Only user can access their information even if they pass a different id


class AccountManagementViewSet(viewsets.ModelViewSet):
    """
    This is for admin to view and manage all users.
    Admins can view, update, delete, create users.
    """
    permission_classes = [IsAdminUser] #only the admin can access this view
    queryset = User.objects.all()
    serializer_class = UserCreateSerializer
    http_method_names = ['get', 'put', 'patch','delete'] #explicitly not mentioning POST ,as it does not make sense for admin to create user

