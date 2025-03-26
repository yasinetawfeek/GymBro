from rest_framework.generics import CreateAPIView
from django.contrib.auth.models import User
from rest_framework import viewsets
from .serializers import *
from .models import User
from rest_framework.permissions import IsAdminUser,IsAuthenticated
from .permissions import IsOwner,IsMachineLearningExpert
from rest_framework import generics
from rest_framework.response import Response
from rest_framework.exceptions import PermissionDenied
import requests


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
    permission_classes = [IsAuthenticated,IsAdminUser] #only the admin can access this view
    queryset = User.objects.all()
    serializer_class = UserCreateSerializer

    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()

        target_rolename = instance.groups.first().name
        
        current_rolename = request.user.groups.first().name
        if target_rolename == current_rolename:
            raise PermissionDenied("User cannot delete other users with the same role")
            
        return super().destroy(request, *args, **kwargs)

class UserActiveCountViewSet(viewsets.ViewSet):
    permission_classes = [IsAuthenticated,IsAdminUser]
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
FLASK_MACHINE_LEARNING_API_URL="http://10.167.143.148:8001/train_classify_workout"
FLASK_MACHINE_LEARNING_API_PREDICT_URL="http://10.167.143.148:8001/predict_classify_workout"

API_KEY = "job_hunting_ai_memory_leakage"

class TrainWorkoutClassiferViewSet(viewsets.ViewSet):
    """
    ViewSet only accessable by Admin and AI Engineer
    """
    
    permission_classes = [IsAuthenticated,IsMachineLearningExpert] 

    def post(self, request):
        user = request.user
        permissions = user.get_all_permissions()
        print("all permissions",permissions)
        
        headers ={
            "X-API-KEY":API_KEY, 
            "Content-Type": "application/json",
        }
        machine_learning_accuracy = requests.post(FLASK_MACHINE_LEARNING_API_URL, json=request.data, headers=headers)
        return Response(machine_learning_accuracy.json(), status=machine_learning_accuracy.status_code)
    

class PredictWorkoutClassiferViewSet(viewsets.ViewSet):
    """
    ViewSet is accessed by anyone who has registered and has been authenticated
    """
    permission_classes = [IsAuthenticated]
    def post(self, request):

        headers ={
            "X-API-KEY":API_KEY, 
            "Content-Type": "application/json",
        }
        machine_learning_prediction = requests.post(FLASK_MACHINE_LEARNING_API_PREDICT_URL, json=request.data, headers=headers)
        return Response(machine_learning_prediction.json(), status=machine_learning_prediction.status_code)
