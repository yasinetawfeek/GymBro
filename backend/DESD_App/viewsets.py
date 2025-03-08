from rest_framework.generics import CreateAPIView
from django.contrib.auth.models import User
from rest_framework import viewsets
from .serializers import *
from .models import User


class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserCreateSerializer