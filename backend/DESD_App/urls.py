from django.contrib import admin
from django.urls import path, include
from rest_framework.routers import DefaultRouter, Route, DynamicRoute
from .viewsets import *



router = DefaultRouter()
router.register(r'users', UserViewSet, basename='user')

urlpatterns = router.urls

"""
6 CRUD endpoints for user

GET /users/ --> Returns a list of users

GET /users/1/ --> Returns a single user with id 1

POST /users/ --> Creates a new user

PUT /users/1/ --> Updates the user with id 1 [Updates all user information]

PATCH /users/1/ --> Updates the user with id 1 [Updates specific user information]

DELETE /users/1/ --> Deletes the user with id 1


to check more about all url in this app write python manage.py show_urls

"""


