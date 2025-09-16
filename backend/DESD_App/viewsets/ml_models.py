"""
Machine Learning model management viewsets for the GymBro application.
"""
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from rest_framework.decorators import action
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from ..models import MLModel
from ..serializers import MLModelSerializer
from ..permissions import IsAIEngineerRole, IsApprovedAIEngineer


class MLModelViewSet(viewsets.ModelViewSet):
    """ViewSet for ML model management (AI Engineers only)."""
    
    queryset = MLModel.objects.all()
    serializer_class = MLModelSerializer
    permission_classes = [IsAuthenticated, IsApprovedAIEngineer]
    
    @swagger_auto_schema(
        operation_description="Get all ML models",
        operation_summary="List ML models",
        responses={200: openapi.Response(description="List of ML models")}
    )
    def list(self, request):
        """Get all ML models."""
        models = self.get_queryset()
        serializer = self.get_serializer(models, many=True)
        return Response(serializer.data)
    
    @swagger_auto_schema(
        operation_description="Create a new ML model",
        operation_summary="Create ML model",
        responses={
            201: openapi.Response(description="ML model created successfully"),
            400: openapi.Response(description="Invalid data provided")
        }
    )
    def create(self, request):
        """Create a new ML model."""
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    @swagger_auto_schema(
        operation_description="Deploy an ML model",
        operation_summary="Deploy model",
        responses={200: openapi.Response(description="Model deployed successfully")}
    )
    @action(detail=True, methods=['patch'])
    def deploy(self, request, pk=None):
        """Deploy an ML model."""
        model = self.get_object()
        model.deployed = True
        model.save()
        return Response({'status': 'Model deployed successfully'})
    
    @swagger_auto_schema(
        operation_description="Undeploy an ML model",
        operation_summary="Undeploy model",
        responses={200: openapi.Response(description="Model undeployed successfully")}
    )
    @action(detail=True, methods=['patch'])
    def undeploy(self, request, pk=None):
        """Undeploy an ML model."""
        model = self.get_object()
        model.deployed = False
        model.save()
        return Response({'status': 'Model undeployed successfully'})
    
    @swagger_auto_schema(
        operation_description="Get deployed models",
        operation_summary="Get deployed models",
        responses={200: openapi.Response(description="List of deployed models")}
    )
    @action(detail=False, methods=['get'])
    def deployed(self, request):
        """Get all deployed models."""
        deployed_models = self.get_queryset().filter(deployed=True)
        serializer = self.get_serializer(deployed_models, many=True)
        return Response(serializer.data)


class TrainWorkoutClassiferViewSet(viewsets.ViewSet):
    """ViewSet for training workout classifier models."""
    
    permission_classes = [IsAuthenticated, IsApprovedAIEngineer]
    
    @swagger_auto_schema(
        operation_description="Train a workout classifier model",
        operation_summary="Train model",
        responses={
            200: openapi.Response(description="Training started successfully"),
            400: openapi.Response(description="Invalid training parameters")
        }
    )
    def post(self, request):
        """Train a workout classifier model."""
        # This would integrate with the AI service for model training
        # For now, return a placeholder response
        return Response({
            'status': 'Training started',
            'model_id': 'placeholder_model_id',
            'estimated_time': '30 minutes'
        })


class PredictWorkoutClassiferViewSet(viewsets.ViewSet):
    """ViewSet for workout classification predictions."""
    
    permission_classes = [IsAuthenticated]
    
    @swagger_auto_schema(
        operation_description="Predict workout type from pose data",
        operation_summary="Predict workout",
        responses={
            200: openapi.Response(description="Workout prediction successful"),
            400: openapi.Response(description="Invalid pose data")
        }
    )
    def post(self, request):
        """Predict workout type from pose data."""
        # This would integrate with the AI service for predictions
        # For now, return a placeholder response
        return Response({
            'predicted_workout': 'plank',
            'confidence': 0.85,
            'workout_type': 12
        })