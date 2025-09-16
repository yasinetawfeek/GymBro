"""
Analytics and tracking viewsets for the GymBro application.
"""
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from rest_framework.decorators import action
from django.db.models import Sum, Count, Avg
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from ..models import UsageRecord, ModelPerformanceMetric, UserLastViewedExercise
from ..serializers import UsageRecordSerializer, ModelPerformanceMetricSerializer, UserLastViewedExerciseSerializer
from ..permissions import IsAIEngineerRole, IsApprovedAIEngineer


class UsageTrackingViewSet(viewsets.ModelViewSet):
    """ViewSet for usage tracking and session management."""
    
    queryset = UsageRecord.objects.all()
    serializer_class = UsageRecordSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        """Return usage records for the current user or all if admin/ai engineer."""
        if self.request.user.is_staff or self.request.user.groups.filter(name='AI Engineers').exists():
            return UsageRecord.objects.all()
        return UsageRecord.objects.filter(user=self.request.user)
    
    @swagger_auto_schema(
        operation_description="Start a new usage tracking session",
        operation_summary="Start session",
        responses={
            201: openapi.Response(description="Session started successfully"),
            400: openapi.Response(description="Invalid data provided")
        }
    )
    @action(detail=False, methods=['post'])
    def start_session(self, request):
        """Start a new usage tracking session."""
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            serializer.save(user=request.user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    @swagger_auto_schema(
        operation_description="End a usage tracking session",
        operation_summary="End session",
        responses={200: openapi.Response(description="Session ended successfully")}
    )
    @action(detail=True, methods=['patch'])
    def end_session(self, request, pk=None):
        """End a usage tracking session."""
        usage_record = self.get_object()
        usage_record.end_session()
        serializer = self.get_serializer(usage_record)
        return Response(serializer.data)
    
    @swagger_auto_schema(
        operation_description="Update session metrics",
        operation_summary="Update metrics",
        responses={200: openapi.Response(description="Metrics updated successfully")}
    )
    @action(detail=True, methods=['patch'])
    def update_metrics(self, request, pk=None):
        """Update session metrics."""
        usage_record = self.get_object()
        frames_processed = request.data.get('frames_processed', 0)
        corrections_sent = request.data.get('corrections_sent', 0)
        
        usage_record.frames_processed += frames_processed
        usage_record.corrections_sent += corrections_sent
        usage_record.save()
        
        serializer = self.get_serializer(usage_record)
        return Response(serializer.data)


class ModelPerformanceViewSet(viewsets.ModelViewSet):
    """ViewSet for model performance metrics (AI Engineers only)."""
    
    queryset = ModelPerformanceMetric.objects.all()
    serializer_class = ModelPerformanceMetricSerializer
    permission_classes = [IsAuthenticated, IsApprovedAIEngineer]
    
    @swagger_auto_schema(
        operation_description="Record model performance metrics",
        operation_summary="Record metrics",
        responses={
            201: openapi.Response(description="Metrics recorded successfully"),
            400: openapi.Response(description="Invalid data provided")
        }
    )
    def create(self, request):
        """Record model performance metrics."""
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    @swagger_auto_schema(
        operation_description="Get performance analytics",
        operation_summary="Get analytics",
        responses={200: openapi.Response(
            description="Performance analytics",
            examples={
                "application/json": {
                    "avg_confidence": 0.85,
                    "avg_response_time": 45,
                    "total_sessions": 150,
                    "performance_trend": "improving"
                }
            }
        )}
    )
    @action(detail=False, methods=['get'])
    def analytics(self, request):
        """Get performance analytics."""
        metrics = self.get_queryset()
        
        avg_confidence = metrics.aggregate(Avg('avg_prediction_confidence'))['avg_prediction_confidence__avg'] or 0
        avg_response_time = metrics.aggregate(Avg('avg_response_latency'))['avg_response_latency__avg'] or 0
        total_sessions = metrics.count()
        
        return Response({
            'avg_confidence': avg_confidence,
            'avg_response_time': avg_response_time,
            'total_sessions': total_sessions,
            'performance_trend': 'improving'  # This could be calculated based on historical data
        })


class UserLastViewedExerciseViewSet(viewsets.ModelViewSet):
    """ViewSet for tracking user's last viewed exercise."""
    
    queryset = UserLastViewedExercise.objects.all()
    serializer_class = UserLastViewedExerciseSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        """Return last viewed exercise for the current user."""
        return UserLastViewedExercise.objects.filter(user=self.request.user)
    
    @swagger_auto_schema(
        operation_description="Update last viewed exercise",
        operation_summary="Update last viewed",
        responses={
            200: openapi.Response(description="Last viewed exercise updated"),
            201: openapi.Response(description="Last viewed exercise created")
        }
    )
    def create(self, request):
        """Create or update last viewed exercise."""
        exercise_data = request.data.copy()
        exercise_data['user'] = request.user.id
        
        # Try to get existing record
        try:
            existing = UserLastViewedExercise.objects.get(user=request.user)
            serializer = self.get_serializer(existing, data=exercise_data, partial=True)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data)
        except UserLastViewedExercise.DoesNotExist:
            pass
        
        # Create new record
        serializer = self.get_serializer(data=exercise_data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)