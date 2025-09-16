"""
Analytics and tracking models for the GymBro application.
"""
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import uuid


class UsageRecord(models.Model):
    """Tracks user session data for AI workout analysis and billing purposes."""
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='usage_records')
    session_id = models.UUIDField(default=uuid.uuid4, unique=True)
    session_start = models.DateTimeField(default=timezone.now)
    session_end = models.DateTimeField(null=True, blank=True)
    total_duration = models.IntegerField(help_text="Duration in seconds", null=True, blank=True)
    
    # Usage metrics
    frames_processed = models.IntegerField(default=0)
    corrections_sent = models.IntegerField(default=0)
    workout_type = models.IntegerField(default=0)
    
    # Billing information
    billable_amount = models.DecimalField(max_digits=10, decimal_places=2, default=0.0)
    subscription_plan = models.CharField(max_length=50, blank=True)
    
    # Status flag
    is_active = models.BooleanField(default=True)
    
    # Device info
    client_ip = models.GenericIPAddressField(null=True, blank=True)
    platform = models.CharField(max_length=50, null=True, blank=True)
    
    class Meta:
        ordering = ['-session_start']
        verbose_name = "Usage Record"
        verbose_name_plural = "Usage Records"
    
    def end_session(self):
        """End the current session and calculate duration."""
        if self.is_active:
            self.session_end = timezone.now()
            self.total_duration = (self.session_end - self.session_start).total_seconds()
            self.is_active = False
            self.save()
    
    def calculate_billable_amount(self, rate=0.001):
        """Calculate billable amount based on usage metrics and rate."""
        # Example calculation: $0.001 per correction
        self.billable_amount = self.corrections_sent * rate
        self.save()


class ModelPerformanceMetric(models.Model):
    """Tracks AI model performance metrics for admin analytics."""
    
    # Timestamps and identification
    timestamp = models.DateTimeField(default=timezone.now)
    model_version = models.CharField(max_length=50)
    workout_type = models.IntegerField(default=0)
    
    # Accuracy metrics
    avg_prediction_confidence = models.FloatField()
    min_prediction_confidence = models.FloatField()
    max_prediction_confidence = models.FloatField()
    correction_magnitude_avg = models.FloatField()
    stable_prediction_rate = models.FloatField(help_text="% of predictions that remained stable")
    
    # Performance metrics
    avg_response_latency = models.IntegerField(help_text="Average latency in milliseconds")
    processing_time_per_frame = models.IntegerField(help_text="Processing time in milliseconds")
    time_to_first_correction = models.IntegerField(help_text="Time in milliseconds until first correction")
    frame_processing_rate = models.FloatField(help_text="Frames processed per second")
    
    # System metrics
    cpu_usage = models.FloatField(null=True, blank=True)
    memory_usage = models.FloatField(null=True, blank=True)
    
    class Meta:
        ordering = ['-timestamp']
        verbose_name = "Model Performance Metric"
        verbose_name_plural = "Model Performance Metrics"
        
    def __str__(self):
        return f"Performance metrics for {self.model_version} at {self.timestamp}"


class UserLastViewedExercise(models.Model):
    """Tracks the last viewed exercise/workout for each user."""
    
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='last_viewed_exercise')
    workout_type = models.IntegerField(default=0)
    workout_name = models.CharField(max_length=100, blank=True, null=True)
    last_viewed_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-last_viewed_at']
        verbose_name = "User Last Viewed Exercise"
        verbose_name_plural = "User Last Viewed Exercises"
        
    def __str__(self):
        return f"{self.user.username}'s last viewed exercise: {self.workout_name}"