"""
Machine Learning models and related data structures.
"""
from django.db import models


class MLModel(models.Model):
    """Machine Learning model metadata and configuration."""
    
    MODEL_TYPES = (
        ('workout', 'Workout Classification'),
        ('pose', 'Pose Correction'),
        ('muscle', 'Muscle Activation'),
    )
    
    name = models.CharField(max_length=255)
    model_type = models.CharField(max_length=32, choices=MODEL_TYPES)
    version = models.CharField(max_length=20, default="1.0.0")
    description = models.TextField(null=True, blank=True)
    learning_rate = models.FloatField(default=0.001)
    epochs = models.IntegerField(default=100)
    batch_size = models.IntegerField(default=32)
    accuracy = models.FloatField(default=0.0)
    deployed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_trained = models.DateField(null=True, blank=True)
    
    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=['model_type', 'deployed'],
                condition=models.Q(deployed=True),
                name='unique_deployed_model_per_type'
            )
        ]
        verbose_name = "ML Model"
        verbose_name_plural = "ML Models"
    
    def save(self, *args, **kwargs):
        """Override save to ensure only one model per type is deployed."""
        # If this model is being deployed, un-deploy other models of the same type
        if self.deployed:
            MLModel.objects.filter(
                model_type=self.model_type, 
                deployed=True
            ).exclude(pk=self.pk).update(deployed=False)
        super().save(*args, **kwargs)
    
    def __str__(self):
        return f"{self.name} ({self.get_model_type_display()})"