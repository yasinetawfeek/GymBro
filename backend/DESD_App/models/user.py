"""
User-related models for the GymBro application.
"""
from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver


class UserProfile(models.Model):
    """Extended user profile with fitness and personal information."""
    
    TITLE_CHOICES = [
        ('Mr', 'Mr'),
        ('Mrs', 'Mrs'),
        ('Miss', 'Miss'),
        ('Ms', 'Ms'),
        ('Dr', 'Dr'),
        ('Prof', 'Professor'),
    ]
    
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    is_approved = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Personal Information
    title = models.CharField(max_length=5, choices=TITLE_CHOICES, blank=True, null=True)
    forename = models.CharField(max_length=50, blank=True, null=True)
    surname = models.CharField(max_length=50, blank=True, null=True)
    location = models.CharField(max_length=100, blank=True, null=True)
    phone_number = models.CharField(max_length=20, blank=True, null=True)
    
    # Fitness Profile
    height = models.CharField(max_length=50, blank=True, null=True)
    weight = models.CharField(max_length=50, blank=True, null=True)
    body_fat = models.CharField(max_length=50, blank=True, null=True)
    fitness_level = models.CharField(max_length=50, blank=True, null=True)
    
    # Preferences
    primary_goal = models.CharField(max_length=100, blank=True, null=True)
    workout_frequency = models.CharField(max_length=50, blank=True, null=True)
    preferred_time = models.CharField(max_length=50, blank=True, null=True)
    focus_areas = models.CharField(max_length=100, blank=True, null=True)
    
    # Achievements
    workouts_completed = models.CharField(max_length=20, blank=True, null=True)
    days_streak = models.CharField(max_length=20, blank=True, null=True)
    personal_bests = models.CharField(max_length=100, blank=True, null=True)
    points = models.CharField(max_length=20, blank=True, null=True)

    def __str__(self):
        return f'{self.user.username} Profile'

    class Meta:
        verbose_name = "User Profile"
        verbose_name_plural = "User Profiles"


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    """Create a user profile when a user is created."""
    if created:
        UserProfile.objects.create(user=instance)


@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    """Save the profile when the user is saved."""
    try:
        instance.profile.save()
    except UserProfile.DoesNotExist:
        UserProfile.objects.create(user=instance)