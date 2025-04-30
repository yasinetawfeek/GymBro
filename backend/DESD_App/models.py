from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.utils import timezone

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    is_approved = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f'{self.user.username} Profile'

# Signal to create a user profile when a user is created
@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)

# Signal to save the profile when the user is saved
@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    try:
        instance.profile.save()
    except UserProfile.DoesNotExist:
        UserProfile.objects.create(user=instance)

# Billing model to track user billing information
class BillingRecord(models.Model):
    SUBSCRIPTION_CHOICES = [
        ('free', 'Free'),
        ('basic', 'Basic'),
        ('premium', 'Premium'),
        ('enterprise', 'Enterprise'),
    ]
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('paid', 'Paid'),
        ('overdue', 'Overdue'),
        ('cancelled', 'Cancelled'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='billing_records')
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    subscription_type = models.CharField(max_length=20, choices=SUBSCRIPTION_CHOICES, default='free')
    billing_date = models.DateField()
    due_date = models.DateField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    description = models.TextField(blank=True, null=True)
    api_calls = models.IntegerField(default=0)
    data_usage = models.DecimalField(max_digits=10, decimal_places=2, default=0)  # in MB
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f'{self.user.username} - {self.amount} - {self.billing_date}'
    
    class Meta:
        ordering = ['-billing_date']

# AI Engineer Request model for approval workflow
class AIEngineerRequest(models.Model):
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('APPROVED', 'Approved'),
        ('REJECTED', 'Rejected')
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    request_date = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='PENDING')
    experience = models.TextField(help_text="Applicant's AI/ML experience", blank=True, null=True)
    qualifications = models.TextField(help_text="Relevant qualifications", blank=True, null=True)
    approved_by = models.ForeignKey(
        User, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True,
        related_name='approved_requests'
    )
    reviewed_date = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.user.username}'s AI Engineer Request ({self.status})"
    
    class Meta:
        ordering = ['-request_date']

# Model for ML model versioning - Feature 5
class ModelVersion(models.Model):
    name = models.CharField(max_length=100)
    version = models.CharField(max_length=20)
    description = models.TextField(blank=True, null=True)
    file_path = models.CharField(max_length=255)
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='created_models')
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=False)
    model_type = models.CharField(max_length=50, help_text="Type of ML model, e.g. 'classification', 'regression'")
    hyperparameters = models.JSONField(default=dict, help_text="Model hyperparameters as JSON")
    
    def __str__(self):
        return f"{self.name} v{self.version}"
    
    class Meta:
        ordering = ['-created_at']
        unique_together = ['name', 'version']

# Model for tracking model performance metrics - Feature 5
class ModelPerformance(models.Model):
    model_version = models.ForeignKey(ModelVersion, on_delete=models.CASCADE, related_name='performance_metrics')
    accuracy = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    loss = models.FloatField(null=True, blank=True)
    confusion_matrix = models.JSONField(null=True, blank=True)
    recorded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Performance for {self.model_version} at {self.recorded_at}"
    
    class Meta:
        ordering = ['-recorded_at']

# Model for tracking model updates - Feature 5
class ModelUpdate(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('completed', 'Completed'),
        ('failed', 'Failed')
    ]
    
    model_version = models.ForeignKey(ModelVersion, on_delete=models.CASCADE, related_name='updates')
    updated_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    description = models.TextField()
    hyperparameters = models.JSONField(default=dict)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return f"Update to {self.model_version} by {self.updated_by}"
    
    class Meta:
        ordering = ['-started_at']

# Model for tracking API usage - Feature 4 & 6
class UsageTracking(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='usage_records')
    endpoint = models.CharField(max_length=255, help_text="API endpoint that was used")
    timestamp = models.DateTimeField(auto_now_add=True)
    response_time = models.FloatField(help_text="Time taken for the request in ms", null=True, blank=True)
    status_code = models.IntegerField()
    request_data = models.JSONField(null=True, blank=True)
    response_data = models.JSONField(null=True, blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.endpoint} - {self.timestamp}"
    
    class Meta:
        ordering = ['-timestamp']

# Subscription model for Feature 6
class Subscription(models.Model):
    PLAN_CHOICES = [
        ('free', 'Free'),
        ('basic', 'Basic'),
        ('premium', 'Premium'),
        ('enterprise', 'Enterprise')
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='subscriptions')
    plan = models.CharField(max_length=20, choices=PLAN_CHOICES, default='free')
    start_date = models.DateField(default=timezone.now)
    end_date = models.DateField()
    is_active = models.BooleanField(default=True)
    auto_renew = models.BooleanField(default=True)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    max_api_calls = models.IntegerField(default=100, help_text="Maximum allowed API calls per month")
    max_data_usage = models.IntegerField(default=100, help_text="Maximum allowed data usage in MB per month")
    
    def __str__(self):
        return f"{self.user.username}'s {self.plan} subscription"
    
    class Meta:
        ordering = ['-start_date']

# Invoice model for Feature 6
class Invoice(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('paid', 'Paid'),
        ('overdue', 'Overdue'),
        ('cancelled', 'Cancelled'),
    ]
    
    subscription = models.ForeignKey(Subscription, on_delete=models.SET_NULL, null=True, related_name='invoices')
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='invoices')
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    invoice_date = models.DateField(auto_now_add=True)
    due_date = models.DateField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    payment_date = models.DateField(null=True, blank=True)
    description = models.TextField(blank=True, null=True)
    
    def __str__(self):
        return f"{self.user.username}'s invoice - {self.amount}"
    
    class Meta:
        ordering = ['-invoice_date']

# Usage Quota model for Feature 6
class UsageQuota(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='quota')
    api_calls_used = models.IntegerField(default=0)
    data_usage = models.DecimalField(max_digits=10, decimal_places=2, default=0)  # in MB
    reset_date = models.DateField(help_text="Date when the quota resets")
    
    def __str__(self):
        return f"{self.user.username}'s usage quota"
