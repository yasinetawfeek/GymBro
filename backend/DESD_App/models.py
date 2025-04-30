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
