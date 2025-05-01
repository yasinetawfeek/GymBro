from django.utils.timesince import timesince
from rest_framework import serializers
from djoser.serializers import UserCreateSerializer as BaseUserCreateSerializer
from django.contrib.auth import get_user_model
from .models import *
from django.contrib.auth.models import Group
import re
from django.core.validators import EmailValidator
from django.core.exceptions import ValidationError

User = get_user_model()

# Company domain for employees
COMPANY_DOMAIN = "ufcfur_15_3.com"

# Simple email validator function that accepts underscores and numbers in domain
def custom_email_validator(value):
    """
    Very simple email validator that allows underscores and numbers in domain names
    """
    # Pattern that specifically allows underscores in domain names
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9._%-]+(\.[a-zA-Z0-9_%-]+)+$'
    if not re.match(pattern, value):
        raise ValidationError('Enter a valid email address.')
    return value

# Email validator function
def validate_email_domain(email, group_name):
    """
    Validates that AI Engineers and Admins use company email domain
    """
    if group_name in ['Admin', 'AI Engineer']:
        if not email.endswith(f'@{COMPANY_DOMAIN}'):
            raise serializers.ValidationError(f"Admin and AI Engineer accounts must use company email (@{COMPANY_DOMAIN}).")
    return email

class UserCreateSerializer(serializers.ModelSerializer):
    rolename = serializers.SerializerMethodField()  # This is read-only field
    group = serializers.CharField(write_only=True, required=False)  # this field is for assigning group to user [Write]
    groups = serializers.SerializerMethodField()  # Include groups in the response
    is_admin = serializers.SerializerMethodField()  # Add is_admin field
    is_approved = serializers.BooleanField(required=False, default=False, write_only=True)
    # Use our custom email validator
    email = serializers.CharField(validators=[custom_email_validator])  # Changed to CharField with our validator
    
    class Meta:
        model = User
        fields = ['id', 'username', 'password', 'email', 'rolename', 'group', 'groups', 'is_admin', 'is_approved']
        extra_kwargs = {'password': {'write_only': True}}

    def get_rolename(self, obj):
        # Get the first group name associated with the user (if any)
        return obj.groups.first().name if obj.groups.exists() else "No Role"
    
    def get_groups(self, obj):
        # Return all groups the user belongs to
        return [{'id': g.id, 'name': g.name} for g in obj.groups.all()]
    
    def get_is_admin(self, obj):
        # Check if user is in the Admin group
        return obj.groups.filter(name='Admin').exists()
    
    def validate(self, data):
        # Get the group name if provided
        group_name = data.get('group')
        email = data.get('email')
        
        if email and group_name:
            # Validate email domain for company employees
            validate_email_domain(email, group_name)
        
        return data
    
    def create(self, validated_data):
        group_name = validated_data.pop('group', None)
        is_approved = validated_data.pop('is_approved', False)
        user = User.objects.create_user(**validated_data)
        request = self.context.get('request')

        # Set default approval status based on role
        # Customers are auto-approved, company roles need admin approval
        if group_name:
            try:
                group = Group.objects.get(name=group_name)
                user.groups.add(group)
                
                # Set is_approved flag - Customers are automatically approved
                if group_name in ['Admin', 'AI Engineer']:
                    user.profile.is_approved = is_approved
                else:
                    user.profile.is_approved = True
                user.profile.save()
                
            except Group.DoesNotExist:
                raise serializers.ValidationError(f"Group '{group_name}' does not exist.")
        else:
            # Default to Customer role
            group = Group.objects.get(name='Customer')
            user.groups.add(group)
            user.profile.is_approved = True
            user.profile.save()

        return user
    

    def update(self, instance, validated_data):
        group_name = validated_data.pop('group', None)
        user_id = instance.id

        request = self.context.get('request')
        is_self_update = self.context.get('is_self_update', False)
        
        # If this is a self-update, bypass all role restriction checks
        if is_self_update:
            return super().update(instance, validated_data)
            
        user = User.objects.get(id=user_id)
        role_name = user.groups.first().name if user.groups.exists() else None
        
        # Skip the role check if user is updating their own profile
        if request and request.user.id == user_id:
            # Allow users to update their own profile
            pass
        # Only apply role restriction for admins updating other users
        elif role_name and request and role_name == request.user.groups.first().name:
            raise serializers.ValidationError("You cannot modify a user who has same role/group as you")
        
        if group_name is not None: 
            if request and not request.user.is_staff: 
                raise serializers.ValidationError("You do not have the permissions to change this users' role/group")
            elif request and group_name == request.user.groups.first().name:
                raise serializers.ValidationError("You cannot assign the user the same role as you")
            else:
                instance.groups.clear()
                group = Group.objects.get(name=group_name)
                instance.groups.add(group)

        return super().update(instance, validated_data)
    

# Serializer for BillingRecord model
class BillingRecordSerializer(serializers.ModelSerializer):
    username = serializers.SerializerMethodField()
    email = serializers.SerializerMethodField()
    
    class Meta:
        model = BillingRecord
        fields = [
            'id', 'username', 'email', 'amount', 'subscription_type', 
            'billing_date', 'due_date', 'status', 'description', 
            'api_calls', 'data_usage', 'created_at'
        ]
    
    def get_username(self, obj):
        return obj.user.username
    
    def get_email(self, obj):
        return obj.user.email

# Serializer for Subscription
class SubscriptionSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.username', read_only=True)
    days_remaining = serializers.SerializerMethodField()
    
    class Meta:
        model = Subscription
        fields = [
            'id', 'user', 'username', 'plan', 'start_date', 'end_date',
            'is_active', 'auto_renew', 'price', 'max_api_calls',
            'max_data_usage', 'days_remaining'
        ]
    
    def get_days_remaining(self, obj):
        today = timezone.now().date()
        if obj.end_date < today:
            return 0
        return (obj.end_date - today).days

# Serializer for Invoice
class InvoiceSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.username', read_only=True)
    subscription_plan = serializers.CharField(source='subscription.plan', read_only=True, allow_null=True)
    subscription_type = serializers.CharField(source='subscription.plan', read_only=True, allow_null=True)
    subscription_name = serializers.CharField(source='subscription.plan', read_only=True, allow_null=True)
    days_overdue = serializers.SerializerMethodField()
    subscription_start_date = serializers.DateField(source='subscription.start_date', read_only=True, allow_null=True)
    subscription_end_date = serializers.DateField(source='subscription.end_date', read_only=True, allow_null=True)
    
    class Meta:
        model = Invoice
        fields = [
            'id', 'subscription', 'subscription_plan', 'subscription_type', 'subscription_name',
            'user', 'username', 'amount', 'invoice_date', 'due_date', 'status', 'payment_date',
            'description', 'days_overdue', 'subscription_start_date', 'subscription_end_date'
        ]
    
    def get_days_overdue(self, obj):
        today = timezone.now().date()
        if obj.status != 'overdue' or obj.due_date >= today:
            return 0
        return (today - obj.due_date).days

class UsageRecordSerializer(serializers.ModelSerializer):
    """Serializer for usage tracking records"""
    username = serializers.SerializerMethodField()
    duration_formatted = serializers.SerializerMethodField()
    
    class Meta:
        model = UsageRecord
        fields = [
            'id', 'user', 'username', 'session_id', 'session_start', 'session_end', 
            'total_duration', 'duration_formatted', 'frames_processed', 'corrections_sent', 
            'workout_type', 'billable_amount', 'subscription_plan', 'is_active'
        ]
        read_only_fields = ['id', 'username', 'duration_formatted']
    
    def get_username(self, obj):
        return obj.user.username if obj.user else None
    
    def get_duration_formatted(self, obj):
        """Return human-readable duration"""
        if not obj.total_duration:
            return "Active" if obj.is_active else "Unknown"
        
        minutes, seconds = divmod(obj.total_duration, 60)
        hours, minutes = divmod(minutes, 60)
        
        if hours > 0:
            return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        elif minutes > 0:
            return f"{int(minutes)}m {int(seconds)}s"
        else:
            return f"{int(seconds)}s"

class ModelPerformanceMetricSerializer(serializers.ModelSerializer):
    """Serializer for model performance metrics"""
    workout_name = serializers.SerializerMethodField()
    latency_rating = serializers.SerializerMethodField()
    
    class Meta:
        model = ModelPerformanceMetric
        fields = '__all__'
    
    def get_workout_name(self, obj):
        """Return the workout name based on the workout type"""
        workout_map = {
            0: "Barbell Bicep Curl", 1: "Bench Press", 2: "Chest Fly Machine",
            3: "Deadlift", 4: "Decline Bench Press", 5: "Hammer Curl",
            6: "Hip Thrust", 7: "Incline Bench Press", 8: "Lat Pulldown",
            9: "Lateral Raises", 10: "Leg Extensions", 11: "Leg Raises",
            12: "Plank", 13: "Pull Up", 14: "Push Ups", 15: "Romanian Deadlift",
            16: "Russian Twist", 17: "Shoulder Press", 18: "Squat",
            19: "T Bar Row", 20: "Tricep Dips", 21: "Tricep Pushdown"
        }
        return workout_map.get(obj.workout_type, "Unknown")
    
    def get_latency_rating(self, obj):
        """Return a rating for the latency performance"""
        if obj.avg_response_latency < 50:
            return "Excellent"
        elif obj.avg_response_latency < 100:
            return "Good"
        elif obj.avg_response_latency < 200:
            return "Average"
        else:
            return "Poor"

class MLModelSerializer(serializers.ModelSerializer):
    model_type_display = serializers.CharField(source='get_model_type_display', read_only=True)
    
    class Meta:
        model = MLModel
        fields = [
            'id', 'name', 'model_type', 'model_type_display', 'version',
            'learning_rate', 'epochs', 'batch_size', 
            'accuracy', 'deployed', 'created_at', 
            'updated_at', 'last_trained'
        ]
        read_only_fields = ['created_at', 'updated_at']







