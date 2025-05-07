import os
import django

"""
To avoid the following error keep the below statement

backend-1           |     raise ImproperlyConfigured(
backend-1           | django.core.exceptions.ImproperlyConfigured: Requested setting INSTALLED_APPS, but settings are not configured. You must either define the environment variable DJANGO_SETTINGS_MODULE or call settings.configure() 
before accessing settings.
"""
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from django.contrib.auth.models import User
from DESD_App.models import UserProfile

from django.contrib.auth.models import Group, Permission
from django.contrib.contenttypes.models import ContentType
from DESD_App.models import BillingRecord, MLModel, ModelPerformanceMetric
from django.utils import timezone
from datetime import datetime, timedelta
import random


"""
Create Groups
1-Admin
2-AI Engineer
3-Customer

"""

# Helper function to create user if it doesn't exist
def create_user_if_not_exists(username, password, email=None, groups=None):
    try:
        user = User.objects.get(username=username)
        print(f"User {username} already exists, skipping creation")
        # Ensure user is in the correct group
        if groups:
            for group in groups:
                user.groups.add(group)
            user.save()
        return user
    except User.DoesNotExist:
        print(f"Creating new user: {username}")
        user = User.objects.create_user(username=username, password=password, email=email)
        if groups:
            for group in groups:
                user.groups.add(group)
        user.save()
        return user

# content_type = ContentType.objects.get(app_label='auth', model='user')
# Get or create the content type
content_type, _ = ContentType.objects.get_or_create(app_label='auth', model='user')

machine_learning_permission, created = Permission.objects.get_or_create(
    codename='machine_learning_permission',
    name='Can Train Machine Learning Models',
    content_type=content_type,
)


admin_group, created = Group.objects.get_or_create(name='Admin')
all_permissions = Permission.objects.all()
admin_group.permissions.add(*all_permissions)


ai_engineer_group, created = Group.objects.get_or_create(name='AI Engineer')
ai_engineer_group.permissions.add(machine_learning_permission)
ai_engineer_group.save()
customer_group, created = Group.objects.get_or_create(name='Customer')


# Create users with safeguards
# staff_user = create_user_if_not_exists('Dong2025', 'Dong2025', groups=[admin_group])
# ChatGPT = create_user_if_not_exists('ChatGPT', 'ChatGPT', groups=[ai_engineer_group])
# staff_two = create_user_if_not_exists('Yahia2025', 'Yahia2025', groups=[admin_group])
# Admin_user = create_user_if_not_exists('Admin', 'Admin', email='admin@ufcfur_15_3.com', groups=[admin_group])
# First_name = create_user_if_not_exists('First', 'First', email='first.name@ufcfur_15_3.com', groups=[ai_engineer_group])
# Tensa_name = create_user_if_not_exists('Tensa', 'Tensa', email='tensa.flow@ufcfur_15_3.com', groups=[ai_engineer_group])

# Create client users


list_users = [

    {'username': 'Rob', 'password': 'Rob', 'email':'rob.smith@example.com', 'group': customer_group ,'surname':'smith', 'forename':'Rob', 'title':'Mr'},
    {'username': 'Liz', 'password': 'Liz', 'email':'liz.brown@example.com', 'group': customer_group ,'surname':'brown' , 'forename':'Liz', 'title':'Ms' },
    {'username': 'Hesitant', 'password': 'Hesitant', 'email':'hesitant@example.com', 'group': customer_group ,'surname':'hesitant' , 'forename':'Reven', 'title':'Mr'},
    # {'username': 'Edmond', 'password': 'Edmond', 'email':'edmond.hobbs@darknet.com', 'group': customer_group ,'surname':'hobbs', 'forename':'Edmond', 'title':'Mr'},
    {'username': 'Ahmed', 'password': 'Ahmed', 'email':'Ahmed.moh@example.com', 'group': customer_group ,'surname':'mohamed', 'forename':'Ahmed', 'title':'Mr'},
    {'username': 'Mike', 'password': 'Mike', 'email':'Mike.meh@example.com', 'group': customer_group ,'surname':'mehanheh', 'forename':'Mike', 'title':'Mr'},
    {'username': 'Nathan', 'password': 'Nathan', 'email':'Nathan.hob@example.com', 'group': customer_group ,'surname':'hobb', 'forename':'Nathan', 'title':'Mr'},
    {'username': 'Omar', 'password': 'Omar', 'email':'Omar.ahmed@example.com', 'group': customer_group ,'surname':'ahmed', 'forename':'Omar', 'title':'Mr'},
    {'username': 'Mark', 'password': 'Mark', 'email':'Mark.shon@example.com', 'group': customer_group ,'surname':'shon', 'forename':'Mark', 'title':'Mr'},
    {'username': 'John', 'password': 'John', 'email':'John.sehn@example.com', 'group': customer_group ,'surname':'sehn', 'forename':'John', 'title':'Mr'},
    {'username': 'Paul', 'password': 'Paul', 'email':'Paul.mark@example.com', 'group': customer_group ,'surname':'mark', 'forename':'Paul', 'title':'Mr'},


    {'username': 'First', 'password': 'First', 'email':'first.name@ufcfur_15_3.com', 'group': ai_engineer_group ,'surname':'Second', 'forename':'First', 'title':'Dr'},
    {'username': 'Admin', 'password': 'Admin', 'email':'admin@ufcfur_15_3.com', 'group': admin_group ,'surname':'Admin', 'forename':'Admin', 'title':'Dr'},
    # {'username': 'Admin', 'password': 'Admin', 'email':'admin@ufcfur_15_3.com', 'group': admin_group ,'surname':'Admin', 'forename':'Admin', 'title':'Dr'}
    {'username': 'Dong2025', 'password': 'Dong2025', 'email':'Dong2025.name@ufcfur_15_3.com', 'group': admin_group ,'surname':'wu', 'forename':'Dong', 'title':'Dr'},


]



# First create the users
for user_instance in list_users:
    user, created = User.objects.get_or_create(
        username=user_instance['username'],
        email=user_instance['email']
    )
    
    if created:
        user.set_password(user_instance['password'])
        user.groups.add(user_instance['group'])
        user.save()
    
    # Now create or update the user profile
    UserProfile.objects.update_or_create(
        user=user,  # Link to the user we just created
        defaults={
            'title': user_instance['title'],
            'forename': user_instance['forename'],
            'surname': user_instance['surname']
        }
    )




# Set admin users as staff
users = User.objects.filter(groups=admin_group, is_staff=False)
for user in users:
    user.is_staff = True
    user.profile.is_approved = True
    user.save()
    user.profile.save()

#approve AI engineer users
users = User.objects.filter(groups=ai_engineer_group)
for user in users:
    user.profile.is_approved = True
    user.profile.save()

# Print permissions for debugging
permissions = ai_engineer_group.permissions.all()
for permission in permissions:
    print(f"Permission: {permission.codename} - {permission.name}")
    
# Generate billing data for testing
print("Generating billing data for testing...")

# Check if billing records already exist
if BillingRecord.objects.count() > 0:
    print(f"Found {BillingRecord.objects.count()} existing billing records, skipping billing data generation")
else:
    # Get all customer users
    customer_users = User.objects.filter(groups=customer_group)
    subscription_types = ['free', 'basic', 'premium', 'enterprise']
    status_types = ['pending', 'paid', 'overdue', 'cancelled']

    # Generate random billing data for the past year
    today = datetime.now().date()
    for user in customer_users:
        # Generate between 3-10 billing records per user
        for i in range(random.randint(3, 10)):
            # Random date in the past year
            billing_date = today - timedelta(days=random.randint(1, 365))
            due_date = billing_date + timedelta(days=30)
            
            # Create billing record
            BillingRecord.objects.create(
                user=user,
                amount=random.uniform(10.0, 500.0),
                subscription_type=random.choice(subscription_types),
                billing_date=billing_date,
                due_date=due_date,
                status=random.choice(status_types),
                description=f"Monthly subscription for {user.username}",
                api_calls=random.randint(100, 10000),
                data_usage=random.uniform(50.0, 2000.0)
            )

    print(f"Created {BillingRecord.objects.count()} billing records for testing")

# Generate ML model data for testing
print("Generating ML model data for testing...")

# Check if ML models already exist
if MLModel.objects.count() > 0:
    print(f"Found {MLModel.objects.count()} existing ML models, skipping model data generation")
else:
    # Define model templates
    ml_models = [
        # Workout Classification models
        {
            'name': 'WorkoutClassifier-Base',
            'model_type': 'workout',
            'version': '1.0.0',
            'description': 'Base model for classifying different workout types',
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32,
            'accuracy': 0.87,
            'deployed': True
        },
        {
            'name': 'WorkoutClassifier-Advanced',
            'model_type': 'workout',
            'version': '1.2.0',
            'description': 'Advanced model with fine-tuned parameters for workout classification',
            'learning_rate': 0.0005,
            'epochs': 200,
            'batch_size': 64,
            'accuracy': 0.92,
            'deployed': False
        },
        {
            'name': 'WorkoutClassifier-Experimental',
            'model_type': 'workout',
            'version': '2.0.0-beta',
            'description': 'Experimental model using transformer architecture',
            'learning_rate': 0.0003,
            'epochs': 150,
            'batch_size': 16,
            'accuracy': 0.89,
            'deployed': False
        },
        
        # Muscle Activation models
        {
            'name': 'MuscleGroupAnalyzer-Standard',
            'model_type': 'muscle',
            'version': '1.0.0',
            'description': 'Standard model for muscle activation tracking',
            'learning_rate': 0.001,
            'epochs': 80,
            'batch_size': 32,
            'accuracy': 0.85,
            'deployed': True
        },
        {
            'name': 'MuscleGroupAnalyzer-Pro',
            'model_type': 'muscle',
            'version': '1.5.0',
            'description': 'Professional model with improved accuracy for muscle group activation',
            'learning_rate': 0.0008,
            'epochs': 120,
            'batch_size': 48,
            'accuracy': 0.91,
            'deployed': False
        },
        
        # Pose Correction models (replacing Displacement Estimation Values)
        {
            'name': 'PoseCorrection-Base',
            'model_type': 'pose',
            'version': '1.0.0',
            'description': 'Base model for form correction during workouts',
            'learning_rate': 0.0012,
            'epochs': 90,
            'batch_size': 32,
            'accuracy': 0.83,
            'deployed': True
        },
        {
            'name': 'PoseCorrection-Advanced',
            'model_type': 'pose',
            'version': '1.3.0',
            'description': 'Advanced model for precise workout form correction',
            'learning_rate': 0.0007,
            'epochs': 150,
            'batch_size': 32,
            'accuracy': 0.89,
            'deployed': False
        },
    ]
    
    # Create the models
    for model_data in ml_models:
        model = MLModel.objects.create(**model_data)
        print(f"Created ML model: {model.name} ({model.get_model_type_display()})")
    
    print(f"Created {MLModel.objects.count()} ML models for testing")

# Generate model performance metrics
print("Generating model performance metrics for testing...")

# Check if performance metrics already exist
if ModelPerformanceMetric.objects.count() > 0:
    print(f"Found {ModelPerformanceMetric.objects.count()} existing model performance metrics, skipping generation")
else:
    # Common workout types
    workout_types = [0, 1, 12, 18]  # Barbell Bicep Curl, Bench Press, Plank, Squat
    
    # Create metrics for the past 30 days
    today = timezone.now()
    for day in range(30):
        timestamp = today - timedelta(days=day)
        
        # Create multiple records per day with different workout types
        for _ in range(random.randint(5, 15)):
            workout_type = random.choice(workout_types)
            
            # Base confidence and metrics with some randomness
            base_confidence = random.uniform(0.75, 0.95)
            latency = random.randint(50, 250)
            
            ModelPerformanceMetric.objects.create(
                timestamp=timestamp - timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59)),
                model_version=f"1.{random.randint(0, 3)}.0",
                workout_type=workout_type,
                avg_prediction_confidence=base_confidence,
                min_prediction_confidence=base_confidence * 0.85,
                max_prediction_confidence=min(0.99, base_confidence * 1.15),
                correction_magnitude_avg=random.uniform(0.02, 0.1),
                stable_prediction_rate=random.uniform(0.8, 0.98),
                avg_response_latency=latency,
                processing_time_per_frame=random.randint(30, 60),
                time_to_first_correction=random.randint(300, 800),
                frame_processing_rate=random.uniform(20, 30),
                cpu_usage=random.uniform(5, 25),
                memory_usage=random.uniform(200, 600)
            )
    
    print(f"Created {ModelPerformanceMetric.objects.count()} model performance metrics for testing")