import os
import django
import random
from datetime import datetime, timedelta

# Configure Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

# Import models
from django.contrib.auth.models import User, Group
from DESD_App.models import BillingRecord

print("Generating billing data for testing...")

# Delete existing billing records to avoid duplicates
BillingRecord.objects.all().delete()

# Get all customer users
customer_group = Group.objects.get(name='Customer')
customer_users = User.objects.filter(groups=customer_group)

if not customer_users:
    print("No customer users found. Please run automatically_start_script.py first.")
    exit(1)

# Billing configuration
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
        
        # Amount calculation
        if random.choice(['fixed', 'usage']) == 'fixed':
            # Fixed subscription cost
            sub_type = random.choice(subscription_types)
            if sub_type == 'free':
                amount = 0.0
            elif sub_type == 'basic':
                amount = 9.99
            elif sub_type == 'premium':
                amount = 29.99
            else:  # enterprise
                amount = 99.99
        else:
            # Usage-based cost
            sub_type = random.choice(['basic', 'premium', 'enterprise'])
            api_calls = random.randint(100, 10000)
            data_usage = random.uniform(50.0, 2000.0)
            
            # Calculate amount based on usage
            amount = 0.0
            if sub_type == 'basic':
                amount = 4.99 + (api_calls * 0.001) + (data_usage * 0.01)
            elif sub_type == 'premium':
                amount = 19.99 + (api_calls * 0.0005) + (data_usage * 0.005)
            else:  # enterprise
                amount = 49.99 + (api_calls * 0.0001) + (data_usage * 0.001)
            
            # Round to 2 decimal places
            amount = round(amount, 2)
        
        # Create billing record
        BillingRecord.objects.create(
            user=user,
            amount=amount,
            subscription_type=sub_type,
            billing_date=billing_date,
            due_date=due_date,
            status=random.choice(status_types),
            description=f"Monthly subscription for {user.username}",
            api_calls=random.randint(100, 10000),
            data_usage=random.uniform(50.0, 2000.0)
        )

print(f"Created {BillingRecord.objects.count()} billing records for testing") 