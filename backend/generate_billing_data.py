import os
import django
import random
from datetime import datetime, timedelta

# Configure Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

# Import models
from django.contrib.auth.models import User, Group
from DESD_App.models import BillingRecord, Invoice, Subscription, UsageRecord

print("Generating billing data for testing...")

# Delete existing billing records to avoid duplicates
BillingRecord.objects.all().delete()
Invoice.objects.all().delete()
UsageRecord.objects.all().delete()
Subscription.objects.all().delete()

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
    # Create a subscription for the user
    subscription_type = random.choice(subscription_types)
    subscription_start = today - timedelta(days=random.randint(30, 365))
    subscription_end = subscription_start + timedelta(days=365)
    
    # Set price based on plan
    if subscription_type == 'free':
        price = 0.0
        max_api_calls = 50
        max_data_usage = 100
    elif subscription_type == 'basic':
        price = 9.99
        max_api_calls = 1000
        max_data_usage = 500
    elif subscription_type == 'premium':
        price = 29.99
        max_api_calls = 5000
        max_data_usage = 2000
    else:  # enterprise
        price = 99.99
        max_api_calls = 100000
        max_data_usage = 10000
    
    # Create subscription
    subscription = Subscription.objects.create(
        user=user,
        plan=subscription_type,
        start_date=subscription_start,
        end_date=subscription_end,
        is_active=True,
        auto_renew=random.choice([True, False]),
        price=price,
        max_api_calls=max_api_calls,
        max_data_usage=max_data_usage
    )
    
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
        billing_record = BillingRecord.objects.create(
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
        
        # Create corresponding invoice
        invoice_status = random.choice(status_types)
        payment_date = None
        if invoice_status == 'paid':
            payment_date = billing_date + timedelta(days=random.randint(1, 10))
            
        Invoice.objects.create(
            subscription=subscription,
            user=user,
            amount=amount,
            invoice_date=billing_date,
            due_date=due_date,
            status=invoice_status,
            payment_date=payment_date,
            description=f"Invoice for {user.username}'s {sub_type} subscription"
        )
        
    # Generate usage records
    for i in range(random.randint(5, 15)):
        session_start = datetime.now() - timedelta(days=random.randint(1, 180), 
                                                hours=random.randint(1, 24))
        session_duration = random.randint(15*60, 120*60)  # 15-120 minutes in seconds
        session_end = session_start + timedelta(seconds=session_duration)
        
        frames_processed = random.randint(500, 10000)
        corrections_sent = random.randint(20, 200)
        
        # Calculate billable amount based on subscription type
        if subscription_type == 'free':
            billable_rate = 0.002
        elif subscription_type == 'basic':
            billable_rate = 0.001
        elif subscription_type == 'premium':
            billable_rate = 0.0005
        else:  # enterprise
            billable_rate = 0.0001
            
        billable_amount = round(corrections_sent * billable_rate, 2)
        
        UsageRecord.objects.create(
            user=user,
            session_start=session_start,
            session_end=session_end,
            total_duration=session_duration,
            frames_processed=frames_processed,
            corrections_sent=corrections_sent,
            workout_type=random.randint(1, 5),
            billable_amount=billable_amount,
            subscription_plan=subscription_type,
            is_active=False,
            platform=random.choice(['Web', 'iOS', 'Android'])
        )

print(f"Created {BillingRecord.objects.count()} billing records for testing")
print(f"Created {Invoice.objects.count()} invoices for testing")
print(f"Created {UsageRecord.objects.count()} usage records for testing")
print(f"Created {Subscription.objects.count()} subscriptions for testing") 