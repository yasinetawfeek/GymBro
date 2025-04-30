from django.test import TestCase
from django.contrib.auth.models import User, Group
from rest_framework.test import APIClient
from DESD_App.models import Subscription, Invoice
from datetime import datetime, timedelta
from decimal import Decimal

class SubscriptionTest(TestCase):
    def setUp(self):
        # Create an Admin group
        admin_group = Group.objects.create(name='Admin')
        
        # Create a test user
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpassword'
        )
        
        # Set up API client
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)
    
    def test_subscription_plan_change_cancels_pending_invoices(self):
        """
        Test that changing subscription plans cancels any pending invoices
        """
        # First subscribe to basic plan
        response = self.client.post('/api/subscriptions/subscribe/', {
            'plan': 'basic',
            'auto_renew': True
        })
        
        self.assertEqual(response.status_code, 200)
        
        # Verify invoice was created with pending status
        invoices = Invoice.objects.filter(user=self.user)
        self.assertEqual(invoices.count(), 1)
        self.assertEqual(invoices[0].status, 'pending')
        self.assertEqual(invoices[0].amount, Decimal('9.99'))
        
        # Now change to premium plan
        response = self.client.post('/api/subscriptions/subscribe/', {
            'plan': 'premium',
            'auto_renew': True
        })
        
        self.assertEqual(response.status_code, 200)
        
        # Verify the old invoice is now cancelled
        old_invoice = Invoice.objects.get(id=invoices[0].id)
        self.assertEqual(old_invoice.status, 'cancelled')
        
        # Verify a new invoice was created
        invoices = Invoice.objects.filter(user=self.user)
        self.assertEqual(invoices.count(), 2)
        
        # Get the new invoice (which should be the only pending one)
        new_invoice = Invoice.objects.filter(user=self.user, status='pending').first()
        self.assertIsNotNone(new_invoice)
        self.assertEqual(new_invoice.amount, Decimal('29.99'))  # Premium plan price
        
        # Make sure we have different invoice IDs
        self.assertNotEqual(old_invoice.id, new_invoice.id) 