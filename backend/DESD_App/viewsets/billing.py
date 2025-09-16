"""
Billing and subscription viewsets for the GymBro application.
"""
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from rest_framework.decorators import action
from django.db.models import Sum, Count
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from ..models import Subscription, Invoice, BillingRecord
from ..serializers import SubscriptionSerializer, InvoiceSerializer, BillingRecordSerializer
from ..permissions import IsOwner


class SubscriptionViewSet(viewsets.ModelViewSet):
    """ViewSet for subscription management."""
    
    queryset = Subscription.objects.all()
    serializer_class = SubscriptionSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        """Return subscriptions for the current user or all if admin."""
        if self.request.user.is_staff:
            return Subscription.objects.all()
        return Subscription.objects.filter(user=self.request.user)
    
    @swagger_auto_schema(
        operation_description="Get user's subscription information",
        operation_summary="Get subscriptions",
        responses={200: openapi.Response(description="User subscriptions")}
    )
    def list(self, request):
        """Get user's subscriptions."""
        subscriptions = self.get_queryset()
        serializer = self.get_serializer(subscriptions, many=True)
        return Response(serializer.data)
    
    @swagger_auto_schema(
        operation_description="Create a new subscription",
        operation_summary="Create subscription",
        responses={
            201: openapi.Response(description="Subscription created successfully"),
            400: openapi.Response(description="Invalid data provided")
        }
    )
    def create(self, request):
        """Create a new subscription."""
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            serializer.save(user=request.user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    @swagger_auto_schema(
        operation_description="Cancel a subscription",
        operation_summary="Cancel subscription",
        responses={200: openapi.Response(description="Subscription cancelled successfully")}
    )
    @action(detail=True, methods=['patch'])
    def cancel(self, request, pk=None):
        """Cancel a subscription."""
        subscription = self.get_object()
        subscription.is_active = False
        subscription.save()
        return Response({'status': 'Subscription cancelled successfully'})


class InvoiceViewSet(viewsets.ModelViewSet):
    """ViewSet for invoice management."""
    
    queryset = Invoice.objects.all()
    serializer_class = InvoiceSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        """Return invoices for the current user or all if admin."""
        if self.request.user.is_staff:
            return Invoice.objects.all()
        return Invoice.objects.filter(user=self.request.user)
    
    @swagger_auto_schema(
        operation_description="Get user's invoices",
        operation_summary="Get invoices",
        responses={200: openapi.Response(description="User invoices")}
    )
    def list(self, request):
        """Get user's invoices."""
        invoices = self.get_queryset()
        serializer = self.get_serializer(invoices, many=True)
        return Response(serializer.data)
    
    @swagger_auto_schema(
        operation_description="Mark invoice as paid",
        operation_summary="Pay invoice",
        responses={200: openapi.Response(description="Invoice marked as paid")}
    )
    @action(detail=True, methods=['patch'])
    def pay(self, request, pk=None):
        """Mark invoice as paid."""
        invoice = self.get_object()
        invoice.status = 'paid'
        invoice.save()
        return Response({'status': 'Invoice marked as paid'})


class BillingViewSet(viewsets.ModelViewSet):
    """ViewSet for billing record management."""
    
    queryset = BillingRecord.objects.all()
    serializer_class = BillingRecordSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        """Return billing records for the current user or all if admin."""
        if self.request.user.is_staff:
            return BillingRecord.objects.all()
        return BillingRecord.objects.filter(user=self.request.user)
    
    @swagger_auto_schema(
        operation_description="Get billing overview with statistics",
        operation_summary="Get billing overview",
        responses={200: openapi.Response(
            description="Billing overview",
            examples={
                "application/json": {
                    "total_amount": 150.00,
                    "total_invoices": 5,
                    "pending_amount": 50.00,
                    "paid_amount": 100.00
                }
            }
        )}
    )
    @action(detail=False, methods=['get'])
    def overview(self, request):
        """Get billing overview with statistics."""
        billing_records = self.get_queryset()
        
        total_amount = billing_records.aggregate(Sum('amount'))['amount__sum'] or 0
        total_invoices = billing_records.count()
        pending_amount = billing_records.filter(status='pending').aggregate(Sum('amount'))['amount__sum'] or 0
        paid_amount = billing_records.filter(status='paid').aggregate(Sum('amount'))['amount__sum'] or 0
        
        return Response({
            'total_amount': total_amount,
            'total_invoices': total_invoices,
            'pending_amount': pending_amount,
            'paid_amount': paid_amount
        })