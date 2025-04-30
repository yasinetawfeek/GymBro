import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import { 
  Calendar, CreditCard, CheckCircle, 
  AlertCircle, Clock, Download, Printer, Share2, X
} from 'lucide-react';
import LoadingSpinner from './LoadingSpinner';

const InvoiceDetailModal = ({ invoiceId, isDarkMode, onClose, onPaymentSuccess }) => {
  const [invoice, setInvoice] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [paymentProcessing, setPaymentProcessing] = useState(false);
  const [paymentSuccess, setPaymentSuccess] = useState(false);
  const [paymentError, setPaymentError] = useState(null);

  // Format currency with $ symbol and 2 decimal places
  const formatCurrency = (amount) => {
    // Check if amount is null, undefined, NaN or not a valid number
    if (amount === null || amount === undefined || isNaN(amount) || amount === '') {
      return '$0.00'; // Return default value for invalid amounts
    }
    
    // Ensure amount is treated as a number
    const numericAmount = parseFloat(amount);
    
    // Check if conversion resulted in a valid number
    if (isNaN(numericAmount)) {
      return '$0.00';
    }
    
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(numericAmount);
  };

  // Format date to readable format
  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    
    try {
      const date = new Date(dateString);
      // Check if date is valid
      if (isNaN(date.getTime())) {
        return 'Invalid Date';
      }
      return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
      });
    } catch (error) {
      console.error('Date formatting error:', error);
      return 'Invalid Date';
    }
  };

  // Get formatted subscription details for display
  const getSubscriptionDetails = (invoice) => {
    // First try to get plan name from various sources
    const planName = invoice?.subscription_plan || 
                    invoice?.subscription_type || 
                    (invoice?.subscription?.plan) || 
                    'Standard Plan';
    
    // Get start and end dates
    const startDate = invoice?.subscription_start_date ? 
                      formatDate(invoice.subscription_start_date) : 
                      formatDate(invoice.invoice_date);
    
    const endDate = invoice?.subscription_end_date ? 
                   formatDate(invoice.subscription_end_date) : 
                   formatDate(invoice.due_date);
    
    return {
      planName: planName.charAt(0).toUpperCase() + planName.slice(1),
      dateRange: `${startDate} - ${endDate}`
    };
  };

  // Get status badge style based on invoice status
  const getStatusBadgeStyle = (status) => {
    switch (status) {
      case 'paid':
        return isDarkMode 
          ? 'bg-green-900/30 text-green-400 border border-green-700/50'
          : 'bg-green-100 text-green-700 border border-green-200';
      case 'pending':
        return isDarkMode
          ? 'bg-yellow-900/30 text-yellow-400 border border-yellow-700/50'
          : 'bg-yellow-100 text-yellow-700 border border-yellow-200';
      case 'overdue':
        return isDarkMode
          ? 'bg-red-900/30 text-red-400 border border-red-700/50'
          : 'bg-red-100 text-red-700 border border-red-200';
      case 'cancelled':
        return isDarkMode
          ? 'bg-gray-900/30 text-gray-400 border border-gray-700/50'
          : 'bg-gray-100 text-gray-700 border border-gray-200';
      default:
        return isDarkMode
          ? 'bg-gray-900/30 text-gray-400 border border-gray-700/50'
          : 'bg-gray-100 text-gray-700 border border-gray-200';
    }
  };

  // Get status icon based on invoice status
  const StatusIcon = ({ status }) => {
    switch (status) {
      case 'paid':
        return <CheckCircle className="w-4 h-4 mr-1" />;
      case 'pending':
        return <Clock className="w-4 h-4 mr-1" />;
      case 'overdue':
        return <AlertCircle className="w-4 h-4 mr-1" />;
      default:
        return <Clock className="w-4 h-4 mr-1" />;
    }
  };

  // Fetch invoice data on component mount
  useEffect(() => {
    const fetchInvoice = async () => {
      try {
        console.log("Attempting to fetch invoice with ID:", invoiceId);
        
        const response = await axios.get(`http://localhost:8000/api/invoices/${invoiceId}/`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('access_token')}`
          }
        });
        
        // Add detailed console logging for debugging
        console.log("=== INVOICE DATA DEBUGGING ===");
        console.log("Raw invoice response:", response.data);
        console.log("Subscription plan:", response.data.subscription_plan);
        console.log("Subscription type:", response.data.subscription_type);
        console.log("Subscription name:", response.data.subscription_name);
        console.log("Subscription object:", response.data.subscription);
        console.log("Subscription start date:", response.data.subscription_start_date);
        console.log("Subscription end date:", response.data.subscription_end_date);
        console.log("================================");
        
        // Add subscription_type to subscription_plan if it's not already there
        const invoiceData = response.data;
        if (invoiceData.subscription && invoiceData.subscription.plan && !invoiceData.subscription_plan) {
          console.log("Adding missing subscription_plan from subscription.plan");
          invoiceData.subscription_plan = invoiceData.subscription.plan;
        }
        if (invoiceData.subscription_type && !invoiceData.subscription_plan) {
          console.log("Adding missing subscription_plan from subscription_type");
          invoiceData.subscription_plan = invoiceData.subscription_type;
        }
        
        setInvoice(invoiceData);
        setLoading(false);
      } catch (err) {
        console.error('Error fetching invoice:', err);
        console.error('Error details:', err.response || err.message);
        setError('Failed to load invoice details. Please try again.');
        setLoading(false);
      }
    };

    fetchInvoice();
  }, [invoiceId]);

  // Handle invoice payment
  const handlePayInvoice = async () => {
    if (paymentProcessing || invoice?.status !== 'pending') return;

    setPaymentProcessing(true);
    setPaymentError(null);

    try {
      await axios.post(`http://localhost:8000/api/invoices/${invoiceId}/pay/`, {}, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        }
      });
      
      // Update invoice data after payment
      const updatedInvoice = await axios.get(`http://localhost:8000/api/invoices/${invoiceId}/`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        }
      });
      
      setInvoice(updatedInvoice.data);
      setPaymentSuccess(true);
      setPaymentProcessing(false);
      
      // Notify parent component of successful payment
      if (onPaymentSuccess) {
        onPaymentSuccess();
      }
      
      // Auto-hide success message after 5 seconds
      setTimeout(() => {
        setPaymentSuccess(false);
      }, 5000);
      
    } catch (err) {
      console.error('Error processing payment:', err);
      setPaymentError('Payment processing failed. Please try again or contact support.');
      setPaymentProcessing(false);
    }
  };

  // Function to print the invoice
  const handlePrintInvoice = () => {
    // Clone invoice content to a new window for printing
    const printContent = document.getElementById('invoice-content').cloneNode(true);
    const printWindow = window.open('', '_blank');
    
    if (printWindow) {
      printWindow.document.write('<html><head><title>Invoice</title>');
      printWindow.document.write('<link href="https://unpkg.com/tailwindcss@^2/dist/tailwind.min.css" rel="stylesheet">');
      printWindow.document.write('</head><body class="p-8 bg-white">');
      printWindow.document.write(printContent.outerHTML);
      printWindow.document.write('</body></html>');
      printWindow.document.close();
      
      // Wait for styles to load then print
      setTimeout(() => {
        printWindow.print();
        printWindow.close();
      }, 250);
    } else {
      alert('Please allow pop-ups to print the invoice.');
    }
  };

  // Function to download the invoice (in a real app, this would generate a PDF)
  const handleDownloadInvoice = () => {
    // This is a placeholder - in a real app you would generate and download a PDF
    alert('In a production environment, this would download a PDF of the invoice.');
  };

  // Function to share the invoice
  const handleShareInvoice = () => {
    // This is a placeholder - in a real app you would implement sharing functionality
    if (navigator.share) {
      navigator.share({
        title: `Invoice #${invoice.id}`,
        text: `View invoice #${invoice.id} for ${formatCurrency(invoice.amount)}`,
        url: window.location.href
      });
    } else {
      // Fallback for browsers that don't support Web Share API
      navigator.clipboard.writeText(window.location.href);
      alert('Invoice link copied to clipboard!');
    }
  };

  // Modal layout with backdrop
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 overflow-y-auto"
      onClick={onClose}
    >
      {/* Modal content - stop propagation to prevent closing when clicking inside the modal */}
      <motion.div 
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        transition={{ duration: 0.2 }}
        className="relative w-full max-w-4xl max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close button */}
        <button
          onClick={onClose}
          className={`absolute top-4 right-4 z-10 p-2 rounded-full ${
            isDarkMode 
              ? 'bg-gray-700 text-gray-300 hover:bg-gray-600 hover:text-white' 
              : 'bg-gray-200 text-gray-600 hover:bg-gray-300 hover:text-gray-900'
          }`}
          aria-label="Close modal"
        >
          <X className="w-5 h-5" />
        </button>
        
        {loading ? (
          <div className={`flex items-center justify-center p-12 rounded-lg ${
            isDarkMode ? 'bg-gray-800' : 'bg-white'
          }`}>
            <LoadingSpinner size="lg" />
          </div>
        ) : error ? (
          <div className={`p-8 rounded-lg ${isDarkMode ? 'bg-gray-800 text-white' : 'bg-white text-gray-900'}`}>
            <div className="text-center">
              <AlertCircle className="w-16 h-16 mx-auto text-red-500 mb-4" />
              <h2 className="text-2xl font-bold mb-2">Error Loading Invoice</h2>
              <p className="mb-6">{error}</p>
              <button 
                onClick={onClose}
                className={`px-4 py-2 rounded-lg ${
                  isDarkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-200 hover:bg-gray-300'
                }`}
              >
                Close
              </button>
            </div>
          </div>
        ) : !invoice ? (
          <div className={`p-8 rounded-lg ${isDarkMode ? 'bg-gray-800 text-white' : 'bg-white text-gray-900'}`}>
            <div className="text-center">
              <AlertCircle className="w-16 h-16 mx-auto text-amber-500 mb-4" />
              <h2 className="text-2xl font-bold mb-2">Invoice Not Found</h2>
              <p className="mb-6">The requested invoice could not be found or you don't have permission to view it.</p>
              <button 
                onClick={onClose} 
                className={`px-4 py-2 rounded-lg ${
                  isDarkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-gray-200 hover:bg-gray-300'
                }`}
              >
                Close
              </button>
            </div>
          </div>
        ) : (
          <div 
            id="invoice-content"
            className={`rounded-lg shadow-xl overflow-hidden ${
              isDarkMode ? 'bg-gray-800 text-white' : 'bg-white text-gray-900'
            }`}
          >
            {/* Header */}
            <div className={`p-6 border-b ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                <div>
                  <h2 className="text-2xl font-bold flex items-center">
                    Invoice #<span className={isDarkMode ? 'text-purple-400' : 'text-indigo-600'}>{invoice.id}</span>
                  </h2>
                  <div className="flex items-center mt-2">
                    <Calendar className="w-4 h-4 mr-1 text-gray-500" />
                    <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>
                      {formatDate(invoice.invoice_date)}
                    </span>
                  </div>
                </div>
                
                <div className="flex items-center">
                  <span className={`px-3 py-1 rounded-full text-sm font-medium flex items-center ${getStatusBadgeStyle(invoice.status)}`}>
                    <StatusIcon status={invoice.status} />
                    {invoice.status ? invoice.status.charAt(0).toUpperCase() + invoice.status.slice(1) : 'Unknown'}
                  </span>
                </div>
              </div>
            </div>
            
            {/* Customer and Invoice Details */}
            <div className="p-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Customer Information */}
                <div>
                  <h3 className={`text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    CUSTOMER
                  </h3>
                  <p className="font-medium text-lg">{invoice.username}</p>
                  <p className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>
                    {invoice.user_email || 'Email not available'}
                  </p>
                </div>
                
                {/* Invoice Details */}
                <div>
                  <h3 className={`text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    INVOICE DETAILS
                  </h3>
                  <div className="space-y-1">
                    <div className="flex justify-between">
                      <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Invoice Number:</span>
                      <span className="font-medium">{invoice.id}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Due Date:</span>
                      <span className="font-medium">{formatDate(invoice.due_date)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Subscription Plan:</span>
                      <span className="font-medium capitalize">
                        {invoice.subscription_plan || invoice.subscription_type || 
                         (invoice.subscription && invoice.subscription.plan) || 'Standard Plan'}
                      </span>
                    </div>
                    {invoice.status === 'paid' && invoice.payment_date && (
                      <div className="flex justify-between">
                        <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Payment Date:</span>
                        <span className="font-medium">{formatDate(invoice.payment_date)}</span>
                      </div>
                    )}
                    {invoice.status === 'overdue' && (
                      <div className="flex justify-between">
                        <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Days Overdue:</span>
                        <span className="font-medium text-red-500">{invoice.days_overdue}</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
            
            {/* Invoice Items */}
            <div className={`p-6 border-t ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
              <h3 className={`text-sm font-medium mb-4 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                INVOICE ITEMS
              </h3>
              <div className={`rounded-lg overflow-hidden border ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                {/* Table Header */}
                <div className={`grid grid-cols-12 gap-4 px-4 py-3 ${
                  isDarkMode ? 'bg-gray-900/50 text-gray-300' : 'bg-gray-50 text-gray-600'
                } text-sm`}>
                  <div className="col-span-8">Description</div>
                  <div className="col-span-4 text-right">Amount</div>
                </div>
                
                {/* Table Body */}
                <div className={`px-4 py-4 ${isDarkMode ? 'bg-gray-800/50' : 'bg-white'}`}>
                  <div className="grid grid-cols-12 gap-4">
                    <div className="col-span-8">
                      {/* Use our new helper function to get consistent subscription details */}
                      {(() => {
                        const subscriptionDetails = getSubscriptionDetails(invoice);
                        return (
                          <>
                            <p className="font-medium capitalize">
                              {subscriptionDetails.planName} Subscription
                            </p>
                            <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                              {invoice.description || `Monthly subscription fee (${subscriptionDetails.dateRange})`}
                            </p>
                          </>
                        );
                      })()}
                    </div>
                    <div className="col-span-4 text-right font-medium">
                      {formatCurrency(invoice.amount)}
                    </div>
                  </div>
                </div>
                
                {/* Table Footer */}
                <div className={`grid grid-cols-12 gap-4 px-4 py-3 border-t ${
                  isDarkMode ? 'bg-gray-900/50 border-gray-700' : 'bg-gray-50 border-gray-200'
                }`}>
                  <div className="col-span-8 text-right font-medium">Total</div>
                  <div className="col-span-4 text-right font-medium">
                    {formatCurrency(invoice.amount)}
                  </div>
                </div>
              </div>
            </div>
            
            {/* Payment Section - Only shown for pending invoices */}
            {invoice.status === 'pending' && (
              <div className={`p-6 border-t ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
                <h3 className={`text-sm font-medium mb-4 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                  PAYMENT
                </h3>
                {paymentSuccess && (
                  <div className={`mb-4 p-3 rounded-lg flex items-start ${
                    isDarkMode ? 'bg-green-900/30 text-green-400 border border-green-700/50' : 'bg-green-100 text-green-700 border border-green-200'
                  }`}>
                    <CheckCircle className="w-5 h-5 mr-2 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-medium">Payment Successful!</p>
                      <p className="text-sm">Your invoice has been paid and your subscription is active.</p>
                    </div>
                  </div>
                )}
                
                {paymentError && (
                  <div className={`mb-4 p-3 rounded-lg flex items-start ${
                    isDarkMode ? 'bg-red-900/30 text-red-400 border border-red-700/50' : 'bg-red-100 text-red-700 border border-red-200'
                  }`}>
                    <AlertCircle className="w-5 h-5 mr-2 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-medium">Payment Error</p>
                      <p className="text-sm">{paymentError}</p>
                    </div>
                  </div>
                )}
                
                <motion.button
                  onClick={handlePayInvoice}
                  disabled={paymentProcessing || paymentSuccess}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className={`w-full py-3 px-4 rounded-lg font-medium flex items-center justify-center ${
                    paymentProcessing || paymentSuccess
                      ? isDarkMode ? 'bg-gray-700 text-gray-400' : 'bg-gray-200 text-gray-500'
                      : isDarkMode ? 'bg-purple-600 hover:bg-purple-700 text-white' : 'bg-indigo-600 hover:bg-indigo-700 text-white'
                  }`}
                >
                  {paymentProcessing ? (
                    <>
                      <LoadingSpinner size="sm" className="mr-2" />
                      Processing Payment...
                    </>
                  ) : paymentSuccess ? (
                    <>
                      <CheckCircle className="w-5 h-5 mr-2" />
                      Payment Successful
                    </>
                  ) : (
                    <>
                      <CreditCard className="w-5 h-5 mr-2" />
                      Pay {formatCurrency(invoice.amount)}
                    </>
                  )}
                </motion.button>
              </div>
            )}
            
            {/* Actions Footer */}
            <div className={`p-6 border-t ${isDarkMode ? 'border-gray-700' : 'border-gray-200'}`}>
              <div className="flex flex-wrap gap-2">
                <button
                  onClick={handlePrintInvoice}
                  className={`flex items-center px-3 py-2 rounded-lg ${
                    isDarkMode
                      ? 'bg-gray-700 hover:bg-gray-600 text-white'
                      : 'bg-gray-200 hover:bg-gray-300 text-gray-800'
                  }`}
                >
                  <Printer className="w-4 h-4 mr-2" />
                  Print Invoice
                </button>
                
                <button
                  onClick={handleShareInvoice}
                  className={`flex items-center px-3 py-2 rounded-lg ${
                    isDarkMode
                      ? 'bg-gray-700 hover:bg-gray-600 text-white'
                      : 'bg-gray-200 hover:bg-gray-300 text-gray-800'
                  }`}
                >
                  <Share2 className="w-4 h-4 mr-2" />
                  Share
                </button>
                
                <button
                  onClick={onClose}
                  className={`flex items-center px-3 py-2 ml-auto rounded-lg ${
                    isDarkMode
                      ? 'bg-gray-700 hover:bg-gray-600 text-white'
                      : 'bg-gray-200 hover:bg-gray-300 text-gray-800'
                  }`}
                >
                  <X className="w-4 h-4 mr-2" />
                  Close
                </button>
              </div>
            </div>
          </div>
        )}
      </motion.div>
    </motion.div>
  );
};

export default InvoiceDetailModal; 