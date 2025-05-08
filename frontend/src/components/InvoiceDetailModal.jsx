import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Calendar, CreditCard, CheckCircle, 
  AlertCircle, Clock, Download, Printer, Share2, X,
  Mail, ChevronRight, ArrowDown
} from 'lucide-react';
import LoadingSpinner from './LoadingSpinner';
import { invoiceService } from '../services/apiService';

// Animation variants
const fadeIn = {
  hidden: { opacity: 0 },
  visible: { 
    opacity: 1,
    transition: { duration: 0.3 }
  },
  exit: { 
    opacity: 0,
    transition: { duration: 0.2 }
  }
};

const modalVariant = {
  hidden: { opacity: 0, scale: 0.95, y: 10 },
  visible: { 
    opacity: 1, 
    scale: 1,
    y: 0,
    transition: { type: "spring", stiffness: 300, damping: 25 }
  },
  exit: { 
    opacity: 0, 
    scale: 0.95,
    y: 10,
    transition: { duration: 0.2 }
  }
};

const InvoiceDetailModal = ({ invoiceId, isDarkMode, onClose, onPaymentSuccess, isAdminView = false }) => {
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
    
    return new Intl.NumberFormat('en-GB', {
      style: 'currency',
      currency: 'GBP',
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
      return date.toLocaleDateString('en-GB', {
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
          ? 'bg-green-500/20 text-green-400 border border-green-700/50'
          : 'bg-green-100 text-green-700 border border-green-200';
      case 'pending':
        return isDarkMode
          ? 'bg-amber-500/20 text-amber-400 border border-amber-700/50'
          : 'bg-amber-100 text-amber-700 border border-amber-200';
      case 'overdue':
        return isDarkMode
          ? 'bg-red-500/20 text-red-400 border border-red-700/50'
          : 'bg-red-100 text-red-700 border border-red-200';
      case 'cancelled':
        return isDarkMode
          ? 'bg-gray-500/20 text-gray-400 border border-gray-700/50'
          : 'bg-gray-100 text-gray-700 border border-gray-200';
      default:
        return isDarkMode
          ? 'bg-gray-500/20 text-gray-400 border border-gray-700/50'
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
      if (!invoiceId) return;
      
      setLoading(true);
      setError(null);
      
      try {
        // First try to get invoice details from invoice endpoint
        let invoiceData;
        try {
          const response = await invoiceService.getInvoiceDetails(invoiceId);
          invoiceData = response.data;
        } catch (err) {
          // If invoice endpoint fails with 404, try billing endpoint as fallback
          if (err.response && err.response.status === 404) {
            console.log("Invoice not found, trying billing endpoint instead");
            const billingResponse = await axios.get(`/api/billing/${invoiceId}/`, {
              headers: {
                Authorization: `Bearer ${localStorage.getItem('access_token')}`
              }
            });
            invoiceData = billingResponse.data;
            
            // Map billing record fields to invoice fields if needed
            if (!invoiceData.subscription_type && invoiceData.subscription) {
              invoiceData.subscription_type = invoiceData.subscription.plan;
            }
          } else {
            // If it's not a 404 or the billing endpoint also fails, throw the error
            throw err;
          }
        }
        
        // Ensure some properties exist
        if (!invoiceData.status) invoiceData.status = 'pending';
        if (!invoiceData.description) invoiceData.description = 'No description provided';
        
        // Handle subscription type vs plan naming inconsistency
        if (invoiceData.subscription_type && !invoiceData.subscription_plan) {
          console.log("Adding missing subscription_plan from subscription_type");
          invoiceData.subscription_plan = invoiceData.subscription_type;
        }
        
        setInvoice(invoiceData);
        setLoading(false);
      } catch (err) {
        console.error('Error fetching invoice:', err);
        setError('Failed to load invoice details. Please try again later.');
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
      await invoiceService.payInvoice(invoiceId);
      
      // Update invoice data after payment
      const updatedInvoice = await invoiceService.getInvoiceDetails(invoiceId);
      
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
    <AnimatePresence>
      <motion.div
        key="modal-backdrop"
        variants={fadeIn}
        initial="hidden"
        animate="visible"
        exit="exit"
        className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      >
        {/* Modal content - stop propagation to prevent closing when clicking inside the modal */}
        <motion.div 
          key="modal-content"
          variants={modalVariant}
          initial="hidden"
          animate="visible"
          exit="exit"
          className="relative w-full max-w-4xl max-h-[90vh] overflow-y-auto rounded-xl"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Close button */}
          <motion.button
            whileHover={{ scale: 1.1, backgroundColor: isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)' }}
            whileTap={{ scale: 0.95 }}
            onClick={onClose}
            className={`absolute top-4 right-4 z-10 p-2 rounded-full ${
              isDarkMode 
                ? 'bg-gray-800/80 text-gray-300 hover:text-white backdrop-blur-sm' 
                : 'bg-white/80 text-gray-600 hover:text-gray-900 backdrop-blur-sm'
            } shadow-md`}
            aria-label="Close modal"
          >
            <X className="w-5 h-5" />
          </motion.button>
          
          {loading ? (
            <div className={`flex flex-col items-center justify-center p-16 rounded-xl shadow-2xl ${
              isDarkMode ? 'bg-gray-800/90 backdrop-blur-md border border-white/5' : 'bg-white border border-gray-100'
            }`}>
              <LoadingSpinner size="lg" className={isDarkMode ? 'text-purple-400' : 'text-purple-600'} />
              <p className={`mt-4 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>Loading invoice details...</p>
            </div>
          ) : error ? (
            <div className={`p-8 rounded-xl shadow-2xl ${isDarkMode ? 'bg-gray-800/90 backdrop-blur-md border border-white/5 text-white' : 'bg-white border border-gray-100 text-gray-900'}`}>
              <div className="text-center">
                <AlertCircle className="w-16 h-16 mx-auto text-red-500 mb-4" />
                <h2 className="text-2xl font-bold mb-2">Error Loading Invoice</h2>
                <p className="mb-6">{error}</p>
                <motion.button 
                  whileHover={{ scale: 1.05, y: -2 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={onClose}
                  className={`px-5 py-2.5 rounded-lg font-medium ${
                    isDarkMode ? 'bg-gray-700 hover:bg-gray-600 text-white' : 'bg-gray-200 hover:bg-gray-300 text-gray-800'
                  } shadow-md`}
                >
                  Close
                </motion.button>
              </div>
            </div>
          ) : !invoice ? (
            <div className={`p-8 rounded-xl shadow-2xl ${isDarkMode ? 'bg-gray-800/90 backdrop-blur-md border border-white/5 text-white' : 'bg-white border border-gray-100 text-gray-900'}`}>
              <div className="text-center">
                <AlertCircle className="w-16 h-16 mx-auto text-amber-500 mb-4" />
                <h2 className="text-2xl font-bold mb-2">Invoice Not Found</h2>
                <p className="mb-6">The requested invoice could not be found or you don't have permission to view it.</p>
                <motion.button 
                  whileHover={{ scale: 1.05, y: -2 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={onClose} 
                  className={`px-5 py-2.5 rounded-lg font-medium ${
                    isDarkMode ? 'bg-gray-700 hover:bg-gray-600 text-white' : 'bg-gray-200 hover:bg-gray-300 text-gray-800'
                  } shadow-md`}
                >
                  Close
                </motion.button>
              </div>
            </div>
          ) : (
            <div 
              id="invoice-content"
              className={`rounded-xl shadow-2xl overflow-hidden ${
                isDarkMode ? 'bg-gray-800/90 backdrop-blur-md border border-white/5 text-white' : 'bg-white border border-gray-100 text-gray-900'
              }`}
            >
              {/* Header */}
              <div className={`p-6 border-b ${isDarkMode ? 'border-white/10' : 'border-gray-200'}`}>
                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                  <div>
                    <h2 className="text-2xl font-light flex items-center">
                      Invoice <span className={isDarkMode ? 'text-purple-400 font-medium ml-2' : 'text-purple-600 font-medium ml-2'}>#{invoice.id}</span>
                    </h2>
                    <div className="flex items-center mt-2">
                      <Calendar className={`w-4 h-4 mr-2 ${isDarkMode ? 'text-purple-400/70' : 'text-purple-600/70'}`} />
                      <span className={isDarkMode ? 'text-gray-300' : 'text-gray-600'}>
                        {formatDate(invoice.invoice_date)}
                      </span>
                    </div>
                  </div>
                  
                  <div className="flex items-center">
                    <span className={`px-3 py-1.5 rounded-full text-sm font-medium flex items-center shadow-sm ${getStatusBadgeStyle(invoice.status)}`}>
                      <StatusIcon status={invoice.status} />
                      {invoice.status ? invoice.status.charAt(0).toUpperCase() + invoice.status.slice(1) : 'Unknown'}
                    </span>
                  </div>
                </div>
              </div>
              
              {/* Customer and Invoice Details */}
              <div className={`p-6 ${isDarkMode ? 'bg-black/20' : 'bg-gray-50/80'}`}>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Customer Information */}
                  <div className={`p-5 rounded-lg ${isDarkMode ? 'bg-gray-800/50 border border-white/5' : 'bg-white border border-gray-200/50'} shadow-md`}>
                    <h3 className={`text-sm font-medium mb-3 uppercase tracking-wider ${isDarkMode ? 'text-purple-400/70' : 'text-purple-600/70'}`}>
                      Customer
                    </h3>
                    <div className="space-y-3">
                      <div className="flex items-start space-x-3">
                        <div className={`p-1.5 rounded-full ${isDarkMode ? 'bg-purple-500/20' : 'bg-purple-100'}`}>
                          <Mail className={`w-4 h-4 ${isDarkMode ? 'text-purple-400' : 'text-purple-600'}`} />
                        </div>
                        <div>
                          <p className="font-medium text-lg">{invoice.username}</p>
                          <p className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>
                            {invoice.user_email || 'Email not available'}
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  {/* Invoice Details */}
                  <div className={`p-5 rounded-lg ${isDarkMode ? 'bg-gray-800/50 border border-white/5' : 'bg-white border border-gray-200/50'} shadow-md`}>
                    <h3 className={`text-sm font-medium mb-3 uppercase tracking-wider ${isDarkMode ? 'text-purple-400/70' : 'text-purple-600/70'}`}>
                      Invoice Details
                    </h3>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Invoice Number:</span>
                        <span className="font-medium">{invoice.id}</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Due Date:</span>
                        <span className={`font-medium ${
                          invoice.status === 'overdue' ? (isDarkMode ? 'text-red-400' : 'text-red-600') : ''
                        }`}>
                          {formatDate(invoice.due_date)}
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Subscription Plan:</span>
                        <span className={`font-medium capitalize ${isDarkMode ? 'text-purple-400' : 'text-purple-600'}`}>
                          {invoice.subscription_plan || invoice.subscription_type || 
                           (invoice.subscription && invoice.subscription.plan) || 'Standard Plan'}
                        </span>
                      </div>
                      {invoice.status === 'paid' && invoice.payment_date && (
                        <div className="flex justify-between items-center">
                          <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Payment Date:</span>
                          <span className={`font-medium ${isDarkMode ? 'text-green-400' : 'text-green-600'}`}>
                            {formatDate(invoice.payment_date)}
                          </span>
                        </div>
                      )}
                      {invoice.status === 'overdue' && (
                        <div className="flex justify-between items-center">
                          <span className={isDarkMode ? 'text-gray-400' : 'text-gray-600'}>Days Overdue:</span>
                          <span className={`font-medium ${isDarkMode ? 'text-red-400' : 'text-red-600'}`}>
                            {invoice.days_overdue}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Invoice Items */}
              <div className="p-6">
                <h3 className={`text-sm font-medium mb-4 uppercase tracking-wider ${isDarkMode ? 'text-purple-400/70' : 'text-purple-600/70'}`}>
                  Invoice Items
                </h3>
                <div className={`rounded-lg overflow-hidden border shadow-md ${isDarkMode ? 'border-white/10' : 'border-gray-200'}`}>
                  {/* Table Header */}
                  <div className={`grid grid-cols-12 gap-4 px-6 py-4 ${
                    isDarkMode ? 'bg-black/30 text-gray-300' : 'bg-gray-50 text-gray-600'
                  } text-sm font-medium`}>
                    <div className="col-span-8 flex items-center space-x-1">
                      <span>Description</span>
                      <ArrowDown className="w-3 h-3 opacity-50" />
                    </div>
                    <div className="col-span-4 text-right flex items-center justify-end space-x-1">
                      <span>Amount</span>
                      <ArrowDown className="w-3 h-3 opacity-50" />
                    </div>
                  </div>
                  
                  {/* Table Body */}
                  <div className={`px-6 py-5 ${isDarkMode ? 'bg-gray-800/50' : 'bg-white'}`}>
                    <div className="grid grid-cols-12 gap-4">
                      <div className="col-span-8">
                        {/* Use our helper function to get consistent subscription details */}
                        {(() => {
                          const subscriptionDetails = getSubscriptionDetails(invoice);
                          return (
                            <>
                              <p className="font-medium capitalize">
                                {subscriptionDetails.planName} Subscription
                              </p>
                              <p className={`text-sm mt-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
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
                  <div className={`grid grid-cols-12 gap-4 px-6 py-4 border-t ${
                    isDarkMode ? 'bg-black/30 border-white/10' : 'bg-gray-50 border-gray-200'
                  }`}>
                    <div className="col-span-8 text-right font-medium">Total</div>
                    <div className="col-span-4 text-right font-medium text-lg">
                      <span className={isDarkMode ? 'text-purple-400' : 'text-purple-600'}>
                        {formatCurrency(invoice.amount)}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Payment Section - Only shown for pending invoices and non-admin users */}
              {invoice.status === 'pending' && !isAdminView && (
                <div className={`p-6 border-t ${isDarkMode ? 'border-white/10' : 'border-gray-200'}`}>
                  <h3 className={`text-sm font-medium mb-4 uppercase tracking-wider ${isDarkMode ? 'text-purple-400/70' : 'text-purple-600/70'}`}>
                    Payment
                  </h3>
                  {paymentSuccess && (
                    <motion.div 
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className={`mb-4 p-4 rounded-lg flex items-start ${
                        isDarkMode ? 'bg-green-500/20 text-green-400 border border-green-700/50' : 'bg-green-100 text-green-700 border border-green-200'
                      }`}
                    >
                      <CheckCircle className="w-5 h-5 mr-3 flex-shrink-0 mt-0.5" />
                      <div>
                        <p className="font-medium">Payment Successful!</p>
                        <p className="text-sm mt-1">Your invoice has been paid and your subscription is active.</p>
                      </div>
                    </motion.div>
                  )}
                  
                  {paymentError && (
                    <motion.div 
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className={`mb-4 p-4 rounded-lg flex items-start ${
                        isDarkMode ? 'bg-red-500/20 text-red-400 border border-red-700/50' : 'bg-red-100 text-red-700 border border-red-200'
                      }`}
                    >
                      <AlertCircle className="w-5 h-5 mr-3 flex-shrink-0 mt-0.5" />
                      <div>
                        <p className="font-medium">Payment Error</p>
                        <p className="text-sm mt-1">{paymentError}</p>
                      </div>
                    </motion.div>
                  )}
                  
                  <motion.button
                    onClick={handlePayInvoice}
                    disabled={paymentProcessing || paymentSuccess}
                    whileHover={{ scale: 1.02, y: -2 }}
                    whileTap={{ scale: 0.98 }}
                    className={`w-full py-3.5 px-4 rounded-lg font-medium flex items-center justify-center shadow-md ${
                      paymentProcessing || paymentSuccess
                        ? isDarkMode ? 'bg-gray-700 text-gray-400 cursor-not-allowed' : 'bg-gray-200 text-gray-500 cursor-not-allowed'
                        : isDarkMode ? 'bg-purple-600 hover:bg-purple-700 text-white' : 'bg-purple-600 hover:bg-purple-700 text-white'
                    }`}
                  >
                    {paymentProcessing ? (
                      <>
                        <LoadingSpinner size="sm" className="mr-2 text-white" />
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
              <div className={`p-4 flex flex-wrap gap-3 border-t ${
                isDarkMode ? 'border-white/10 bg-black/20' : 'border-gray-200 bg-gray-50/80'
              }`}>
                <div className="flex-grow flex flex-wrap gap-3">
                  <motion.button
                    whileHover={{ scale: 1.05, backgroundColor: isDarkMode ? 'rgba(107, 70, 193, 0.2)' : 'rgba(107, 70, 193, 0.1)' }}
                    whileTap={{ scale: 0.95 }}
                    onClick={(e) => {
                      e.stopPropagation();
                      handlePrintInvoice();
                    }}
                    className={`px-4 py-2 rounded-lg flex items-center text-sm font-medium ${
                      isDarkMode
                        ? 'bg-gray-800/80 hover:bg-gray-700 text-white shadow-md backdrop-blur-sm'
                        : 'bg-white hover:bg-gray-100 text-gray-800 shadow-md backdrop-blur-sm'
                    }`}
                  >
                    <Printer className={`w-4 h-4 mr-2 ${isDarkMode ? 'text-purple-400' : 'text-purple-600'}`} />
                    Print
                  </motion.button>
                  
                  <motion.button
                    whileHover={{ scale: 1.05, backgroundColor: isDarkMode ? 'rgba(107, 70, 193, 0.2)' : 'rgba(107, 70, 193, 0.1)' }}
                    whileTap={{ scale: 0.95 }}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDownloadInvoice();
                    }}
                    className={`px-4 py-2 rounded-lg flex items-center text-sm font-medium ${
                      isDarkMode
                        ? 'bg-gray-800/80 hover:bg-gray-700 text-white shadow-md backdrop-blur-sm'
                        : 'bg-white hover:bg-gray-100 text-gray-800 shadow-md backdrop-blur-sm'
                    }`}
                  >
                    <Download className={`w-4 h-4 mr-2 ${isDarkMode ? 'text-purple-400' : 'text-purple-600'}`} />
                    Download
                  </motion.button>
                  
                  <motion.button
                    whileHover={{ scale: 1.05, backgroundColor: isDarkMode ? 'rgba(107, 70, 193, 0.2)' : 'rgba(107, 70, 193, 0.1)' }}
                    whileTap={{ scale: 0.95 }}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleShareInvoice();
                    }}
                    className={`px-4 py-2 rounded-lg flex items-center text-sm font-medium ${
                      isDarkMode
                        ? 'bg-gray-800/80 hover:bg-gray-700 text-white shadow-md backdrop-blur-sm'
                        : 'bg-white hover:bg-gray-100 text-gray-800 shadow-md backdrop-blur-sm'
                    }`}
                  >
                    <Share2 className={`w-4 h-4 mr-2 ${isDarkMode ? 'text-purple-400' : 'text-purple-600'}`} />
                    Share
                  </motion.button>
                </div>

                <motion.button
                  whileHover={{ scale: 1.05, backgroundColor: isDarkMode ? 'rgba(239, 68, 68, 0.2)' : 'rgba(239, 68, 68, 0.1)' }}
                  whileTap={{ scale: 0.95 }}
                  onClick={onClose}
                  className={`px-4 py-2 rounded-lg flex items-center text-sm font-medium ${
                    isDarkMode
                      ? 'bg-gray-800/80 hover:bg-gray-700 text-white shadow-md backdrop-blur-sm'
                      : 'bg-white hover:bg-gray-100 text-gray-800 shadow-md backdrop-blur-sm'
                  }`}
                >
                  <X className={`w-4 h-4 mr-2 ${isDarkMode ? 'text-red-400' : 'text-red-600'}`} />
                  Close
                </motion.button>
              </div>
            </div>
          )}
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default InvoiceDetailModal;