import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import {
  X, FileText, Calendar, CreditCard, CheckCircle, AlertCircle, Clock, XCircle,
  User, Building, DollarSign, Download, Printer, ExternalLink
} from 'lucide-react';

const InvoiceDetailModal = ({ isOpen, onClose, invoice, isDarkMode, onPaymentComplete }) => {
  const [isProcessingPayment, setIsProcessingPayment] = useState(false);
  const [paymentError, setPaymentError] = useState(null);
  const [paymentSuccess, setPaymentSuccess] = useState(false);

  if (!invoice) return null;
  
  // Log invoice data when modal is opened with an invoice
  console.log("=== INVOICE MODAL DEBUGGING ===");
  console.log("Modal invoice data:", invoice);
  console.log("Subscription plan:", invoice.subscription_plan);
  console.log("Subscription type:", invoice.subscription_type);
  console.log("Subscription name:", invoice.subscription_name);
  console.log("Subscription object:", invoice.subscription);
  console.log("Subscription start date:", invoice.subscription_start_date);
  console.log("Subscription end date:", invoice.subscription_end_date);
  console.log("==============================");

  // Format currency
  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(amount);
  };
  
  // Format date
  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    
    try {
      const options = { year: 'numeric', month: 'long', day: 'numeric' };
      const date = new Date(dateString);
      // Check if date is valid
      if (isNaN(date.getTime())) {
        return 'N/A';
      }
      return date.toLocaleDateString('en-US', options);
    } catch (error) {
      console.error('Date formatting error:', error);
      return 'N/A';
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

  // Get status icon
  const getStatusIcon = (status) => {
    switch(status) {
      case 'paid':
        return <CheckCircle className="w-4 h-4 mr-1" />;
      case 'pending':
        return <Clock className="w-4 h-4 mr-1" />;
      case 'overdue':
        return <AlertCircle className="w-4 h-4 mr-1" />;
      case 'cancelled':
        return <XCircle className="w-4 h-4 mr-1" />;
      default:
        return <FileText className="w-4 h-4 mr-1" />;
    }
  };
  
  // Status badge color
  const getStatusColor = (status) => {
    switch(status) {
      case 'paid':
        return isDarkMode 
          ? { bg: 'bg-green-500/20', text: 'text-green-400' }
          : { bg: 'bg-green-100', text: 'text-green-700' };
      case 'pending':
        return isDarkMode 
          ? { bg: 'bg-amber-500/20', text: 'text-amber-400' }
          : { bg: 'bg-amber-100', text: 'text-amber-700' };
      case 'overdue':
        return isDarkMode 
          ? { bg: 'bg-red-500/20', text: 'text-red-400' }
          : { bg: 'bg-red-100', text: 'text-red-700' };
      case 'cancelled':
        return isDarkMode 
          ? { bg: 'bg-gray-500/20', text: 'text-gray-400' }
          : { bg: 'bg-gray-100', text: 'text-gray-700' };
      default:
        return isDarkMode 
          ? { bg: 'bg-gray-500/20', text: 'text-gray-400' }
          : { bg: 'bg-gray-100', text: 'text-gray-700' };
    }
  };

  // Handle payment
  const handlePayment = async () => {
    if (invoice.status === 'paid' || invoice.status === 'cancelled') return;
    
    try {
      setIsProcessingPayment(true);
      setPaymentError(null);
      
      // Call the payment endpoint with the correct API path using the proxy
      const response = await axios.post(`/api/invoices/${invoice.id}/pay/`, {}, {
        headers: { Authorization: `Bearer ${localStorage.getItem('access_token')}` }
      });
      
      console.log("Payment response:", response.data);
      
      // Show success message
      setPaymentSuccess(true);
      
      // Notify parent component that payment was completed
      if (onPaymentComplete) {
        onPaymentComplete(invoice.id);
      }
      
    } catch (err) {
      console.error("Payment error:", err);
      setPaymentError(err.response?.data?.detail || "Payment processing failed. Please try again later.");
    } finally {
      setIsProcessingPayment(false);
    }
  };

  // Handle download invoice
  const handleDownloadInvoice = () => {
    // Implement download functionality using the API proxy path
    window.open(`/api/invoices/${invoice.id}/download/`, '_blank');
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
        >
          <motion.div
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.95, opacity: 0 }}
            className={`${
              isDarkMode 
                ? 'bg-gray-900 border border-gray-700' 
                : 'bg-white border border-gray-200'
            } w-full max-w-2xl rounded-xl shadow-xl overflow-hidden`}
          >
            {/* Header */}
            <div className={`flex justify-between items-center p-5 ${
              isDarkMode ? 'border-b border-gray-700' : 'border-b border-gray-200'
            }`}>
              <div className="flex items-center">
                <FileText className={`mr-2 ${isDarkMode ? 'text-purple-400' : 'text-indigo-600'}`} />
                <h2 className={`text-lg font-semibold ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>
                  Invoice #{invoice.invoice_number || invoice.id}
                </h2>
              </div>
              
              <button
                onClick={onClose}
                className={`rounded-full p-1 hover:bg-opacity-80 ${
                  isDarkMode 
                    ? 'hover:bg-gray-700 text-gray-400 hover:text-white' 
                    : 'hover:bg-gray-100 text-gray-500 hover:text-gray-700'
                }`}
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            {/* Body */}
            <div className={`p-5 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
              {/* Status and date info */}
              <div className="flex flex-wrap justify-between items-center mb-6">
                <div className="flex items-center mb-2 sm:mb-0">
                  <span className={`flex items-center px-3 py-1 rounded-full text-xs font-medium ${
                    getStatusColor(invoice.status).bg
                  } ${getStatusColor(invoice.status).text}`}>
                    {getStatusIcon(invoice.status)}
                    {invoice.status.charAt(0).toUpperCase() + invoice.status.slice(1)}
                  </span>
                </div>
                
                <div className="flex items-center space-x-4">
                  <div className="flex items-center text-sm">
                    <Calendar className="w-4 h-4 mr-1 opacity-70" />
                    <span>Issue Date: {formatDate(invoice.issue_date)}</span>
                  </div>
                  
                  <div className="flex items-center text-sm">
                    <Calendar className="w-4 h-4 mr-1 opacity-70" />
                    <span>Due Date: {formatDate(invoice.due_date)}</span>
                  </div>
                </div>
              </div>
              
              {/* Invoice details */}
              <div className={`rounded-lg mb-6 ${
                isDarkMode ? 'bg-gray-800/50' : 'bg-gray-50'
              } p-4`}>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {/* Customer info */}
                  <div>
                    <h3 className={`text-sm font-medium mb-2 ${
                      isDarkMode ? 'text-gray-400' : 'text-gray-500'
                    }`}>
                      Customer
                    </h3>
                    <div className="flex items-start">
                      <User className="w-4 h-4 mr-2 mt-0.5 opacity-70" />
                      <div>
                        <p className="font-medium">{invoice.customer_name}</p>
                        <p className="text-sm">{invoice.customer_email}</p>
                      </div>
                    </div>
                  </div>
                  
                  {/* Billing info */}
                  <div>
                    <h3 className={`text-sm font-medium mb-2 ${
                      isDarkMode ? 'text-gray-400' : 'text-gray-500'
                    }`}>
                      Billed By
                    </h3>
                    <div className="flex items-start">
                      <Building className="w-4 h-4 mr-2 mt-0.5 opacity-70" />
                      <div>
                        <p className="font-medium">DesD AI Pathway</p>
                        <p className="text-sm">support@desdaipathway.com</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Invoice items */}
              <div className="mb-6">
                <h3 className={`text-sm font-medium mb-3 ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-500'
                }`}>
                  Invoice Items
                </h3>
                
                <div className={`rounded-lg overflow-hidden border ${
                  isDarkMode ? 'border-gray-700' : 'border-gray-200'
                }`}>
                  <table className="w-full">
                    <thead className={`${
                      isDarkMode ? 'bg-gray-800' : 'bg-gray-50'
                    }`}>
                      <tr>
                        <th className={`px-4 py-2 text-left text-xs font-medium ${
                          isDarkMode ? 'text-gray-400' : 'text-gray-500'
                        }`}>Description</th>
                        <th className={`px-4 py-2 text-right text-xs font-medium ${
                          isDarkMode ? 'text-gray-400' : 'text-gray-500'
                        }`}>Amount</th>
                      </tr>
                    </thead>
                    <tbody className={`divide-y ${
                      isDarkMode ? 'divide-gray-700' : 'divide-gray-200'
                    }`}>
                      {invoice.items && invoice.items.map((item, index) => (
                        <tr key={index}>
                          <td className="px-4 py-3">{item.description}</td>
                          <td className="px-4 py-3 text-right">{formatCurrency(item.amount)}</td>
                        </tr>
                      ))}
                      {/* If invoice.items is not available, use a single row */}
                      {(!invoice.items || invoice.items.length === 0) && (
                        <tr>
                          <td className="px-4 py-3">
                            <div>
                              <p className="font-medium capitalize">
                                {invoice.subscription_plan || invoice.subscription_type || 
                                 (invoice.subscription && invoice.subscription.plan) || 'Standard'} Subscription
                              </p>
                              <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                                {invoice.description || `Monthly subscription fee (${formatDate(invoice.invoice_date)} - ${formatDate(invoice.due_date)})`}
                              </p>
                            </div>
                          </td>
                          <td className="px-4 py-3 text-right">{formatCurrency(invoice.amount)}</td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
              
              {/* Total */}
              <div className={`rounded-lg ${
                isDarkMode ? 'bg-gray-800/50' : 'bg-gray-50'
              } p-4 mb-6`}>
                <div className="flex justify-between items-center">
                  <span className="font-medium">Total:</span>
                  <span className={`text-lg font-bold ${
                    isDarkMode ? 'text-purple-400' : 'text-indigo-600'
                  }`}>
                    {formatCurrency(invoice.amount)}
                  </span>
                </div>
              </div>
              
              {/* Payment status message */}
              {paymentSuccess && (
                <div className={`p-4 rounded-lg mb-4 ${
                  isDarkMode 
                    ? 'bg-green-900/30 text-green-300 border border-green-800/50' 
                    : 'bg-green-50 text-green-700 border border-green-200'
                }`}>
                  <div className="flex items-center">
                    <CheckCircle className="w-5 h-5 mr-2" />
                    <p>Payment successful! Thank you for your payment.</p>
                  </div>
                </div>
              )}
              
              {paymentError && (
                <div className={`p-4 rounded-lg mb-4 ${
                  isDarkMode 
                    ? 'bg-red-900/30 text-red-300 border border-red-800/50' 
                    : 'bg-red-50 text-red-700 border border-red-200'
                }`}>
                  <div className="flex items-center">
                    <AlertCircle className="w-5 h-5 mr-2" />
                    <p>{paymentError}</p>
                  </div>
                </div>
              )}
            </div>
            
            {/* Footer with actions */}
            <div className={`p-5 flex justify-between ${
              isDarkMode ? 'border-t border-gray-700' : 'border-t border-gray-200'
            }`}>
              <div className="flex space-x-2">
                <button
                  onClick={handleDownloadInvoice}
                  className={`flex items-center space-x-1 px-3 py-2 rounded-lg ${
                    isDarkMode 
                      ? 'bg-gray-800 hover:bg-gray-700 text-gray-300' 
                      : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
                  }`}
                >
                  <Download className="w-4 h-4" />
                  <span>Download</span>
                </button>
                
                <button
                  onClick={() => window.print()}
                  className={`flex items-center space-x-1 px-3 py-2 rounded-lg ${
                    isDarkMode 
                      ? 'bg-gray-800 hover:bg-gray-700 text-gray-300' 
                      : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
                  }`}
                >
                  <Printer className="w-4 h-4" />
                  <span>Print</span>
                </button>
              </div>
              
              {(invoice.status === 'pending' || invoice.status === 'overdue') && (
                <button
                  onClick={handlePayment}
                  disabled={isProcessingPayment || paymentSuccess}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg ${
                    isDarkMode 
                      ? 'bg-purple-600 hover:bg-purple-700 text-white disabled:bg-purple-800/40 disabled:text-gray-400' 
                      : 'bg-indigo-600 hover:bg-indigo-700 text-white disabled:bg-indigo-300 disabled:text-gray-100'
                  }`}
                >
                  {isProcessingPayment ? (
                    <>
                      <span className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></span>
                      <span>Processing...</span>
                    </>
                  ) : (
                    <>
                      <CreditCard className="w-4 h-4" />
                      <span>{paymentSuccess ? 'Paid' : 'Pay Now'}</span>
                    </>
                  )}
                </button>
              )}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default InvoiceDetailModal;