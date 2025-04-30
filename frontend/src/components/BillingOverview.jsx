import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import { Link } from 'react-router-dom';
import { 
  AlertCircle, RefreshCw, Calendar, DollarSign, 
  PieChart, Filter, Download, ArrowDownRight, 
  ArrowUpRight, CheckCircle, Clock, XCircle, FileText,
  User, Search, X, CreditCard, Cpu, Layers, Database,
  Zap, Shield, Award, Gem, ChevronUp, ChevronDown
} from 'lucide-react';
import InvoiceDetailModal from './InvoiceDetailModal';

const BillingOverview = ({ isDarkMode }) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [userBillingData, setUserBillingData] = useState([]);
  const [summary, setSummary] = useState(null);
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [subscriptionFilter, setSubscriptionFilter] = useState('');
  const [isFilterExpanded, setIsFilterExpanded] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  
  // Modal state
  const [selectedInvoiceId, setSelectedInvoiceId] = useState(null);
  const [isInvoiceModalOpen, setIsInvoiceModalOpen] = useState(false);
  
  // Subscription related state
  const [currentSubscription, setCurrentSubscription] = useState(null);
  const [subscriptionLoading, setSubscriptionLoading] = useState(true);
  const [subscriptionError, setSubscriptionError] = useState(null);
  const [selectedPlan, setSelectedPlan] = useState(null);
  const [isSubscribing, setIsSubscribing] = useState(false);
  const [showConfirmDialog, setShowConfirmDialog] = useState(false);
  const [subscriptionSuccess, setSubscriptionSuccess] = useState(null);

  // Plans data with features and pricing
  const plans = [
    {
      id: 'free',
      name: 'Free',
      price: 0,
      icon: <Layers />,
      color: 'gray',
      bgClass: isDarkMode ? 'bg-gray-900/30' : 'bg-gray-50',
      borderClass: isDarkMode ? 'border-gray-700' : 'border-gray-200',
      iconBgClass: isDarkMode ? 'bg-gray-900/30' : 'bg-gray-100',
      iconTextClass: isDarkMode ? 'text-gray-400' : 'text-gray-600',
      buttonClass: isDarkMode ? 'bg-gray-600 hover:bg-gray-700' : 'bg-gray-600 hover:bg-gray-700',
      features: [
        '100 API calls per month',
        '1GB data usage',
        'Basic analytics',
        'Email support'
      ]
    },
    {
      id: 'basic',
      name: 'Basic',
      price: 9.99,
      icon: <Cpu />,
      color: 'blue',
      bgClass: isDarkMode ? 'bg-blue-900/30' : 'bg-blue-50',
      borderClass: isDarkMode ? 'border-blue-700' : 'border-blue-200',
      iconBgClass: isDarkMode ? 'bg-blue-900/30' : 'bg-blue-100',
      iconTextClass: isDarkMode ? 'text-blue-400' : 'text-blue-600',
      buttonClass: isDarkMode ? 'bg-blue-600 hover:bg-blue-700' : 'bg-blue-600 hover:bg-blue-700',
      features: [
        '1,000 API calls per month',
        '10GB data usage',
        'Standard analytics',
        'Priority email support',
        'Access to basic AI features'
      ]
    },
    {
      id: 'premium',
      name: 'Premium',
      price: 29.99,
      icon: <Gem />,
      color: 'purple',
      bgClass: isDarkMode ? 'bg-purple-900/30' : 'bg-purple-50',
      borderClass: isDarkMode ? 'border-purple-700' : 'border-purple-200',
      iconBgClass: isDarkMode ? 'bg-purple-900/30' : 'bg-purple-100',
      iconTextClass: isDarkMode ? 'text-purple-400' : 'text-purple-600',
      buttonClass: isDarkMode ? 'bg-purple-600 hover:bg-purple-700' : 'bg-purple-600 hover:bg-purple-700',
      features: [
        '10,000 API calls per month',
        '100GB data usage',
        'Advanced analytics',
        'Phone support',
        'Access to all AI features',
        'Custom training data'
      ]
    },
    {
      id: 'enterprise',
      name: 'Enterprise',
      price: 99.99,
      icon: <Award />,
      color: 'indigo',
      bgClass: isDarkMode ? 'bg-indigo-900/30' : 'bg-indigo-50',
      borderClass: isDarkMode ? 'border-indigo-700' : 'border-indigo-200',
      iconBgClass: isDarkMode ? 'bg-indigo-900/30' : 'bg-indigo-100',
      iconTextClass: isDarkMode ? 'text-indigo-400' : 'text-indigo-600',
      buttonClass: isDarkMode ? 'bg-indigo-600 hover:bg-indigo-700' : 'bg-indigo-600 hover:bg-indigo-700',
      features: [
        '100,000 API calls per month',
        '1TB data usage',
        'Enterprise analytics',
        'Dedicated support',
        'Access to all AI features',
        'Custom training data',
        'SLA guarantees',
        'Custom implementation'
      ]
    }
  ];
  
  // Fetch billing data
  const fetchBillingData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Build the query string with filters
      let query = [];
      if (startDate) query.push(`start_date=${startDate}`);
      if (endDate) query.push(`end_date=${endDate}`);
      if (statusFilter) query.push(`status=${statusFilter}`);
      if (subscriptionFilter) query.push(`subscription=${subscriptionFilter}`);
      
      const queryString = query.length > 0 ? `?${query.join('&')}` : '';
      
      // Use the my_invoices endpoint to get only the current user's invoices
      const response = await axios.get(`http://localhost:8000/api/invoices/my_invoices/${queryString}`, {
        headers: { Authorization: `Bearer ${localStorage.getItem('access_token')}` }
      });
      
      console.log("User invoice data:", response.data);
      
      // Process the invoice data to ensure all required fields are present
      let processedData = [];
      if (response.data.invoices && Array.isArray(response.data.invoices)) {
        processedData = response.data.invoices.map(invoice => {
          // Look up subscription plan from subscription_plan field first,
          // then from subscription.plan, or fall back to free
          const subscriptionType = 
            invoice.subscription_plan || 
            (invoice.subscription && invoice.subscription.plan) || 
            'free';
            
          return {
            ...invoice,
            amount: invoice.amount !== undefined ? invoice.amount : 0,
            subscription_type: subscriptionType,
            status: invoice.status || 'pending',
            billing_date: invoice.invoice_date || new Date().toISOString().split('T')[0],
            due_date: invoice.due_date || null
          };
        });
      }
      
      console.log("Processed user invoice data:", processedData);
      
      // Set the processed data
      setUserBillingData(processedData);
      setSummary(response.data.summary || {});
    } catch (err) {
      console.error("Error fetching user billing data:", err);
      setError("Failed to fetch your billing data. Please try again.");
    } finally {
      setLoading(false);
    }
  };
  
  // Fetch the current user's subscription
  const fetchCurrentSubscription = async () => {
    try {
      setSubscriptionLoading(true);
      setSubscriptionError(null);
      
      const response = await axios.get('http://localhost:8000/api/subscriptions/my_subscription/', {
        headers: { Authorization: `Bearer ${localStorage.getItem('access_token')}` }
      });
      
      console.log("Current subscription:", response.data);
      setCurrentSubscription(response.data);
      
    } catch (err) {
      console.error("Error fetching subscription:", err);
      // If 404, user doesn't have an active subscription yet, which is okay
      if (err.response && err.response.status === 404) {
        setCurrentSubscription({ plan: 'free' }); // Default to free plan
      } else {
        setSubscriptionError("Failed to load subscription information.");
      }
    } finally {
      setSubscriptionLoading(false);
    }
  };
  
  // Handle plan selection
  const handleSelectPlan = (planId) => {
    setSelectedPlan(planId);
  };
  
  // Handle subscription
  const handleSubscribe = async () => {
    if (!selectedPlan) return;
    
    try {
      setIsSubscribing(true);
      setSubscriptionError(null);
      setSubscriptionSuccess(null);
      
      const response = await axios.post('http://localhost:8000/api/subscriptions/subscribe/', 
        { plan: selectedPlan, auto_renew: true },
        { headers: { Authorization: `Bearer ${localStorage.getItem('access_token')}` }}
      );
      
      console.log("Subscription response:", response.data);
      
      // Update the current subscription
      setCurrentSubscription(response.data.subscription);
      
      // Show success message
      setSubscriptionSuccess({
        message: `Successfully subscribed to ${selectedPlan} plan!`,
        invoiceId: response.data.invoice_id,
        amount: response.data.invoice_amount,
        dueDate: response.data.invoice_due_date
      });
      
      // Clear selected plan
      setSelectedPlan(null);
      
      // Close confirmation dialog
      setShowConfirmDialog(false);
      
      // Refresh the billing data to include the new invoice
      fetchBillingData();
      
    } catch (err) {
      console.error("Error subscribing to plan:", err);
      setSubscriptionError(err.response?.data?.detail || "Failed to process subscription. Please try again.");
    } finally {
      setIsSubscribing(false);
    }
  };
  
  // Load billing data and subscription on component mount
  useEffect(() => {
    fetchBillingData();
    fetchCurrentSubscription();
  }, []);
  
  // Handle filter application
  const applyFilters = () => {
    fetchBillingData();
    setIsFilterExpanded(false);
  };
  
  // Clear all filters
  const clearFilters = () => {
    setStartDate('');
    setEndDate('');
    setStatusFilter('');
    setSubscriptionFilter('');
    setSearchQuery('');
    fetchBillingData();
  };
  
  // Format currency
  const formatCurrency = (amount) => {
    if (amount === undefined || amount === null) {
      return "$0.00";
    }
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(amount);
  };
  
  // Format date
  const formatDate = (dateString) => {
    if (!dateString) return "N/A";
    
    const options = { year: 'numeric', month: 'short', day: 'numeric' };
    try {
      return new Date(dateString).toLocaleDateString('en-US', options);
    } catch (error) {
      console.error("Error formatting date:", error);
      return "Invalid Date";
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
  
  // Get status icon
  const getStatusIcon = (status) => {
    switch(status) {
      case 'paid':
        return <CheckCircle className="w-3 h-3 mr-1" />;
      case 'pending':
        return <Clock className="w-3 h-3 mr-1" />;
      case 'overdue':
        return <AlertCircle className="w-3 h-3 mr-1" />;
      case 'cancelled':
        return <XCircle className="w-3 h-3 mr-1" />;
      default:
        return <FileText className="w-3 h-3 mr-1" />;
    }
  };
  
  // Subscription badge color
  const getSubscriptionColor = (type) => {
    switch(type) {
      case 'free':
        return isDarkMode 
          ? { bg: 'bg-gray-500/20', text: 'text-gray-400' }
          : { bg: 'bg-gray-100', text: 'text-gray-700' };
      case 'basic':
        return isDarkMode 
          ? { bg: 'bg-blue-500/20', text: 'text-blue-400' }
          : { bg: 'bg-blue-100', text: 'text-blue-700' };
      case 'premium':
        return isDarkMode 
          ? { bg: 'bg-purple-500/20', text: 'text-purple-400' }
          : { bg: 'bg-purple-100', text: 'text-purple-700' };
      case 'enterprise':
        return isDarkMode 
          ? { bg: 'bg-indigo-500/20', text: 'text-indigo-400' }
          : { bg: 'bg-indigo-100', text: 'bg-indigo-700' };
      default:
        return isDarkMode 
          ? { bg: 'bg-gray-500/20', text: 'text-gray-400' }
          : { bg: 'bg-gray-100', text: 'text-gray-700' };
    }
  };
  
  // Calculate percentage change
  const calculateChange = () => {
    return {
      amount: 12.5,
      positive: true
    };
  };
  
  // Open invoice modal
  const handleOpenInvoiceModal = (invoiceId) => {
    setSelectedInvoiceId(invoiceId);
    setIsInvoiceModalOpen(true);
  };
  
  // Close invoice modal
  const handleCloseInvoiceModal = () => {
    setIsInvoiceModalOpen(false);
    setSelectedInvoiceId(null);
  };
  
  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-4 gap-4">
        <h2 className={`text-2xl font-light ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>
          Your <span className="text-purple-400 font-medium">Invoices & Billing</span>
        </h2>
        
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={fetchCurrentSubscription}
          className={`mt-2 sm:mt-0 ${
            isDarkMode 
              ? 'bg-gray-800 hover:bg-gray-700 text-white border border-white/10' 
              : 'bg-white hover:bg-gray-50 text-gray-700 border border-gray-200'
          } px-3 py-2 rounded-lg flex items-center space-x-2 shadow-sm`}
        >
          <RefreshCw className={`w-4 h-4 ${subscriptionLoading ? 'animate-spin' : ''}`} />
          <span className="text-sm font-medium">Refresh</span>
        </motion.button>
      </div>
      
      {error && (
        <div className={`p-4 rounded-lg mb-6 ${
          isDarkMode 
            ? 'bg-red-900/30 text-red-200 border border-red-800/50' 
            : 'bg-red-50 text-red-700 border border-red-200'
        }`}>
          <div className="flex items-center">
            <AlertCircle className="w-5 h-5 mr-2" />
            <p>{error}</p>
          </div>
        </div>
      )}
      
      {loading ? (
        <div className="flex justify-center items-center py-12">
          <RefreshCw className={`w-8 h-8 animate-spin ${
            isDarkMode ? 'text-purple-400' : 'text-indigo-500'
          }`} />
          <span className={`ml-3 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
            Loading your billing data...
          </span>
        </div>
      ) : (
        <>          
          {/* Subscription Section */}
          <div className="mb-8">

            {subscriptionError && (
              <div className={`p-4 rounded-lg mb-6 ${
                isDarkMode 
                  ? 'bg-red-900/30 text-red-200 border border-red-800/50' 
                  : 'bg-red-50 text-red-700 border border-red-200'
              }`}>
                <div className="flex items-center">
                  <AlertCircle className="w-5 h-5 mr-2" />
                  <p>{subscriptionError}</p>
                </div>
              </div>
            )}
            
            {subscriptionSuccess && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`p-4 rounded-lg mb-6 ${
                  isDarkMode 
                    ? 'bg-green-900/30 text-green-200 border border-green-800/50' 
                    : 'bg-green-50 text-green-700 border border-green-200'
                }`}
              >
                <div className="flex items-center">
                  <CheckCircle className="w-5 h-5 mr-2" />
                  <div>
                    <p className="font-medium">{subscriptionSuccess.message}</p>
                    <p className="text-sm mt-1">
                      Invoice #{subscriptionSuccess.invoiceId} for {formatCurrency(subscriptionSuccess.amount)} 
                      has been generated. Due date: {formatDate(subscriptionSuccess.dueDate)}.
                    </p>
                  </div>
                </div>
              </motion.div>
            )}
            
            {subscriptionLoading ? (
              <div className="flex justify-center items-center py-8">
                <RefreshCw className={`w-6 h-6 animate-spin ${
                  isDarkMode ? 'text-purple-400' : 'text-indigo-500'
                }`} />
                <span className={`ml-3 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                  Loading subscription data...
                </span>
              </div>
            ) : (
              <>
                {/* Current Subscription */}
                {currentSubscription && (
                  <div className={`mb-6 p-5 rounded-lg border ${
                    isDarkMode 
                      ? 'bg-gray-800/50 border-white/10' 
                      : 'bg-white border-gray-200'
                  }`}>
                    <div className="flex flex-col md:flex-row justify-between md:items-center gap-4">
                      <div>
                        <div className={`text-sm ${
                          isDarkMode ? 'text-gray-400' : 'text-gray-500'
                        }`}>Current Plan</div>
                        <div className="flex items-center mt-1">
                          <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium mr-2
                            ${getSubscriptionColor(currentSubscription.plan).bg} 
                            ${getSubscriptionColor(currentSubscription.plan).text}`}
                          >
                            {currentSubscription.plan?.charAt(0).toUpperCase() + currentSubscription.plan?.slice(1) || 'Free'}
                          </span>
                          <span className={`text-lg font-medium ${
                            isDarkMode ? 'text-white' : 'text-gray-800'
                          }`}>
                            {formatCurrency(currentSubscription.price || 0)}<span className="text-sm font-normal">/month</span>
                          </span>
                        </div>
                      </div>
                      
                      <div className="flex flex-col sm:items-end">
                        <div className={`text-sm ${
                          isDarkMode ? 'text-gray-400' : 'text-gray-500'
                        }`}>
                          {currentSubscription.auto_renew ? 'Auto-renews' : 'Expires'} on
                        </div>
                        <div className={`mt-1 ${
                          isDarkMode ? 'text-white' : 'text-gray-800'
                        }`}>
                          {formatDate(currentSubscription.end_date)}
                        </div>
                      </div>
                      
                      <div className="flex flex-col sm:items-end">
                        <div className={`text-sm ${
                          isDarkMode ? 'text-gray-400' : 'text-gray-500'
                        }`}>API Usage</div>
                        <div className={`mt-1 ${
                          isDarkMode ? 'text-white' : 'text-gray-800'
                        }`}>
                          {currentSubscription.api_calls_used || 0} / {currentSubscription.max_api_calls || 100}
                        </div>
                      </div>
                      
                      <div className="flex flex-col sm:items-end">
                        <div className={`text-sm ${
                          isDarkMode ? 'text-gray-400' : 'text-gray-500'
                        }`}>Data Usage</div>
                        <div className={`mt-1 ${
                          isDarkMode ? 'text-white' : 'text-gray-800'
                        }`}>
                          {currentSubscription.data_usage || 0} MB / {currentSubscription.max_data_usage || 1000} MB
                        </div>
                      </div>
                    </div>
                  </div>
                )}
                
                {/* Subscription Plans */}
                <div className="mb-6">
                  <h4 className={`text-lg font-medium mb-4 ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>
                    Available Plans
                  </h4>
                  
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    {plans.map((plan) => (
                      <motion.div
                        key={plan.id}
                        whileHover={{ y: -4 }}
                        className={`p-5 rounded-lg border ${
                          selectedPlan === plan.id
                            ? `${plan.bgClass} ${plan.borderClass}`
                            : isDarkMode
                              ? 'bg-gray-800 border-white/10 hover:border-gray-600' 
                              : 'bg-white border-gray-200 hover:border-gray-300'
                        } transition-all duration-300 cursor-pointer`}
                        onClick={() => handleSelectPlan(plan.id)}
                      >
                        <div className={`p-2 w-10 h-10 rounded-lg flex items-center justify-center mb-3 ${plan.iconBgClass} ${plan.iconTextClass}`}>
                          {plan.icon}
                        </div>
                        
                        <h5 className={`text-lg font-medium mb-1 ${
                          isDarkMode ? 'text-white' : 'text-gray-800'
                        }`}>
                          {plan.name}
                        </h5>
                        
                        <div className={`mb-4 ${
                          isDarkMode ? 'text-white' : 'text-gray-800'
                        }`}>
                          <span className="text-2xl font-bold">${plan.price}</span>
                          <span className={`${
                            isDarkMode ? 'text-gray-400' : 'text-gray-500'
                          }`}>/month</span>
                        </div>
                        
                        <ul className={`space-y-2 mb-4 text-sm ${
                          isDarkMode ? 'text-gray-300' : 'text-gray-600'
                        }`}>
                          {plan.features.map((feature, i) => (
                            <li key={i} className="flex items-start">
                              <CheckCircle className={`w-4 h-4 mt-0.5 mr-2 ${plan.iconTextClass}`} />
                              {feature}
                            </li>
                          ))}
                        </ul>
                        
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            if (currentSubscription && currentSubscription.plan === plan.id) {
                              // Already subscribed to this plan
                              return;
                            }
                            setSelectedPlan(plan.id);
                            setShowConfirmDialog(true);
                          }}
                          className={`w-full py-2 rounded-lg text-center text-sm font-medium ${
                            currentSubscription && currentSubscription.plan === plan.id
                              ? isDarkMode
                                ? 'bg-gray-700 text-gray-300 cursor-not-allowed'
                                : 'bg-gray-100 text-gray-500 cursor-not-allowed'
                              : `${plan.buttonClass} text-white`
                          }`}
                          disabled={currentSubscription && currentSubscription.plan === plan.id}
                        >
                          {currentSubscription && currentSubscription.plan === plan.id ? 'Current Plan' : 'Select Plan'}
                        </button>
                      </motion.div>
                    ))}
                  </div>
                </div>
                
                {/* Confirmation Dialog */}
                <AnimatePresence>
                  {showConfirmDialog && selectedPlan && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="fixed inset-0 flex items-center justify-center z-50 bg-black/50 p-4"
                    >
                      <motion.div
                        initial={{ scale: 0.9, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        exit={{ scale: 0.9, opacity: 0 }}
                        className={`rounded-lg shadow-xl max-w-md w-full ${
                          isDarkMode ? 'bg-gray-800 border border-white/10' : 'bg-white'
                        }`}
                      >
                        <div className="p-6">
                          <h3 className={`text-xl font-medium mb-4 ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>
                            Confirm Subscription
                          </h3>
                          <p className={`mb-6 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                            Are you sure you want to subscribe to the {plans.find(p => p.id === selectedPlan)?.name} plan?
                            {selectedPlan !== 'free' && ' You will be billed ' + 
                              formatCurrency(plans.find(p => p.id === selectedPlan)?.price) + ' monthly.'}
                          </p>
                          
                          <div className="flex justify-end space-x-3">
                            <button
                              onClick={() => setShowConfirmDialog(false)}
                              className={`px-4 py-2 rounded-lg ${
                                isDarkMode 
                                  ? 'bg-gray-700 hover:bg-gray-600 text-white' 
                                  : 'bg-gray-100 hover:bg-gray-200 text-gray-800'
                              }`}
                            >
                              Cancel
                            </button>
                            <button
                              onClick={handleSubscribe}
                              disabled={isSubscribing}
                              className={`px-4 py-2 rounded-lg ${
                                isDarkMode 
                                  ? 'bg-purple-600 hover:bg-purple-700 text-white' 
                                  : 'bg-indigo-600 hover:bg-indigo-700 text-white'
                              } flex items-center`}
                            >
                              {isSubscribing ? (
                                <>
                                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                                  Processing...
                                </>
                              ) : 'Confirm Subscription'}
                            </button>
                          </div>
                        </div>
                      </motion.div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </>
            )}
          </div>
          
          {/* Billing Table */}
          <div className={`${
            isDarkMode 
              ? 'bg-gray-800 border border-white/10' 
              : 'bg-white border border-gray-200'
          } rounded-xl overflow-hidden shadow-lg`}>
            <div className={`grid grid-cols-12 gap-4 px-6 py-4 border-b ${
              isDarkMode ? 'border-white/5 bg-black/20' : 'border-gray-200 bg-gray-50'
            }`}>
              <div className="col-span-3 font-medium text-xs uppercase tracking-wider">
                <span className={isDarkMode ? 'text-purple-400' : 'text-indigo-500'}>Invoice</span>
              </div>
              <div className="col-span-2 font-medium text-xs uppercase tracking-wider">
                <span className={isDarkMode ? 'text-purple-400' : 'text-indigo-500'}>Amount</span>
              </div>
              <div className="col-span-2 font-medium text-xs uppercase tracking-wider hidden md:block">
                <span className={isDarkMode ? 'text-purple-400' : 'text-indigo-500'}>Status</span>
              </div>
              <div className="col-span-2 font-medium text-xs uppercase tracking-wider hidden md:block">
                <span className={isDarkMode ? 'text-purple-400' : 'text-indigo-500'}>Subscription</span>
              </div>
              <div className="col-span-3 md:col-span-2 font-medium text-xs uppercase tracking-wider">
                <span className={isDarkMode ? 'text-purple-400' : 'text-indigo-500'}>Date</span>
              </div>
              <div className="col-span-1 font-medium text-xs uppercase tracking-wider text-right">
                <span className={isDarkMode ? 'text-purple-400' : 'text-indigo-500'}>Actions</span>
              </div>
            </div>
            
            {userBillingData.length === 0 ? (
              <div className={`py-16 text-center ${
                isDarkMode ? 'text-gray-400' : 'text-gray-500'
              }`}>
                <FileText className="w-12 h-12 mx-auto mb-4 opacity-20" />
                <p className="text-lg">No invoices found</p>
                <p className="text-sm mt-1 opacity-75">You don't have any billing history yet</p>
              </div>
            ) : (
              <div className={`divide-y ${isDarkMode ? 'divide-white/5' : 'divide-gray-200'}`}>
                {userBillingData.map((record) => (
                  <motion.div
                    key={record.id}
                    whileHover={{ 
                      backgroundColor: isDarkMode ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.02)'
                    }}
                    className="grid grid-cols-12 gap-4 px-6 py-4 items-center"
                  >
                    <div className="col-span-3 flex items-center space-x-3">
                      <div className={`${
                        isDarkMode ? 'bg-gray-700' : 'bg-gray-100'
                      } p-2 rounded-full`}>
                        <FileText className={`w-4 h-4 ${
                          isDarkMode ? 'text-gray-300' : 'text-gray-600'
                        }`} />
                      </div>
                      <div>
                        <div className={`font-medium truncate ${
                          isDarkMode ? 'text-white' : 'text-gray-800'
                        }`}>
                          Invoice #{record.id}
                        </div>
                        <div className={`text-xs truncate ${
                          isDarkMode ? 'text-gray-400' : 'text-gray-500'
                        }`}>
                          {record.description || "Billing period"}
                        </div>
                      </div>
                    </div>
                    
                    <div className={`col-span-2 font-medium ${
                      isDarkMode ? 'text-white' : 'text-gray-800'
                    }`}>
                      {formatCurrency(record.amount)}
                    </div>
                    
                    <div className="col-span-2 hidden md:block">
                      <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium
                        ${getStatusColor(record.status).bg} ${getStatusColor(record.status).text}`}
                      >
                        {getStatusIcon(record.status)}
                        {record.status.charAt(0).toUpperCase() + record.status.slice(1)}
                      </span>
                    </div>
                    
                    <div className="col-span-2 hidden md:block">
                      <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium
                        ${getSubscriptionColor(record.subscription_type).bg} 
                        ${getSubscriptionColor(record.subscription_type).text}`}
                      >
                        {record.subscription_type ? 
                          record.subscription_type.charAt(0).toUpperCase() + record.subscription_type.slice(1) : 
                          'Unknown'}
                      </span>
                    </div>
                    
                    <div className={`col-span-3 md:col-span-2 ${
                      isDarkMode ? 'text-gray-300' : 'text-gray-600'
                    }`}>
                      <div className="text-sm">
                        {formatDate(record.billing_date)}
                      </div>
                      <div className="text-xs text-gray-500">
                        Due: {formatDate(record.due_date)}
                      </div>
                    </div>
                    
                    <div className="col-span-1 flex justify-end">
                      <button
                        onClick={() => handleOpenInvoiceModal(record.id)}
                        className={`p-1 rounded-lg ${
                          isDarkMode 
                            ? 'hover:bg-gray-700 text-gray-400 hover:text-white' 
                            : 'hover:bg-gray-100 text-gray-500 hover:text-gray-800'
                        }`}
                        aria-label="View invoice details"
                      >
                        <FileText className="w-4 h-4" />
                      </button>
                    </div>
                  </motion.div>
                ))}
              </div>
            )}
          </div>
        </>
      )}
      
      {/* Invoice Detail Modal */}
      <AnimatePresence>
        {isInvoiceModalOpen && (
          <InvoiceDetailModal
            isDarkMode={isDarkMode}
            invoiceId={selectedInvoiceId}
            onClose={handleCloseInvoiceModal}
            onPaymentSuccess={() => {
              // Refresh billing data after successful payment
              fetchBillingData();
            }}
          />
        )}
      </AnimatePresence>
    </div>
  );
};

export default BillingOverview;