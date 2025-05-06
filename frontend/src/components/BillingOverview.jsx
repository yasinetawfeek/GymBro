import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import { Link } from 'react-router-dom';
import { 
  AlertCircle, RefreshCw, Calendar, DollarSign, 
  PieChart, Filter, Download, ArrowDownRight, 
  ArrowUpRight, CheckCircle, Clock, XCircle, FileText,
  User, Search, X, CreditCard, Cpu, Layers, Database,
  Zap, Shield, Award, Gem, ChevronUp, ChevronDown,
  ReceiptIcon, AlertTriangle, ActivitySquare, BarChart3
} from 'lucide-react';
import InvoiceDetailModal from './InvoiceDetailModal';
import { useAuth } from '../context/AuthContext';

// Import the API services at the top of the file
import { invoiceService, subscriptionService, usageService } from '../services/apiService';

// Animation variants
const fadeIn = {
  hidden: { opacity: 0, y: 20 },
  visible: { 
    opacity: 1, 
    y: 0,
    transition: { duration: 0.4 }
  }
};

const staggerContainer = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1
    }
  }
};

const tableRowVariant = {
  hidden: { opacity: 0, x: -20 },
  visible: { 
    opacity: 1, 
    x: 0,
    transition: { duration: 0.3 }
  },
  exit: { 
    opacity: 0, 
    x: 20,
    transition: { duration: 0.2 }
  }
};

const cardVariant = {
  hidden: { opacity: 0, y: 20 },
  visible: { 
    opacity: 1, 
    y: 0,
    transition: { duration: 0.4 }
  },
  hover: {
    y: -5,
    transition: { duration: 0.2 }
  },
  tap: {
    y: 0,
    transition: { duration: 0.2 }
  }
};

const BillingOverview = ({ isDarkMode }) => {
  const { token } = useAuth();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [userBillingData, setUserBillingData] = useState([]);
  const [summary, setSummary] = useState(null);
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [subscriptionFilter, setSubscriptionFilter] = useState('all');
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

  // Replace the subscription plans data with WebSocket-based metrics
  const subscriptionPlans = [
    {
      id: 'free',
      name: 'Free',
      price: 0,
      features: [
        'Access to AI workout feedback',
        'Limited to 10 minutes/month',
        'Up to 1,000 corrections/month',
        'Single workout type',
        'Community support'
      ],
      description: 'For casual users trying out the platform',
      icon: <Gem className="w-10 h-10" />
    },
    {
      id: 'basic',
      name: 'Basic',
      price: 9.99,
      features: [
        'Access to AI workout feedback',
        'Up to 2 hours/month',
        'Up to 10,000 corrections/month',
        'Access to 5 workout types',
        'Email support'
      ],
      description: 'For regular fitness enthusiasts',
      icon: <Award className="w-10 h-10" />
    },
    {
      id: 'premium',
      name: 'Premium',
      price: 19.99,
      features: [
        'Unlimited time usage',
        'Unlimited corrections',
        'Access to all workout types',
        'Detailed usage analytics',
        'Priority support'
      ],
      description: 'For serious fitness enthusiasts',
      icon: <Shield className="w-10 h-10" />
    }
  ];
  
  // Add state for usage data
  const [usageData, setUsageData] = useState({
    total_sessions: 0,
    active_sessions: 0,
    total_duration_seconds: 0,
    total_frames_processed: 0,
    total_corrections_received: 0,
    total_billed_amount: 0
  });
  
  const [usageDashboardOpen, setUsageDashboardOpen] = useState(false);
  const [usageSessions, setUsageSessions] = useState([]);
  const [isLoadingUsage, setIsLoadingUsage] = useState(false);
  
  // Add the missing filter state variables
  const [dateFilter, setDateFilter] = useState('all');
  
  // Add this near the top of the component
  const [filteredInvoices, setFilteredInvoices] = useState([]);
  
  // Add notification state for subscription changes
  const [subscriptionNotification, setSubscriptionNotification] = useState(null);
  
  // Add this where state declarations are
  const [usageError, setUsageError] = useState(null);
  
  // Fetch billing data
  const fetchBillingData = async () => {
    setLoading(true);
    setError(null);
    try {
      // Use the new invoice service
      console.log('Attempting to fetch invoices data...');
      const response = await invoiceService.getMyInvoices();
      console.log('Raw invoice API response:', response);
      console.log('User invoice data (raw):', response.data);
      
      // Check if response.data is an array or if it has a results property
      let invoicesData = [];
      if (Array.isArray(response.data)) {
        console.log('Response data is an array');
        invoicesData = response.data;
      } else if (response.data && response.data.invoices && Array.isArray(response.data.invoices)) {
        // Access the invoices array in the response
        console.log('Response contains invoices array with', response.data.invoices.length, 'items');
        invoicesData = response.data.invoices;
      } else if (response.data && response.data.results && Array.isArray(response.data.results)) {
        console.log('Response data has results array');
        invoicesData = response.data.results;
      } else if (response.data && typeof response.data === 'object') {
        // If it's a single object, convert to array with that single item
        console.log('Response data is an object, converting to array');
        invoicesData = [response.data];
      } else {
        console.log('Response data format is unexpected, trying fallback endpoint');
        // Try fallback to original endpoint as a last resort
        const fallbackResponse = await axios.get('/api/billing/', {
          headers: {
            Authorization: `Bearer ${token}`
          }
        });
        console.log('Fallback response:', fallbackResponse.data);
        
        if (Array.isArray(fallbackResponse.data)) {
          invoicesData = fallbackResponse.data;
        } else if (fallbackResponse.data && fallbackResponse.data.invoices && Array.isArray(fallbackResponse.data.invoices)) {
          console.log('Fallback response contains invoices array with', fallbackResponse.data.invoices.length, 'items');
          invoicesData = fallbackResponse.data.invoices;
        } else if (fallbackResponse.data && fallbackResponse.data.results && Array.isArray(fallbackResponse.data.results)) {
          invoicesData = fallbackResponse.data.results;
        } else if (fallbackResponse.data && typeof fallbackResponse.data === 'object') {
          invoicesData = [fallbackResponse.data];
        }
      }
      
      // Process the data with careful checks
      if (invoicesData.length > 0) {
        console.log('Processing invoice data:', invoicesData);
        const processedData = invoicesData.map(record => ({
          id: record.id || 0,
          amount: record.amount || 0,
          subscription_type: record.subscription_type || record.subscription_plan || 'unknown',
          billing_date: record.invoice_date || record.billing_date || new Date().toISOString(),
          due_date: record.due_date || new Date().toISOString(),
          status: record.status || 'pending',
          description: record.description || '',
          api_calls: record.api_calls || 0,
          data_usage: record.data_usage || 0
        }));
        
        console.log('Processed user invoice data:', processedData);
        setUserBillingData(processedData);
        // Apply filters immediately
        applyFilters(processedData);
      } else {
        console.log('No invoice data found');
        setUserBillingData([]);
        setFilteredInvoices([]);
      }
    } catch (error) {
      console.error('Error fetching billing data:', error);
      console.error('Error details:', error.response || error.message);
      setError('Failed to load billing data. Please try again later.');
      // Set empty array to avoid undefined errors
      setUserBillingData([]);
      setFilteredInvoices([]);
    } finally {
      setLoading(false);
    }
  };
  
  // Fetch the current user's subscription
  const fetchCurrentSubscription = async () => {
    setSubscriptionLoading(true);
    try {
      console.log('Fetching current subscription...');
      const response = await subscriptionService.getCurrentSubscription();
      console.log('Raw subscription response:', response);
      console.log('Current subscription data:', response.data);
      
      if (response.data) {
        setCurrentSubscription(response.data);
      } else {
        console.error('Subscription response was empty');
        setCurrentSubscription(null);
      }
    } catch (error) {
      console.error('Error fetching subscription:', error);
      setCurrentSubscription(null);
    } finally {
      setSubscriptionLoading(false);
    }
  };
  
  // Handle refresh all
  const handleRefreshAll = () => {
    console.log('Refreshing all data...');
    fetchCurrentSubscription();
    fetchBillingData();
    fetchUsageData();
  };
  
  // Handle plan selection
  const handleSelectPlan = (planId) => {
    setSelectedPlan(planId);
    setShowConfirmDialog(true); // Show confirmation dialog when a plan is selected
  };
  
  // Handle subscription
  const handleSubscribe = async () => {
    setIsSubscribing(true);
    setSubscriptionError(null);
    
    try {
      console.log('Subscribing to plan:', selectedPlan);
      const response = await subscriptionService.subscribe({ plan: selectedPlan });
      
      console.log('Subscription response:', response.data);
      
      // Set success notification with more details
      const successMessage = `Successfully subscribed to ${selectedPlan} plan! ${
        response.data.invoice_id ? `Invoice #${response.data.invoice_id} has been created.` : ''
      }`;
      
      setSubscriptionSuccess({
        message: successMessage,
        invoiceId: response.data.invoice_id,
        amount: response.data.amount,
        dueDate: response.data.due_date
      });
      
      // Update subscription info
      await fetchCurrentSubscription();
      
      // Refresh billing data to show the new invoice
      // Add a slight delay to ensure the backend has processed the subscription
      setTimeout(async () => {
        console.log('Fetching billing data after subscription change...');
        await fetchBillingData();
      }, 1000);
      
      // Close dialog
      setShowConfirmDialog(false);
      
      // Set notification manually in case the subscriptionSuccess state update doesn't trigger the effect
      setSubscriptionNotification({
        type: 'success',
        message: successMessage,
        timestamp: new Date()
      });
    } catch (error) {
      console.error('Error subscribing:', error);
      
      // Show error notification with more details
      const errorMessage = error.response?.data?.detail || 
                           error.response?.data?.message ||
                           'Failed to subscribe. Please try again later.';
      
      setSubscriptionError(errorMessage);
      
      // Set error notification
      setSubscriptionNotification({
        type: 'error',
        message: errorMessage,
        timestamp: new Date()
      });
    } finally {
      setIsSubscribing(false);
    }
  };
  
  // Load billing data and subscription on component mount
  useEffect(() => {
    fetchBillingData();
    fetchCurrentSubscription();
    fetchUsageData();
  }, []);

  // Handle subscription success notifications and refresh data
  useEffect(() => {
    if (subscriptionSuccess) {
      // Set notification
      setSubscriptionNotification({
        type: 'success',
        message: subscriptionSuccess.message,
        timestamp: new Date()
      });
      
      // Clear the notification after 5 seconds
      const timer = setTimeout(() => {
        setSubscriptionNotification(null);
      }, 5000);
      
      return () => clearTimeout(timer);
    }
  }, [subscriptionSuccess]);
  
  // Apply filters whenever userBillingData or filter criteria change
  useEffect(() => {
    if (userBillingData.length > 0) {
      applyFilters(userBillingData);
    }
  }, [userBillingData, dateFilter, statusFilter, subscriptionFilter, searchQuery]);
  
  // Update the applyFilters function to actually filter invoices
  const applyFilters = (data = userBillingData) => {
    let filtered = [...data];
    
    // Apply date filtering
    if (dateFilter !== 'all') {
      const now = new Date();
      const startOfMonth = new Date(now.getFullYear(), now.getMonth(), 1);
      const startOfQuarter = new Date(now.getFullYear(), Math.floor(now.getMonth() / 3) * 3, 1);
      const startOfYear = new Date(now.getFullYear(), 0, 1);
      
      filtered = filtered.filter(invoice => {
        const invoiceDate = new Date(invoice.billing_date);
        switch (dateFilter) {
          case 'month':
            return invoiceDate >= startOfMonth;
          case 'quarter':
            return invoiceDate >= startOfQuarter;
          case 'year':
            return invoiceDate >= startOfYear;
          default:
            return true;
        }
      });
    }
    
    // Apply status filtering
    if (statusFilter !== 'all') {
      filtered = filtered.filter(invoice => invoice.status === statusFilter);
    }
    
    // Apply subscription type filtering
    if (subscriptionFilter !== 'all') {
      filtered = filtered.filter(invoice => invoice.subscription_type === subscriptionFilter);
    }
    
    // Apply search query if exists
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(invoice => 
        invoice.id.toString().includes(query) ||
        invoice.description?.toLowerCase().includes(query) ||
        invoice.subscription_type?.toLowerCase().includes(query)
      );
    }
    
    setFilteredInvoices(filtered);
    return filtered;
  };
  
  // Clear all filters
  const clearFilters = () => {
    setStartDate('');
    setEndDate('');
    setDateFilter('all');
    setStatusFilter('all');
    setSubscriptionFilter('all');
    setSearchQuery('');
    
    // Reset filtered invoices to all invoices
    setFilteredInvoices(userBillingData);
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
          ? { bg: 'bg-green-500/20', text: 'text-green-400', border: 'border-green-500/30' }
          : { bg: 'bg-green-100', text: 'text-green-700', border: 'border-green-200' };
      case 'pending':
        return isDarkMode 
          ? { bg: 'bg-amber-500/20', text: 'text-amber-400', border: 'border-amber-500/30' }
          : { bg: 'bg-amber-100', text: 'text-amber-700', border: 'border-amber-200' };
      case 'overdue':
        return isDarkMode 
          ? { bg: 'bg-red-500/20', text: 'text-red-400', border: 'border-red-500/30' }
          : { bg: 'bg-red-100', text: 'text-red-700', border: 'border-red-200' };
      case 'cancelled':
        return isDarkMode 
          ? { bg: 'bg-gray-500/20', text: 'text-gray-400', border: 'border-gray-500/30' }
          : { bg: 'bg-gray-100', text: 'text-gray-700', border: 'border-gray-200' };
      default:
        return isDarkMode 
          ? { bg: 'bg-gray-500/20', text: 'text-gray-400', border: 'border-gray-500/30' }
          : { bg: 'bg-gray-100', text: 'text-gray-700', border: 'border-gray-200' };
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
          ? { bg: 'bg-gray-500/20', text: 'text-gray-400', border: 'border-gray-500/30' }
          : { bg: 'bg-gray-100', text: 'text-gray-700', border: 'border-gray-200' };
      case 'basic':
        return isDarkMode 
          ? { bg: 'bg-blue-500/20', text: 'text-blue-400', border: 'border-blue-500/30' }
          : { bg: 'bg-blue-100', text: 'text-blue-700', border: 'border-blue-200' };
      case 'premium':
        return isDarkMode 
          ? { bg: 'bg-purple-500/20', text: 'text-purple-400', border: 'border-purple-500/30' }
          : { bg: 'bg-purple-100', text: 'text-purple-700', border: 'border-purple-200' };
      default:
        return isDarkMode 
          ? { bg: 'bg-gray-500/20', text: 'text-gray-400', border: 'border-gray-500/30' }
          : { bg: 'bg-gray-100', text: 'text-gray-700', border: 'border-gray-200' };
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
  
  // Add missing function to handle paying an invoice
  const handlePayInvoice = (invoiceId) => {
    // Open the invoice modal with the selected invoice ID
    setSelectedInvoiceId(invoiceId);
    setIsInvoiceModalOpen(true);
  };
  
  // Update the fetchUsageData function to add better error handling, fallback data, and a timeout to prevent hanging requests
  const fetchUsageData = async () => {
    setIsLoadingUsage(true);
    setUsageError(null);
    
    // Create a timeout promise
    const timeoutPromise = new Promise((_, reject) => {
      setTimeout(() => reject(new Error('Request timeout')), 10000);
    });
    
    try {
      // Use token from context instead of localStorage
      if (!token) {
        console.error('No authentication token found');
        setUsageError('Authentication error. Please try logging in again.');
        setIsLoadingUsage(false);
        return;
      }

      // Fetch usage summary with timeout
      try {
        console.log('[fetchUsageData] Requesting usage summary with token:', token.substring(0, 10) + '...');
        const summaryPromise = axios.get('/api/usage/summary/', {
          headers: {
            Authorization: `Bearer ${token}`
          }
        });
        
        // Race the API call against the timeout
        const summaryResponse = await Promise.race([summaryPromise, timeoutPromise]);
        
        if (summaryResponse.data) {
          console.log('[fetchUsageData] Received usage summary data:', summaryResponse.data);
          // Log specific data points we care about
          console.log('[fetchUsageData] Session duration (seconds):', summaryResponse.data.total_duration_seconds);
          console.log('[fetchUsageData] Total corrections:', summaryResponse.data.total_corrections_received);
          
          // Make sure total_duration_seconds is a number
          let durationSeconds = summaryResponse.data.total_duration_seconds;
          if (durationSeconds && typeof durationSeconds !== 'number') {
            durationSeconds = parseInt(durationSeconds, 10);
            console.log('[fetchUsageData] Converted duration to number:', durationSeconds);
          }
          
          // Update the usageData with the parsed duration
          setUsageData({
            ...summaryResponse.data,
            total_duration_seconds: durationSeconds || 0
          });
        } else {
          console.warn('[fetchUsageData] Usage summary returned empty data');
          // Set default values
          setUsageData({
            total_sessions: 0,
            active_sessions: 0,
            total_duration_seconds: 0,
            total_frames_processed: 0,
            total_corrections_received: 0,
            total_billed_amount: 0
          });
        }
      } catch (summaryError) {
        console.error('[fetchUsageData] Error fetching usage summary:', summaryError);
        // Set default values but don't show error yet, try to get sessions
        setUsageData({
          total_sessions: 0,
          active_sessions: 0,
          total_duration_seconds: 0,
          total_frames_processed: 0,
          total_corrections_received: 0,
          total_billed_amount: 0
        });
      }
      
      // Now try to fetch recent usage sessions with timeout
      try {
        console.log('[fetchUsageData] Requesting usage sessions data');
        const sessionsPromise = axios.get('/api/usage/', {
          headers: {
            Authorization: `Bearer ${token}`
          }
        });
        
        // Race the API call against a new timeout
        const sessionsResponse = await Promise.race([sessionsPromise, timeoutPromise]);
        
        if (sessionsResponse.data) {
          console.log('[fetchUsageData] Received usage sessions data:', sessionsResponse.data);
          
          // Check for duration_seconds in the first session if available
          if (sessionsResponse.data.length > 0) {
            console.log('[fetchUsageData] First session duration:', 
              sessionsResponse.data[0].duration_seconds);
          }
          
          setUsageSessions(sessionsResponse.data);
        } else {
          console.warn('[fetchUsageData] Usage sessions returned empty data');
          setUsageSessions([]);
        }
      } catch (sessionsError) {
        console.error('[fetchUsageData] Error fetching usage sessions:', sessionsError);
        setUsageSessions([]);
        
        // Check for specific backend error about User.role
        if (sessionsError.response) {
          console.log('[fetchUsageData] Server error status:', sessionsError.response.status);
          
          if (sessionsError.response.data && 
              typeof sessionsError.response.data === 'string' && 
              sessionsError.response.data.includes("'User' object has no attribute 'role'")) {
            
            // This is a known backend issue
            setUsageError('The usage tracking system is temporarily unavailable. Our team has been notified.');
          } else {
            // Generic error message for other server errors
            setUsageError('Could not retrieve usage data. Please try again later.');
          }
        } else if (sessionsError.message === 'Request timeout') {
          setUsageError('Request timed out. The server may be experiencing high load.');
        } else {
          setUsageError('Network error while retrieving usage data.');
        }
      }
    } catch (error) {
      console.error('[fetchUsageData] Error in usage data fetching:', error);
      setUsageError('Failed to load usage data. Please try again later.');
    } finally {
      setIsLoadingUsage(false);
    }
  };
  
  // Update the formatDuration function to better handle all possible inputs
  const formatDuration = (seconds) => {
    if (seconds === undefined || seconds === null || isNaN(seconds)) {
      console.log('[formatDuration] Invalid input:', seconds);
      return '0s';
    }
    
    const totalSeconds = Math.floor(seconds);
    console.log('[formatDuration] Formatting duration in seconds:', totalSeconds);
    
    const hrs = Math.floor(totalSeconds / 3600);
    const mins = Math.floor((totalSeconds % 3600) / 60);
    const secs = Math.floor(totalSeconds % 60);
    
    if (hrs > 0) {
      return `${hrs}h ${mins}m ${secs}s`;
    } else if (mins > 0) {
      return `${mins}m ${secs}s`;
    } else {
      return `${secs}s`;
    }
  };
  
  // Add Usage Dashboard section
  const renderUsageDashboard = () => {
    return (
      <motion.div 
        className="mt-8"
        initial="hidden"
        animate="visible"
        variants={fadeIn}
      >
        <div className="flex justify-between items-center mb-6">
          <div className="flex items-center">
            <BarChart3 className={`w-5 h-5 mr-2 ${
              isDarkMode ? 'text-purple-400' : 'text-purple-600'
            }`} />
            <h3 className={`text-lg font-semibold ${
              isDarkMode ? 'text-white' : 'text-gray-800'
            }`}>
              Usage Dashboard
            </h3>
          </div>
          
          <motion.button
            whileHover={{ scale: 1.05, y: -2 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setUsageDashboardOpen(!usageDashboardOpen)}
            className={`flex items-center space-x-1 text-sm ${
              isDarkMode ? 'text-gray-300' : 'text-gray-700'
            }`}
          >
            <span>{usageDashboardOpen ? 'Hide Details' : 'Show Details'}</span>
            {usageDashboardOpen ? 
              <ChevronUp className="w-4 h-4" /> : 
              <ChevronDown className="w-4 h-4" />
            }
          </motion.button>
        </div>
        
        {isLoadingUsage ? (
          <div className="flex justify-center py-12">
            <div className="flex flex-col items-center">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                className={`w-10 h-10 rounded-full border-t-2 border-b-2 ${
                  isDarkMode ? 'border-purple-400' : 'border-purple-600'
                } mb-4`}
              />
              <p className={isDarkMode ? 'text-gray-300' : 'text-gray-600'}>
                Loading usage data...
              </p>
            </div>
          </div>
        ) : usageError ? (
          <motion.div 
            className={`rounded-xl p-4 mb-6 border ${
              isDarkMode 
                ? 'bg-red-900/20 backdrop-blur-md text-red-300 border-red-800/30' 
                : 'bg-red-50 text-red-700 border-red-200'
            }`}
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <div className="flex">
              <AlertTriangle className="w-5 h-5 flex-shrink-0 mr-3" />
              <div>
                <h3 className="text-sm font-medium">Error Loading Usage Data</h3>
                <p className="text-sm mt-1">{usageError}</p>
                <div className="mt-3">
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={fetchUsageData}
                    className={`px-3 py-1.5 text-xs rounded-lg flex items-center ${
                      isDarkMode 
                        ? 'bg-red-900/30 hover:bg-red-900/40' 
                        : 'bg-red-100 hover:bg-red-200'
                    }`}
                  >
                    <RefreshCw className="w-3 h-3 mr-1" />
                    Try Again
                  </motion.button>
                </div>
              </div>
            </div>
          </motion.div>
        ) : (
          <motion.div 
            className="mt-6"
            variants={staggerContainer}
          >
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <motion.div
                variants={cardVariant}
                whileHover="hover"
                whileTap="tap"
                className={`${
                  isDarkMode 
                    ? 'bg-gray-800/80 backdrop-blur-md border border-white/5' 
                    : 'bg-white backdrop-blur-md border border-gray-100'
                } rounded-xl p-5 shadow-lg overflow-hidden relative`}
              >
                <div className="absolute top-0 right-0 w-24 h-24 bg-purple-500/5 rounded-full -mr-10 -mt-10"></div>
                <h4 className={`text-sm font-medium ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-500'
                } relative z-10`}>
                  Total Sessions
                </h4>
                <p className={`text-2xl font-bold mt-2 ${
                  isDarkMode ? 'text-white' : 'text-gray-800'
                } relative z-10`}>
                  {usageData.total_sessions}
                </p>
                <div className="absolute bottom-0 right-0 p-3">
                  <Database className={`w-8 h-8 opacity-10 ${
                    isDarkMode ? 'text-purple-400' : 'text-purple-600'
                  }`} />
                </div>
              </motion.div>
              
              <motion.div
                variants={cardVariant}
                whileHover="hover"
                whileTap="tap"
                className={`${
                  isDarkMode 
                    ? 'bg-gray-800/80 backdrop-blur-md border border-white/5' 
                    : 'bg-white backdrop-blur-md border border-gray-100'
                } rounded-xl p-5 shadow-lg overflow-hidden relative`}
              >
                <div className="absolute top-0 right-0 w-24 h-24 bg-indigo-500/5 rounded-full -mr-10 -mt-10"></div>
                <h4 className={`text-sm font-medium ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-500'
                } relative z-10`}>
                  Total Usage Time
                </h4>
                <p className={`text-2xl font-bold mt-2 ${
                  isDarkMode ? 'text-white' : 'text-gray-800'
                } relative z-10`}>
                  {formatDuration(usageData.total_duration_seconds)}
                </p>
                <div className="absolute bottom-0 right-0 p-3">
                  <Clock className={`w-8 h-8 opacity-10 ${
                    isDarkMode ? 'text-indigo-400' : 'text-indigo-600'
                  }`} />
                </div>
              </motion.div>
              
              <motion.div
                variants={cardVariant}
                whileHover="hover"
                whileTap="tap"
                className={`${
                  isDarkMode 
                    ? 'bg-gray-800/80 backdrop-blur-md border border-white/5' 
                    : 'bg-white backdrop-blur-md border border-gray-100'
                } rounded-xl p-5 shadow-lg overflow-hidden relative`}
              >
                <div className="absolute top-0 right-0 w-24 h-24 bg-green-500/5 rounded-full -mr-10 -mt-10"></div>
                <h4 className={`text-sm font-medium ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-500'
                } relative z-10`}>
                  Corrections Received
                </h4>
                <p className={`text-2xl font-bold mt-2 ${
                  isDarkMode ? 'text-white' : 'text-gray-800'
                } relative z-10`}>
                  {(usageData.total_corrections_received || 0).toLocaleString()}
                </p>
                <div className="absolute bottom-0 right-0 p-3">
                  <CheckCircle className={`w-8 h-8 opacity-10 ${
                    isDarkMode ? 'text-green-400' : 'text-green-600'
                  }`} />
                </div>
              </motion.div>
            </div>
            
            <AnimatePresence>
              {usageDashboardOpen && (
                <motion.div 
                  className="mt-6"
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <h4 className={`font-medium ${
                    isDarkMode ? 'text-white' : 'text-gray-700'
                  } mb-4 flex items-center`}>
                    <ActivitySquare className="w-4 h-4 mr-2" />
                    Recent Sessions
                  </h4>
                  
                  {isLoadingUsage ? (
                    <div className="flex justify-center py-8">
                      <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                        className={`w-8 h-8 rounded-full border-t-2 border-b-2 ${
                          isDarkMode ? 'border-purple-400' : 'border-purple-600'
                        }`}
                      />
                    </div>
                  ) : usageSessions.length === 0 ? (
                    <motion.div 
                      className={`rounded-xl p-6 text-center ${
                        isDarkMode 
                          ? 'bg-gray-800/50 border border-white/5' 
                          : 'bg-gray-50 border border-gray-200'
                      }`}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                    >
                      <div className="flex flex-col items-center justify-center">
                        <ReceiptIcon className={`w-12 h-12 ${
                          isDarkMode ? 'text-gray-600' : 'text-gray-400'
                        } mb-3`} />
                        <p className={`font-medium ${
                          isDarkMode ? 'text-white' : 'text-gray-800'
                        }`}>No usage data available</p>
                        <p className={`text-sm mt-1 max-w-sm ${
                          isDarkMode ? 'text-gray-400' : 'text-gray-500'
                        }`}>
                          {token ? 
                            "There's no workout session data yet or the server is experiencing issues. Try again later." : 
                            "Please log in to view your usage data."}
                        </p>
                        <motion.button 
                          whileHover={{ scale: 1.05 }}
                          whileTap={{ scale: 0.95 }}
                          onClick={fetchUsageData}
                          className={`mt-4 px-4 py-2 rounded-lg text-sm flex items-center gap-1 ${
                            isDarkMode
                              ? 'bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 border border-purple-500/30'
                              : 'bg-purple-100 hover:bg-purple-200 text-purple-700 border border-purple-200'
                          }`}
                        >
                          <RefreshCw className={`w-3 h-3 ${isLoadingUsage ? 'animate-spin' : ''}`} />
                          <span>Retry</span>
                        </motion.button>
                      </div>
                    </motion.div>
                  ) : (
                    <div className={`overflow-x-auto rounded-xl border ${
                      isDarkMode ? 'border-white/5' : 'border-gray-200'
                    }`}>
                      <table className={`min-w-full divide-y ${
                        isDarkMode ? 'divide-gray-700' : 'divide-gray-200'
                      }`}>
                        <thead className={isDarkMode ? 'bg-gray-800/80' : 'bg-gray-50'}>
                          <tr>
                            <th className={`px-6 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                              isDarkMode ? 'text-gray-400' : 'text-gray-500'
                            }`}>
                              Date
                            </th>
                            <th className={`px-6 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                              isDarkMode ? 'text-gray-400' : 'text-gray-500'
                            }`}>
                              Duration
                            </th>
                            <th className={`px-6 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                              isDarkMode ? 'text-gray-400' : 'text-gray-500'
                            }`}>
                              Workout Type
                            </th>
                            <th className={`px-6 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                              isDarkMode ? 'text-gray-400' : 'text-gray-500'
                            }`}>
                              Corrections
                            </th>
                            
                          </tr>
                        </thead>
                        <tbody className={`${
                          isDarkMode ? 'bg-gray-900/50 backdrop-blur-md' : 'bg-white'
                        } divide-y ${
                          isDarkMode ? 'divide-gray-800' : 'divide-gray-200'
                        }`}>
                          {usageSessions.map((session) => (
                            <motion.tr 
                              key={session.session_id || session.id || Math.random().toString()} 
                              className={`${
                                isDarkMode 
                                  ? 'hover:bg-gray-800/70' 
                                  : 'hover:bg-gray-50'
                              } transition-colors`}
                              whileHover={{ x: 5 }}
                            >
                              <td className={`px-6 py-4 whitespace-nowrap text-sm ${
                                isDarkMode ? 'text-white' : 'text-gray-800'
                              }`}>
                                {formatDate(session.session_start)}
                              </td>
                              <td className={`px-6 py-4 whitespace-nowrap text-sm ${
                                isDarkMode ? 'text-white' : 'text-gray-800'
                              }`}>
                                {session.duration_formatted || formatDuration(session.duration_seconds)}
                              </td>
                              <td className={`px-6 py-4 whitespace-nowrap text-sm ${
                                isDarkMode ? 'text-white' : 'text-gray-800'
                              }`}>
                                {getWorkoutName(session.workout_type)}
                              </td>
                              <td className={`px-6 py-4 whitespace-nowrap text-sm ${
                                isDarkMode ? 'text-white' : 'text-gray-800'
                              }`}>
                                {(session.corrections_sent || 0).toLocaleString()}
                              </td>
                             
                            </motion.tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        )}
      </motion.div>
    );
  };
  
  // Add function to get workout name
  const getWorkoutName = (workoutType) => {
    const workoutMap = { 
      0: "Barbell Bicep Curl", 1: "Bench Press", 2: "Chest Fly Machine", 
      3: "Deadlift", 4: "Decline Bench Press", 5: "Hammer Curl", 
      6: "Hip Thrust", 7: "Incline Bench Press", 8: "Lat Pulldown", 
      9: "Lateral Raises", 10: "Leg Extensions", 11: "Leg Raises",
      12: "Plank", 13: "Pull Up", 14: "Push Ups", 15: "Romanian Deadlift", 
      16: "Russian Twist", 17: "Shoulder Press", 18: "Squat", 
      19: "T Bar Row", 20: "Tricep Dips", 21: "Tricep Pushdown"
    };
    return workoutMap[workoutType] || "Unknown";
  };
  
  // Update the billing summary cards to focus on WebSocket metrics
  const renderBillingSummary = () => {
    if (isLoadingUsage) {
      return (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6 animate-pulse">
          {[1, 2, 3].map((i) => (
            <div key={i} className={`${
              isDarkMode ? 'bg-gray-700/50' : 'bg-gray-200'
            } rounded-xl h-32`}></div>
          ))}
        </div>
      );
    }

    // Calculate duration limits based on subscription type
    const getDurationLimit = () => {
      if (!currentSubscription || !currentSubscription.plan) return 600; // 10 minutes default
      
      switch (currentSubscription.plan) {
        case 'free':
          return 600; // 10 minutes = 600 seconds
        case 'basic':
          return 7200; // 2 hours = 7200 seconds
        case 'premium':
          return Infinity; // Unlimited
        default:
          return 600; // Default to 10 minutes
      }
    };
    
    const durationLimit = getDurationLimit();
    const currentDuration = usageData.total_duration_seconds || 0;
    
    // Calculate percentage for progress bar, capping at 100%
    const durationPercentage = Math.min((currentDuration / durationLimit) * 100, 100);
    
    // Format the duration limit for display
    const formattedDurationLimit = durationLimit === Infinity ? 
      'Unlimited' : 
      (durationLimit >= 3600 ? 
        `${Math.floor(durationLimit / 3600)} hour${Math.floor(durationLimit / 3600) !== 1 ? 's' : ''}` : 
        `${Math.floor(durationLimit / 60)} min`);
    
    console.log('[renderBillingSummary] Current duration:', currentDuration, 'seconds');
    console.log('[renderBillingSummary] Duration limit:', durationLimit, 'seconds');
    console.log('[renderBillingSummary] Duration percentage:', durationPercentage, '%');

    return (
      <motion.div 
        className="grid grid-cols-1 md:grid-cols-3 gap-5 mb-8"
        variants={staggerContainer}
        initial="hidden"
        animate="visible"
      >
        <motion.div 
          className={`${
            isDarkMode 
              ? 'bg-gray-800/80 backdrop-blur-md border border-white/5' 
              : 'bg-white backdrop-blur-md border border-gray-100'
          } rounded-xl p-5 shadow-lg overflow-hidden relative`}
          variants={cardVariant}
          whileHover="hover"
          whileTap="tap"
        >
          <div className="absolute top-0 right-0 w-32 h-32 bg-purple-500/5 rounded-full -mr-10 -mt-10"></div>
          <h3 className={`text-sm font-medium ${
            isDarkMode ? 'text-gray-400' : 'text-gray-500'
          } relative z-10`}>
            Current Subscription
          </h3>
          <div className="mt-3 flex items-end">
            <span className={`text-2xl font-bold ${
              isDarkMode ? 'text-white' : 'text-gray-900'
            } relative z-10`}>
              {currentSubscription && currentSubscription.plan 
                ? currentSubscription.plan.charAt(0).toUpperCase() + currentSubscription.plan.slice(1) 
                : 'None'}
            </span>
            {currentSubscription && currentSubscription.price !== undefined && (
              <span className={`ml-2 text-sm ${
                isDarkMode ? 'text-gray-400' : 'text-gray-500'
              } relative z-10`}>
                ${typeof currentSubscription.price === 'number' 
                  ? currentSubscription.price.toFixed(2) 
                  : currentSubscription.price}/mo
              </span>
            )}
          </div>
          <div className={`mt-2 text-sm ${
            isDarkMode ? 'text-gray-400' : 'text-gray-600'
          } relative z-10`}>
            {currentSubscription && currentSubscription.days_remaining !== undefined ? (
              <>Expires in {currentSubscription.days_remaining} days</>
            ) : (
              <>No active subscription</>
            )}
          </div>
          <div className="absolute bottom-0 right-0 p-3">
            <DollarSign className={`w-10 h-10 opacity-10 ${
              isDarkMode ? 'text-purple-400' : 'text-purple-600'
            }`} />
          </div>
        </motion.div>

        <motion.div 
          className={`${
            isDarkMode 
              ? 'bg-gray-800/80 backdrop-blur-md border border-white/5' 
              : 'bg-white backdrop-blur-md border border-gray-100'
          } rounded-xl p-5 shadow-lg overflow-hidden relative`}
          variants={cardVariant}
          whileHover="hover"
          whileTap="tap"
        >
          <div className="absolute top-0 right-0 w-32 h-32 bg-blue-500/5 rounded-full -mr-10 -mt-10"></div>
          <h3 className={`text-sm font-medium ${
            isDarkMode ? 'text-gray-400' : 'text-gray-500'
          } relative z-10`}>
            Monthly Usage Time
          </h3>
          <div className="mt-3 relative z-10">
            <span className={`text-2xl font-bold ${
              isDarkMode ? 'text-white' : 'text-gray-900'
            }`}>
              {formatDuration(currentDuration)}
            </span>
            <span className={`ml-2 text-sm ${
              isDarkMode ? 'text-gray-400' : 'text-gray-500'
            }`}>
              of {formattedDurationLimit}
            </span>
          </div>
          <div className="mt-4 relative z-10">
            <div className={`w-full ${
              isDarkMode ? 'bg-gray-700' : 'bg-gray-200'
            } rounded-full h-2.5 overflow-hidden`}>
              <div 
                className={`${
                  durationPercentage >= 90 ? 
                    (isDarkMode ? 'bg-red-500' : 'bg-red-500') :
                    (isDarkMode ? 'bg-purple-500' : 'bg-purple-600')
                } h-2.5 rounded-full transition-all duration-500 ease-in-out`}
                style={{ width: `${durationPercentage}%` }}
              ></div>
            </div>
            <p className={`text-xs ${
              isDarkMode ? 'text-gray-400' : 'text-gray-500'
            } mt-1.5`}>
              {durationPercentage >= 90 && durationLimit !== Infinity ? 
                <span className={isDarkMode ? 'text-red-400' : 'text-red-600'}>Almost at limit!</span> : 
                `${formattedDurationLimit} monthly limit`}
            </p>
          </div>
          <div className="absolute bottom-0 right-0 p-3">
            <Clock className={`w-10 h-10 opacity-10 ${
              isDarkMode ? 'text-blue-400' : 'text-blue-600'
            }`} />
          </div>
        </motion.div>

        <motion.div 
          className={`${
            isDarkMode 
              ? 'bg-gray-800/80 backdrop-blur-md border border-white/5' 
              : 'bg-white backdrop-blur-md border border-gray-100'
          } rounded-xl p-5 shadow-lg overflow-hidden relative`}
          variants={cardVariant}
          whileHover="hover"
          whileTap="tap"
        >
          <div className="absolute top-0 right-0 w-32 h-32 bg-green-500/5 rounded-full -mr-10 -mt-10"></div>
          <h3 className={`text-sm font-medium ${
            isDarkMode ? 'text-gray-400' : 'text-gray-500'
          } relative z-10`}>
            Corrections Used
          </h3>
          <div className="mt-3 relative z-10">
            <span className={`text-2xl font-bold ${
              isDarkMode ? 'text-white' : 'text-gray-900'
            }`}>
              {(usageData.total_corrections_received || 0).toLocaleString()}
            </span>
          </div>
          <div className="mt-4 relative z-10">
            <div className={`w-full ${
              isDarkMode ? 'bg-gray-700' : 'bg-gray-200'
            } rounded-full h-2.5 overflow-hidden`}>
              <div 
                className={`${isDarkMode ? 'bg-green-500' : 'bg-green-600'} h-2.5 rounded-full`}
                style={{ 
                  width: `${Math.min(
                    ((usageData.total_corrections_received || 0) / 
                     (currentSubscription && currentSubscription.plan === 'free' ? 1000 : 
                      currentSubscription && currentSubscription.plan === 'basic' ? 10000 : 100000)) * 100, 100
                  )}%` 
                }}
              ></div>
            </div>
            <p className={`text-xs ${
              isDarkMode ? 'text-gray-400' : 'text-gray-500'
            } mt-1.5`}>
              {currentSubscription && currentSubscription.plan === 'free' ? '1,000 limit' : 
               currentSubscription && currentSubscription.plan === 'basic' ? '10,000 limit' : 'Unlimited'}
            </p>
          </div>
          <div className="absolute bottom-0 right-0 p-3">
            <CheckCircle className={`w-10 h-10 opacity-10 ${
              isDarkMode ? 'text-green-400' : 'text-green-600'
            }`} />
          </div>
        </motion.div>
      </motion.div>
    );
  };
  
  // Render subscription plans
  const renderSubscriptionPlans = () => {
    return (
      <motion.div 
        initial="hidden"
        animate="visible"
        variants={staggerContainer}
        className="mt-8"
      >
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center">
            <CreditCard className={`w-5 h-5 mr-2 ${
              isDarkMode ? 'text-purple-400' : 'text-purple-600'
            }`} />
            <h3 className={`text-lg font-semibold ${
              isDarkMode ? 'text-white' : 'text-gray-800'
            }`}>
              Subscription Plans
            </h3>
          </div>
          
          <motion.button
            whileHover={{ scale: 1.05, y: -2 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleRefreshAll}
            className={`px-3 py-1.5 rounded-lg text-sm flex items-center gap-2 ${
              isDarkMode
                ? 'bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 border border-purple-500/30'
                : 'bg-purple-100 hover:bg-purple-200 text-purple-700 border border-purple-200'
            }`}
          >
            <RefreshCw className={`w-3.5 h-3.5 ${
              loading || subscriptionLoading ? 'animate-spin' : ''
            }`} />
            <span>Refresh</span>
          </motion.button>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {subscriptionPlans.map((plan) => (
            <motion.div
              key={plan.id}
              variants={cardVariant}
              whileHover="hover"
              whileTap="tap"
              className={`${
                isDarkMode 
                  ? 'bg-gray-800/80 backdrop-blur-md border border-white/5' 
                  : 'bg-white backdrop-blur-md border border-gray-100'
              } rounded-xl shadow-lg overflow-hidden ${
                currentSubscription?.plan === plan.id ?
                  (isDarkMode 
                    ? 'ring-2 ring-purple-500/50' 
                    : 'ring-2 ring-purple-500/30')
                  : ''
              }`}
            >
              <div className="p-6">
                <div className={`${
                  isDarkMode ? 'bg-purple-500/10' : 'bg-purple-100'
                } w-12 h-12 rounded-full flex items-center justify-center mb-4 ${
                  currentSubscription?.plan === plan.id ?
                    (isDarkMode ? 'text-purple-400' : 'text-purple-600') :
                    (isDarkMode ? 'text-gray-400' : 'text-gray-600')
                }`}>
                  {plan.icon}
                </div>
                
                <div className="flex justify-between items-start mb-4">
                  <h3 className={`text-lg font-bold ${
                    isDarkMode ? 'text-white' : 'text-gray-900'
                  }`}>
                    {plan.name}
                  </h3>
                  <div className="text-right">
                    <div className={`text-2xl font-bold ${
                      isDarkMode ? 'text-white' : 'text-gray-900'
                    }`}>
                      ${plan.price.toFixed(2)}
                    </div>
                    <div className={`text-sm ${
                      isDarkMode ? 'text-gray-400' : 'text-gray-500'
                    }`}>
                      per month
                    </div>
                  </div>
                </div>
                
                {plan.description && (
                  <p className={`text-sm ${
                    isDarkMode ? 'text-gray-400' : 'text-gray-600'
                  } mb-4`}>
                    {plan.description}
                  </p>
                )}
                
                <ul className="space-y-2 mb-6">
                  {plan.features.map((feature, i) => (
                    <li key={i} className="flex items-start">
                      <CheckCircle className={`h-4 w-4 ${
                        isDarkMode ? 'text-green-400' : 'text-green-600'
                      } mr-2 flex-shrink-0 mt-0.5`} />
                      <span className={`text-sm ${
                        isDarkMode ? 'text-gray-300' : 'text-gray-700'
                      }`}>
                        {feature}
                      </span>
                    </li>
                  ))}
                </ul>
                
                <div className="mt-auto">
                  {currentSubscription?.plan === plan.id ? (
                    <button
                      className={`w-full py-2 px-4 rounded-lg font-medium flex items-center justify-center ${
                        isDarkMode 
                          ? 'bg-green-500/20 text-green-400 border border-green-500/30' 
                          : 'bg-green-100 text-green-700 border border-green-200'
                      } cursor-default`}
                      disabled
                    >
                      <CheckCircle className="w-4 h-4 mr-2" />
                      Current Plan
                    </button>
                  ) : (
                    <motion.button
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={() => handleSelectPlan(plan.id)}
                      className={`w-full py-2 px-4 rounded-lg font-medium transition-colors ${
                        isDarkMode 
                          ? 'bg-purple-500/80 hover:bg-purple-600/80 text-white' 
                          : 'bg-purple-600 hover:bg-purple-700 text-white'
                      }`}
                    >
                      Subscribe
                    </motion.button>
                  )}
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>
    );
  };
  
  // Render invoices table
  const renderInvoicesTable = () => {
    return (
      <motion.div 
        className={`${
          isDarkMode 
            ? 'bg-gray-800/80 backdrop-blur-md border border-white/5' 
            : 'bg-white backdrop-blur-md border border-gray-100'
        } rounded-xl shadow-lg overflow-hidden mt-10`}
        initial="hidden"
        animate="visible"
        variants={fadeIn}
      >
        <div className="p-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-6 gap-4">
            <div className="flex items-center">
              <FileText className={`w-5 h-5 mr-2 ${
                isDarkMode ? 'text-purple-400' : 'text-purple-600'
              }`} />
              <h3 className={`text-lg font-semibold ${
                isDarkMode ? 'text-white' : 'text-gray-800'
              }`}>
                Invoices History
              </h3>
            </div>
            
            <div className="flex flex-wrap items-center gap-3">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={fetchBillingData}
                className={`flex items-center space-x-1 px-3 py-1.5 rounded-lg text-sm ${
                  isDarkMode 
                    ? 'bg-gray-700 hover:bg-gray-600 text-white' 
                    : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
                }`}
              >
                <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
                <span>Refresh</span>
              </motion.button>
              
              <div className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg ${
                isDarkMode 
                  ? 'bg-gray-700 text-white border border-white/5' 
                  : 'bg-white text-gray-700 border border-gray-200'
              }`}>
                <Calendar className="h-4 w-4" />
                <select
                  className={`bg-transparent text-sm focus:outline-none ${
                    isDarkMode 
                      ? 'text-white' 
                      : 'text-gray-700'
                  }`}
                  value={dateFilter}
                  onChange={(e) => setDateFilter(e.target.value)}
                >
                  <option value="all">All Time</option>
                  <option value="month">This Month</option>
                  <option value="quarter">This Quarter</option>
                  <option value="year">This Year</option>
                </select>
              </div>
              
              <div className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg ${
                isDarkMode 
                  ? 'bg-gray-700 text-white border border-white/5' 
                  : 'bg-white text-gray-700 border border-gray-200'
              }`}>
                <Filter className="h-4 w-4" />
                <select
                  className={`bg-transparent text-sm focus:outline-none ${
                    isDarkMode 
                      ? 'text-white' 
                      : 'text-gray-700'
                  }`}
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value)}
                >
                  <option value="all">All Status</option>
                  <option value="paid">Paid</option>
                  <option value="pending">Pending</option>
                  <option value="overdue">Overdue</option>
                </select>
              </div>
              
              {(dateFilter !== 'all' || statusFilter !== 'all') && (
                <button
                  onClick={clearFilters}
                  className={`text-sm ${
                    isDarkMode 
                      ? 'text-purple-400 hover:text-purple-300' 
                      : 'text-purple-600 hover:text-purple-800'
                  }`}
                >
                  Clear Filters
                </button>
              )}
            </div>
          </div>
          
          {/* Invoices Table */}
          {loading ? (
            <div className="animate-pulse space-y-3">
              <div className={`h-10 ${
                isDarkMode ? 'bg-gray-700' : 'bg-gray-200'
              } rounded-lg mb-4`}></div>
              {[1, 2, 3].map((i) => (
                <div key={i} className={`h-16 ${
                  isDarkMode ? 'bg-gray-700/50' : 'bg-gray-100'
                } rounded-lg`}></div>
              ))}
            </div>
          ) : filteredInvoices.length === 0 ? (
            <motion.div 
              className="text-center py-12"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2 }}
            >
              <FileText className={`mx-auto h-16 w-16 ${
                isDarkMode ? 'text-gray-600' : 'text-gray-300'
              }`} />
              <h3 className={`mt-4 text-lg font-medium ${
                isDarkMode ? 'text-white' : 'text-gray-900'
              }`}>
                No invoices
              </h3>
              <p className={`mt-2 text-sm max-w-md mx-auto ${
                isDarkMode ? 'text-gray-400' : 'text-gray-500'
              }`}>
                {userBillingData.length > 0 
                  ? 'No invoices match your current filters.' 
                  : 'You don\'t have any invoices yet.'}
              </p>
              <motion.button 
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={fetchBillingData}
                className={`mt-4 px-4 py-2 rounded-lg text-sm flex items-center gap-1 mx-auto ${
                  isDarkMode
                    ? 'bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 border border-purple-500/30'
                    : 'bg-purple-100 hover:bg-purple-200 text-purple-700 border border-purple-200'
                }`}
              >
                <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
                <span>Refresh</span>
              </motion.button>
            </motion.div>
          ) : (
            <div className={`overflow-x-auto rounded-xl border ${
              isDarkMode ? 'border-white/5' : 'border-gray-200'
            }`}>
              <table className={`min-w-full divide-y ${
                isDarkMode ? 'divide-gray-700' : 'divide-gray-200'
              }`}>
                <thead className={isDarkMode ? 'bg-gray-800/80' : 'bg-gray-50'}>
                  <tr>
                    <th className={`px-6 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                      isDarkMode ? 'text-gray-400' : 'text-gray-500'
                    }`}>
                      Invoice
                    </th>
                    <th className={`px-6 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                      isDarkMode ? 'text-gray-400' : 'text-gray-500'
                    }`}>
                      Date
                    </th>
                    <th className={`px-6 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                      isDarkMode ? 'text-gray-400' : 'text-gray-500'
                    }`}>
                      Amount
                    </th>
                    <th className={`px-6 py-3 text-left text-xs font-medium uppercase tracking-wider ${
                      isDarkMode ? 'text-gray-400' : 'text-gray-500'
                    }`}>
                      Status
                    </th>
                    <th className={`px-6 py-3 text-right text-xs font-medium uppercase tracking-wider ${
                      isDarkMode ? 'text-gray-400' : 'text-gray-500'
                    }`}>
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className={`${
                  isDarkMode ? 'bg-gray-900/50 backdrop-blur-md' : 'bg-white'
                } divide-y ${
                  isDarkMode ? 'divide-gray-800' : 'divide-gray-200'
                }`}>
                  <AnimatePresence initial={false}>
                    {filteredInvoices.map((record) => (
                      <motion.tr 
                        key={record.id} 
                        className={`${
                          isDarkMode 
                            ? 'hover:bg-gray-800/70' 
                            : 'hover:bg-gray-50'
                        } transition-all`}
                        variants={tableRowVariant}
                        initial="hidden"
                        animate="visible"
                        exit="exit"
                        whileHover={{ x: 5 }}
                      >
                        <td className="px-6 py-4">
                          <div className="flex items-center">
                            <div>
                              <div className={`text-sm font-medium ${
                                isDarkMode ? 'text-white' : 'text-gray-900'
                              }`}>
                                INV-{record.id.toString().padStart(4, '0')}
                              </div>
                              <div className={`text-sm ${
                                isDarkMode ? 'text-gray-400' : 'text-gray-500'
                              }`}>
                                {record.subscription_type}
                              </div>
                            </div>
                          </div>
                        </td>
                        <td className="px-6 py-4">
                          <div className={`text-sm ${
                            isDarkMode ? 'text-white' : 'text-gray-900'
                          }`}>
                            {formatDate(record.billing_date)}
                          </div>
                          <div className={`text-sm ${
                            isDarkMode ? 'text-gray-400' : 'text-gray-500'
                          }`}>
                            Due: {formatDate(record.due_date)}
                          </div>
                        </td>
                        <td className="px-6 py-4">
                          <div className={`text-sm font-medium ${
                            isDarkMode ? 'text-white' : 'text-gray-900'
                          }`}>
                            {formatCurrency(record.amount)}
                          </div>
                        </td>
                        <td className="px-6 py-4">
                          <span className={`px-2.5 py-1 inline-flex text-xs leading-5 font-medium rounded-full border ${
                            getStatusColor(record.status).bg
                          } ${getStatusColor(record.status).text} ${
                            getStatusColor(record.status).border
                          }`}>
                            {getStatusIcon(record.status)}
                            {record.status.charAt(0).toUpperCase() + record.status.slice(1)}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-right text-sm font-medium">
                          <motion.button
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            onClick={() => handleOpenInvoiceModal(record.id)}
                            className={`${
                              isDarkMode 
                                ? 'text-purple-400 hover:text-purple-300' 
                                : 'text-purple-600 hover:text-purple-800'
                            } mr-3`}
                          >
                            View
                          </motion.button>
                          {record.status === 'pending' && (
                            <motion.button
                              whileHover={{ scale: 1.05 }}
                              whileTap={{ scale: 0.95 }}
                              onClick={() => handlePayInvoice(record.id)}
                              className={`${
                                isDarkMode 
                                  ? 'text-green-400 hover:text-green-300' 
                                  : 'text-green-600 hover:text-green-800'
                              }`}
                            >
                              Pay
                            </motion.button>
                          )}
                        </td>
                      </motion.tr>
                    ))}
                  </AnimatePresence>
                </tbody>
              </table>
            </div>
          )}
        </div>
      </motion.div>
    );
  };
  
  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-6 gap-4">
        <h2 className={`text-2xl font-light ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>
          Your <span className="text-purple-400 font-medium">Invoices & Billing</span>
        </h2>
        
        <motion.button
          whileHover={{ scale: 1.05, y: -2 }}
          whileTap={{ scale: 0.95 }}
          onClick={handleRefreshAll}
          className={`${
            isDarkMode 
              ? 'bg-gray-800/80 backdrop-blur-md border border-white/5 text-white' 
              : 'bg-white border border-gray-200 text-gray-700'
          } px-4 py-2 rounded-lg flex items-center space-x-2 shadow-md`}
        >
          <RefreshCw className={`w-4 h-4 ${subscriptionLoading || loading || isLoadingUsage ? 'animate-spin' : ''}`} />
          <span className="font-medium">Refresh All</span>
        </motion.button>
      </div>
      
      {/* Show subscription notification */}
      <AnimatePresence>
        {subscriptionNotification && (
          <motion.div 
            className={`rounded-xl p-4 mb-6 border ${
              subscriptionNotification.type === 'success'
                ? (isDarkMode 
                    ? 'bg-green-900/20 backdrop-blur-md text-green-300 border-green-800/30' 
                    : 'bg-green-50 text-green-700 border-green-200')
                : (isDarkMode 
                    ? 'bg-red-900/20 backdrop-blur-md text-red-300 border-red-800/30' 
                    : 'bg-red-50 text-red-700 border-red-200')
            }`}
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <div className="flex items-center">
              {subscriptionNotification.type === 'success' 
                ? <CheckCircle className="w-5 h-5 mr-2 flex-shrink-0" />
                : <AlertCircle className="w-5 h-5 mr-2 flex-shrink-0" />
              }
              <p>{subscriptionNotification.message}</p>
              <motion.button 
                onClick={() => setSubscriptionNotification(null)}
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                className="ml-auto"
              >
                <X className="w-4 h-4" />
              </motion.button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      <AnimatePresence>
        {error && (
          <motion.div 
            className={`rounded-xl p-4 mb-6 border ${
              isDarkMode 
                ? 'bg-red-900/20 backdrop-blur-md text-red-300 border-red-800/30' 
                : 'bg-red-50 text-red-700 border-red-200'
            }`}
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <div className="flex items-center">
              <AlertCircle className="w-5 h-5 mr-2 flex-shrink-0" />
              <p>{error}</p>
              <motion.button 
                onClick={() => setError(null)}
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                className="ml-auto"
              >
                <X className="w-4 h-4" />
              </motion.button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      {loading && subscriptionLoading && isLoadingUsage ? (
        <div className="flex justify-center items-center py-24">
          <div className="flex flex-col items-center">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
              className={`w-12 h-12 rounded-full border-t-2 border-b-2 ${
                isDarkMode ? 'border-purple-400' : 'border-purple-600'
              } mb-4`}
            />
            <p className={isDarkMode ? 'text-gray-300' : 'text-gray-600'}>
              Loading your billing data...
            </p>
          </div>
        </div>
      ) : (
        <>          
          {/* Subscription Section */}
          <motion.div 
            className={`${
              isDarkMode 
                ? 'bg-gray-800/80 backdrop-blur-md border border-white/5' 
                : 'bg-white backdrop-blur-md border border-gray-100'
            } rounded-xl shadow-lg overflow-hidden`}
            initial="hidden"
            animate="visible"
            variants={fadeIn}
          >
            <div className="p-6">
              <div className="flex items-center mb-6">
                <DollarSign className={`w-5 h-5 mr-2 ${
                  isDarkMode ? 'text-purple-400' : 'text-purple-600'
                }`} />
                <h2 className={`text-xl font-semibold ${
                  isDarkMode ? 'text-white' : 'text-gray-800'
                }`}>
                  Subscription & Usage
                </h2>
              </div>
              
              {/* Billing Summary */}
              {renderBillingSummary()}
              
              {/* Subscription Plans */}
              {renderSubscriptionPlans()}
              
              {/* Show confirmation dialog */}
              <AnimatePresence>
                {showConfirmDialog && (
                  <motion.div 
                    className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    onClick={() => setShowConfirmDialog(false)}
                  >
                    <motion.div 
                      className={`${
                        isDarkMode 
                          ? 'bg-gray-800 border border-white/10' 
                          : 'bg-white border border-gray-200'
                      } rounded-xl p-6 max-w-md w-full mx-4 shadow-xl`}
                      initial={{ scale: 0.9, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      exit={{ scale: 0.9, opacity: 0 }}
                      onClick={e => e.stopPropagation()}
                    >
                      <h3 className={`text-xl font-semibold mb-4 ${
                        isDarkMode ? 'text-white' : 'text-gray-800'
                      }`}>
                        Confirm Subscription
                      </h3>
                      
                      {subscriptionError && (
                        <div className={`mb-4 p-3 rounded-lg ${
                          isDarkMode 
                            ? 'bg-red-900/30 text-red-300 border border-red-800/30' 
                            : 'bg-red-50 text-red-700 border border-red-200'
                        }`}>
                          <div className="flex">
                            <AlertCircle className="w-5 h-5 mr-2 flex-shrink-0" />
                            <p>{subscriptionError}</p>
                          </div>
                        </div>
                      )}
                      
                      <p className={`mb-6 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                        Are you sure you want to subscribe to the {
                          subscriptionPlans.find(p => p.id === selectedPlan)?.name
                        } plan?
                        {selectedPlan !== 'free' && ' You will be billed ' + 
                         formatCurrency(subscriptionPlans.find(p => p.id === selectedPlan)?.price) + ' monthly.'}
                      </p>
                      
                      <div className="flex justify-end space-x-4">
                        <motion.button
                          whileHover={{ scale: 1.05 }}
                          whileTap={{ scale: 0.95 }}
                          onClick={() => setShowConfirmDialog(false)}
                          className={`px-4 py-2 rounded-lg ${
                            isDarkMode 
                              ? 'bg-gray-700 text-gray-300 hover:bg-gray-600' 
                              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                          }`}
                        >
                          Cancel
                        </motion.button>
                        <motion.button
                          whileHover={{ scale: 1.05 }}
                          whileTap={{ scale: 0.95 }}
                          onClick={handleSubscribe}
                          disabled={isSubscribing}
                          className={`px-4 py-2 rounded-lg ${
                            isDarkMode 
                              ? 'bg-purple-500 hover:bg-purple-600 text-white' 
                              : 'bg-purple-600 hover:bg-purple-700 text-white'
                          } ${isSubscribing ? 'opacity-70 cursor-not-allowed' : ''}`}
                        >
                          {isSubscribing ? (
                            <div className="flex items-center space-x-2">
                              <motion.div
                                animate={{ rotate: 360 }}
                                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                                className="w-4 h-4 border-2 border-white border-t-transparent rounded-full"
                              />
                              <span>Processing...</span>
                            </div>
                          ) : (
                            'Confirm'
                          )}
                        </motion.button>
                      </div>
                    </motion.div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>
          
          {/* Usage Dashboard */}
          {renderUsageDashboard()}
          
          {/* Invoices History section */}
          {renderInvoicesTable()}
        </>
      )}
      
      {/* Invoice Modal */}
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
            isAdminView={false}
          />
        )}
      </AnimatePresence>
    </div>
  );
};

export default BillingOverview;