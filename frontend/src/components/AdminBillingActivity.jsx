import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import { 
  AlertCircle, RefreshCw, Calendar, DollarSign, 
  Filter, Download, FileText, Search, X, 
  User, ChevronLeft, ChevronRight
} from 'lucide-react';
import InvoiceDetailModal from './InvoiceDetailModal';

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

const AdminBillingActivity = ({ isDarkMode }) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [billingData, setBillingData] = useState({
    billing_records: [],
    invoices: [],
    usage_records: [],
    summary: {},
    subscription_activity: {},
    pagination: {
      page: 1,
      page_size: 50,
      total_pages: 1
    }
  });
  
  // Invoice modal state
  const [selectedInvoiceId, setSelectedInvoiceId] = useState(null);
  const [isInvoiceModalOpen, setIsInvoiceModalOpen] = useState(false);
  
  // Filter state
  const [startDate, setStartDate] = useState(
    new Date(new Date().setDate(new Date().getDate() - 30)).toISOString().split('T')[0]
  );
  const [endDate, setEndDate] = useState(
    new Date().toISOString().split('T')[0]
  );
  const [statusFilter, setStatusFilter] = useState('');
  const [subscriptionFilter, setSubscriptionFilter] = useState('');
  const [usernameFilter, setUsernameFilter] = useState('');
  const [isFilterExpanded, setIsFilterExpanded] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [currentTab, setCurrentTab] = useState('all');

  // Fetch billing activity data
  const fetchBillingActivity = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Build the query string with filters
      let query = [
        `start_date=${startDate}`,
        `end_date=${endDate}`,
        `page=${currentPage}`,
        `page_size=50`
      ];
      
      if (statusFilter) query.push(`status=${statusFilter}`);
      if (subscriptionFilter) query.push(`subscription_type=${subscriptionFilter}`);
      if (usernameFilter) query.push(`username=${usernameFilter}`);
      
      const queryString = query.join('&');
      
      // Call the billable_activity endpoint
      const response = await axios.get(
        `http://localhost:8000/api/billing/billable_activity/?${queryString}`, 
        { headers: { Authorization: `Bearer ${localStorage.getItem('access_token')}` }}
      );
      
      console.log("Billing activity data:", response.data);
      setBillingData(response.data);
      
    } catch (err) {
      console.error("Error fetching billing activity data:", err);
      setError(err.response?.data?.detail || "Failed to load billing activity data. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  // Load data when component mounts
  useEffect(() => {
    fetchBillingActivity();
  }, [currentPage]);
  
  // Apply filters
  const applyFilters = () => {
    setCurrentPage(1); // Reset to first page when filters change
    fetchBillingActivity();
    setIsFilterExpanded(false);
  };
  
  // Clear filters
  const clearFilters = () => {
    setStartDate(new Date(new Date().setDate(new Date().getDate() - 30)).toISOString().split('T')[0]);
    setEndDate(new Date().toISOString().split('T')[0]);
    setStatusFilter('');
    setSubscriptionFilter('');
    setUsernameFilter('');
    setCurrentPage(1);
    fetchBillingActivity();
  };
  
  // Format currency
  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(amount);
  };
  
  // Format date
  const formatDate = (dateString) => {
    const options = { year: 'numeric', month: 'short', day: 'numeric' };
    return new Date(dateString).toLocaleDateString('en-US', options);
  };

  // Get status color styling
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
  
  // Get subscription type color styling
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
          : { bg: 'bg-indigo-100', text: 'text-indigo-700' };
      default:
        return isDarkMode 
          ? { bg: 'bg-gray-500/20', text: 'text-gray-400' }
          : { bg: 'bg-gray-100', text: 'text-gray-700' };
    }
  };

  // Handle opening the invoice modal
  const handleOpenInvoiceModal = (invoiceId) => {
    setSelectedInvoiceId(invoiceId);
    setIsInvoiceModalOpen(true);
  };
  
  // Handle closing the invoice modal
  const handleCloseInvoiceModal = () => {
    setIsInvoiceModalOpen(false);
    setSelectedInvoiceId(null);
  };
  
  // Handle invoice payment success
  const handleInvoicePaymentSuccess = () => {
    // Refresh the billing data to reflect the changes
    fetchBillingActivity();
  };

  const renderActivityTab = () => {
    let dataToDisplay = [];
    
    // Determine what data to show based on the selected tab
    switch(currentTab) {
      case 'billing':
        dataToDisplay = billingData.billing_records || [];
        break;
      case 'invoices':
        dataToDisplay = billingData.invoices || [];
        break;
      case 'usage':
        dataToDisplay = billingData.usage_records || [];
        break;
      default:
        // For 'all' tab, combine and sort all records by date
        const allRecords = [
          ...(billingData.billing_records || []).map(record => ({
            ...record,
            type: 'billing',
            date: record.billing_date
          })),
          ...(billingData.invoices || []).map(invoice => ({
            ...invoice,
            type: 'invoice',
            date: invoice.invoice_date
          })),
          ...(billingData.usage_records || []).map(usage => ({
            ...usage,
            type: 'usage',
            date: usage.timestamp?.split('T')[0]
          }))
        ];
        
        // Sort by date (most recent first)
        dataToDisplay = allRecords.sort((a, b) => 
          new Date(b.date) - new Date(a.date)
        );
    }
    
    if (dataToDisplay.length === 0) {
      return (
        <div className={`py-16 text-center ${
          isDarkMode ? 'text-gray-400' : 'text-gray-500'
        }`}>
          <FileText className="w-12 h-12 mx-auto mb-4 opacity-20" />
          <p className="text-lg">No billable activity found</p>
          <p className="text-sm mt-1 opacity-75">Try adjusting your filters</p>
        </div>
      );
    }

    return (
      <div className={`divide-y ${isDarkMode ? 'divide-white/5' : 'divide-gray-200'}`}>
        {dataToDisplay.map((record) => (
          <motion.div
            key={`${record.type || 'item'}-${record.id}`}
            whileHover={{ 
              backgroundColor: isDarkMode ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.02)'
            }}
            className="grid grid-cols-12 gap-4 px-6 py-4 items-center"
          >
            <div className="col-span-2 md:col-span-3 flex items-center space-x-3">
              <div className={`${
                isDarkMode ? 'bg-gray-700' : 'bg-gray-100'
              } p-2 rounded-full`}>
                <User className={`w-4 h-4 ${
                  isDarkMode ? 'text-gray-300' : 'text-gray-600'
                }`} />
              </div>
              <div>
                <div className={`font-medium truncate ${
                  isDarkMode ? 'text-white' : 'text-gray-800'
                }`}>
                  {record.username || record.user?.username || 'Unknown User'}
                </div>
                <div className={`text-xs truncate ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-500'
                }`}>
                  {record.email || record.user?.email || 'No email'}
                </div>
              </div>
            </div>
            
            <div className={`col-span-2 font-medium ${
              isDarkMode ? 'text-white' : 'text-gray-800'
            }`}>
              {formatCurrency(record.amount || 0)}
            </div>
            
            <div className="col-span-2 hidden md:block">
              <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium
                ${getStatusColor(record.status || 'pending').bg} 
                ${getStatusColor(record.status || 'pending').text}`}
              >
                {(record.status || 'Pending').charAt(0).toUpperCase() + (record.status || 'Pending').slice(1)}
              </span>
            </div>
            
            <div className="col-span-2 hidden md:block">
              <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium
                ${getSubscriptionColor(record.subscription_type || 'free').bg} 
                ${getSubscriptionColor(record.subscription_type || 'free').text}`}
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
                {formatDate(record.date || record.billing_date || record.invoice_date || record.timestamp?.split('T')[0] || new Date().toISOString().split('T')[0])}
              </div>
              {record.due_date && (
                <div className="text-xs text-gray-500">
                  Due: {formatDate(record.due_date)}
                </div>
              )}
            </div>
            
            <div className="col-span-1 flex justify-end">
              <button 
                onClick={() => handleOpenInvoiceModal(record.id)}
                className={`p-1 rounded-lg ${
                  isDarkMode 
                    ? 'hover:bg-gray-700 text-gray-400 hover:text-white' 
                    : 'hover:bg-gray-100 text-gray-500 hover:text-gray-800'
                }`}
              >
                <FileText className="w-4 h-4" />
              </button>
            </div>
          </motion.div>
        ))}
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-8 gap-4">
        <h2 className={`text-2xl font-light ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>
          Billable <span className="text-purple-400 font-medium">Activity</span>
        </h2>
        
        <div className="flex space-x-2">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setIsFilterExpanded(!isFilterExpanded)}
            className={`${
              isDarkMode 
                ? 'bg-gray-800 hover:bg-gray-700 text-white border border-white/10' 
                : 'bg-white hover:bg-gray-50 text-gray-700 border border-gray-200'
            } px-3 py-2 rounded-lg flex items-center space-x-2 shadow-sm`}
          >
            <Filter className="w-4 h-4" />
            <span className="text-sm font-medium">Filters</span>
          </motion.button>
          
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={fetchBillingActivity}
            className={`${
              isDarkMode 
                ? 'bg-gray-800 hover:bg-gray-700 text-white border border-white/10' 
                : 'bg-white hover:bg-gray-50 text-gray-700 border border-gray-200'
            } px-3 py-2 rounded-lg flex items-center space-x-2 shadow-sm`}
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            <span className="text-sm font-medium">Refresh</span>
          </motion.button>
          
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className={`${
              isDarkMode 
                ? 'bg-gray-800 hover:bg-gray-700 text-white border border-white/10' 
                : 'bg-white hover:bg-gray-50 text-gray-700 border border-gray-200'
            } px-3 py-2 rounded-lg flex items-center space-x-2 shadow-sm`}
          >
            <Download className="w-4 h-4" />
            <span className="text-sm font-medium">Export</span>
          </motion.button>
        </div>
      </div>
      
      <AnimatePresence>
        {isFilterExpanded && (
          <motion.div 
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className={`${
              isDarkMode 
                ? 'bg-gray-800 border border-white/10' 
                : 'bg-gray-50 border border-gray-200'
            } rounded-lg p-4 mb-6 overflow-hidden shadow-sm`}
          >
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div>
                <label className={`block text-sm font-medium mb-2 ${
                  isDarkMode ? 'text-gray-300' : 'text-gray-700'
                }`}>
                  Start Date
                </label>
                <input
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                  className={`w-full ${
                    isDarkMode 
                      ? 'bg-gray-700 border-gray-600 text-white focus:border-purple-500' 
                      : 'bg-white border-gray-300 text-gray-800 focus:border-indigo-500'
                  } border rounded-lg px-3 py-2 focus:outline-none focus:ring-2 ${
                    isDarkMode ? 'focus:ring-purple-500/30' : 'focus:ring-indigo-500/40'
                  }`}
                />
              </div>
              
              <div>
                <label className={`block text-sm font-medium mb-2 ${
                  isDarkMode ? 'text-gray-300' : 'text-gray-700'
                }`}>
                  End Date
                </label>
                <input
                  type="date"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                  className={`w-full ${
                    isDarkMode 
                      ? 'bg-gray-700 border-gray-600 text-white focus:border-purple-500' 
                      : 'bg-white border-gray-300 text-gray-800 focus:border-indigo-500'
                  } border rounded-lg px-3 py-2 focus:outline-none focus:ring-2 ${
                    isDarkMode ? 'focus:ring-purple-500/30' : 'focus:ring-indigo-500/40'
                  }`}
                />
              </div>
              
              <div>
                <label className={`block text-sm font-medium mb-2 ${
                  isDarkMode ? 'text-gray-300' : 'text-gray-700'
                }`}>
                  Status
                </label>
                <select
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value)}
                  className={`w-full ${
                    isDarkMode 
                      ? 'bg-gray-700 border-gray-600 text-white focus:border-purple-500' 
                      : 'bg-white border-gray-300 text-gray-800 focus:border-indigo-500'
                  } border rounded-lg px-3 py-2 focus:outline-none focus:ring-2 ${
                    isDarkMode ? 'focus:ring-purple-500/30' : 'focus:ring-indigo-500/40'
                  }`}
                >
                  <option value="">All Statuses</option>
                  <option value="paid">Paid</option>
                  <option value="pending">Pending</option>
                  <option value="overdue">Overdue</option>
                  <option value="cancelled">Cancelled</option>
                </select>
              </div>
              
              <div>
                <label className={`block text-sm font-medium mb-2 ${
                  isDarkMode ? 'text-gray-300' : 'text-gray-700'
                }`}>
                  Subscription Type
                </label>
                <select
                  value={subscriptionFilter}
                  onChange={(e) => setSubscriptionFilter(e.target.value)}
                  className={`w-full ${
                    isDarkMode 
                      ? 'bg-gray-700 border-gray-600 text-white focus:border-purple-500' 
                      : 'bg-white border-gray-300 text-gray-800 focus:border-indigo-500'
                  } border rounded-lg px-3 py-2 focus:outline-none focus:ring-2 ${
                    isDarkMode ? 'focus:ring-purple-500/30' : 'focus:ring-indigo-500/40'
                  }`}
                >
                  <option value="">All Types</option>
                  <option value="free">Free</option>
                  <option value="basic">Basic</option>
                  <option value="premium">Premium</option>
                  <option value="enterprise">Enterprise</option>
                </select>
              </div>
            </div>
            
            <div className="mt-4">
              <label className={`block text-sm font-medium mb-2 ${
                isDarkMode ? 'text-gray-300' : 'text-gray-700'
              }`}>
                Filter by Username
              </label>
              <input
                type="text"
                value={usernameFilter}
                onChange={(e) => setUsernameFilter(e.target.value)}
                placeholder="Enter username"
                className={`w-full ${
                  isDarkMode 
                    ? 'bg-gray-700 border-gray-600 text-white focus:border-purple-500' 
                    : 'bg-white border-gray-300 text-gray-800 focus:border-indigo-500'
                } border rounded-lg px-3 py-2 focus:outline-none focus:ring-2 ${
                  isDarkMode ? 'focus:ring-purple-500/30' : 'focus:ring-indigo-500/40'
                }`}
              />
            </div>
            
            <div className="flex items-center mt-4 justify-end space-x-3">
              <button
                onClick={clearFilters}
                className={`text-sm ${
                  isDarkMode 
                    ? 'text-gray-400 hover:text-gray-300' 
                    : 'text-gray-600 hover:text-gray-800'
                }`}
              >
                Clear Filters
              </button>
              
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={applyFilters}
                className={`${
                  isDarkMode 
                    ? 'bg-purple-600 hover:bg-purple-700 text-white' 
                    : 'bg-indigo-600 hover:bg-indigo-700 text-white'
                } px-4 py-2 rounded-lg text-sm font-medium`}
              >
                Apply Filters
              </motion.button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
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
            Loading billing activity...
          </span>
        </div>
      ) : (
        <>
          {/* Summary Cards */}
          {billingData.summary && (
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
              {/* Total Billed Amount */}
              <motion.div
                whileHover={{ y: -4 }}
                className={`${
                  isDarkMode 
                    ? 'bg-gray-800 border border-white/10' 
                    : 'bg-white border border-gray-200'
                } rounded-xl p-5 shadow-lg`}
              >
                <div className="flex justify-between items-start mb-4">
                  <span className={`${
                    isDarkMode ? 'text-gray-400' : 'text-gray-500'
                  } text-sm`}>Total Billed</span>
                  <div className={`${
                    isDarkMode ? 'bg-green-900/30' : 'bg-green-100'
                  } p-2 rounded-lg`}>
                    <DollarSign className={`w-4 h-4 ${
                      isDarkMode ? 'text-green-400' : 'text-green-600'
                    }`} />
                  </div>
                </div>
                <div className={`text-2xl font-bold ${
                  isDarkMode ? 'text-white' : 'text-gray-800'
                }`}>
                  {formatCurrency(billingData.summary.total_billed_amount || 0)}
                </div>
              </motion.div>
              
              {/* Total Invoice Amount */}
              <motion.div
                whileHover={{ y: -4 }}
                className={`${
                  isDarkMode 
                    ? 'bg-gray-800 border border-white/10' 
                    : 'bg-white border border-gray-200'
                } rounded-xl p-5 shadow-lg`}
              >
                <div className="flex justify-between items-start mb-4">
                  <span className={`${
                    isDarkMode ? 'text-gray-400' : 'text-gray-500'
                  } text-sm`}>Total Invoiced</span>
                  <div className={`${
                    isDarkMode ? 'bg-blue-900/30' : 'bg-blue-100'
                  } p-2 rounded-lg`}>
                    <FileText className={`w-4 h-4 ${
                      isDarkMode ? 'text-blue-400' : 'text-blue-600'
                    }`} />
                  </div>
                </div>
                <div className={`text-2xl font-bold ${
                  isDarkMode ? 'text-white' : 'text-gray-800'
                }`}>
                  {formatCurrency(billingData.summary.total_invoice_amount || 0)}
                </div>
              </motion.div>
              
              {/* Total API Calls */}
              <motion.div
                whileHover={{ y: -4 }}
                className={`${
                  isDarkMode 
                    ? 'bg-gray-800 border border-white/10' 
                    : 'bg-white border border-gray-200'
                } rounded-xl p-5 shadow-lg`}
              >
                <div className="flex justify-between items-start mb-4">
                  <span className={`${
                    isDarkMode ? 'text-gray-400' : 'text-gray-500'
                  } text-sm`}>Total API Calls</span>
                  <div className={`${
                    isDarkMode ? 'bg-purple-900/30' : 'bg-purple-100'
                  } p-2 rounded-lg`}>
                    <Calendar className={`w-4 h-4 ${
                      isDarkMode ? 'text-purple-400' : 'text-purple-600'
                    }`} />
                  </div>
                </div>
                <div className={`text-2xl font-bold ${
                  isDarkMode ? 'text-white' : 'text-gray-800'
                }`}>
                  {billingData.summary.total_api_calls || 0}
                </div>
              </motion.div>
              
              {/* Active Users Count */}
              <motion.div
                whileHover={{ y: -4 }}
                className={`${
                  isDarkMode 
                    ? 'bg-gray-800 border border-white/10' 
                    : 'bg-white border border-gray-200'
                } rounded-xl p-5 shadow-lg`}
              >
                <div className="flex justify-between items-start mb-4">
                  <span className={`${
                    isDarkMode ? 'text-gray-400' : 'text-gray-500'
                  } text-sm`}>Active Users</span>
                  <div className={`${
                    isDarkMode ? 'bg-amber-900/30' : 'bg-amber-100'
                  } p-2 rounded-lg`}>
                    <User className={`w-4 h-4 ${
                      isDarkMode ? 'text-amber-400' : 'text-amber-600'
                    }`} />
                  </div>
                </div>
                <div className={`text-2xl font-bold ${
                  isDarkMode ? 'text-white' : 'text-gray-800'
                }`}>
                  {billingData.summary.active_users_count || 0}
                </div>
              </motion.div>
            </div>
          )}
          
          {/* Tab Switcher */}
          <div className="flex border-b mb-4 space-x-4 overflow-x-auto pb-1 scrollbar-hidden whitespace-nowrap">
            {['all', 'billing', 'invoices', 'usage'].map((tab) => (
              <button
                key={tab}
                onClick={() => setCurrentTab(tab)}
                className={`px-4 py-2 font-medium text-sm transition-colors ${
                  currentTab === tab 
                    ? isDarkMode
                      ? 'text-purple-400 border-b-2 border-purple-400'
                      : 'text-indigo-600 border-b-2 border-indigo-600'
                    : isDarkMode
                      ? 'text-gray-400 hover:text-gray-300'
                      : 'text-gray-600 hover:text-gray-800'
                }`}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </div>
          
          {/* Billing Table */}
          <div className={`${
            isDarkMode 
              ? 'bg-gray-800 border border-white/10' 
              : 'bg-white border border-gray-200'
          } rounded-xl overflow-hidden shadow-lg mb-6`}>
            <div className={`grid grid-cols-12 gap-4 px-6 py-4 border-b ${
              isDarkMode ? 'border-white/5 bg-black/20' : 'border-gray-200 bg-gray-50'
            }`}>
              <div className="col-span-2 md:col-span-3 font-medium text-xs uppercase tracking-wider">
                <span className={isDarkMode ? 'text-purple-400' : 'text-indigo-500'}>User</span>
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
            
            {renderActivityTab()}
          </div>
          
          {/* Pagination Controls */}
          {billingData.pagination && billingData.pagination.total_pages > 1 && (
            <div className="flex justify-between items-center py-4">
              <div className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                Page {currentPage} of {billingData.pagination.total_pages}
              </div>
              
              <div className="flex space-x-2">
                <button
                  onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                  disabled={currentPage === 1}
                  className={`p-2 rounded-lg ${
                    currentPage === 1
                      ? isDarkMode
                        ? 'bg-gray-800 text-gray-600 cursor-not-allowed'
                        : 'bg-gray-100 text-gray-400 cursor-not-allowed'
                      : isDarkMode
                        ? 'bg-gray-800 hover:bg-gray-700 text-white'
                        : 'bg-white hover:bg-gray-50 text-gray-700 border border-gray-200'
                  }`}
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>
                
                <button
                  onClick={() => setCurrentPage(Math.min(billingData.pagination.total_pages, currentPage + 1))}
                  disabled={currentPage === billingData.pagination.total_pages}
                  className={`p-2 rounded-lg ${
                    currentPage === billingData.pagination.total_pages
                      ? isDarkMode
                        ? 'bg-gray-800 text-gray-600 cursor-not-allowed'
                        : 'bg-gray-100 text-gray-400 cursor-not-allowed'
                      : isDarkMode
                        ? 'bg-gray-800 hover:bg-gray-700 text-white'
                        : 'bg-white hover:bg-gray-50 text-gray-700 border border-gray-200'
                  }`}
                >
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          )}
        </>
      )}
      
      {/* Invoice Detail Modal */}
      <AnimatePresence>
        {isInvoiceModalOpen && selectedInvoiceId && (
          <InvoiceDetailModal
            invoiceId={selectedInvoiceId}
            isDarkMode={isDarkMode}
            onClose={handleCloseInvoiceModal}
            onPaymentSuccess={handleInvoicePaymentSuccess}
            isAdminView={true}
          />
        )}
      </AnimatePresence>
    </div>
  );
};

export default AdminBillingActivity;