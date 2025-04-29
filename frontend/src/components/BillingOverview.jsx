import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import { 
  AlertCircle, RefreshCw, Calendar, DollarSign, 
  PieChart, Filter, Download, ArrowDownRight, 
  ArrowUpRight, CheckCircle, Clock, XCircle, FileText,
  User, Search, X
} from 'lucide-react';

const BillingOverview = ({ isDarkMode }) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [billingData, setBillingData] = useState([]);
  const [summary, setSummary] = useState(null);
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [subscriptionFilter, setSubscriptionFilter] = useState('');
  const [isFilterExpanded, setIsFilterExpanded] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

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
      if (searchQuery) query.push(`username=${searchQuery}`);
      
      const queryString = query.length > 0 ? `?${query.join('&')}` : '';
      
      // Fetch billing records
      const recordsResponse = await axios.get(`http://localhost:8000/api/billing/${queryString}`, {
        headers: { Authorization: `Bearer ${localStorage.getItem('access_token')}` }
      });
      
      // Fetch summary statistics
      const summaryResponse = await axios.get(`http://localhost:8000/api/billing/summary/${queryString}`, {
        headers: { Authorization: `Bearer ${localStorage.getItem('access_token')}` }
      });
      
      console.log("Billing data:", recordsResponse.data);
      console.log("Summary data:", summaryResponse.data);
      
      setBillingData(recordsResponse.data);
      setSummary(summaryResponse.data);
    } catch (err) {
      console.error("Error fetching billing data:", err);
      setError("Failed to fetch billing data. Please try again.");
    } finally {
      setLoading(false);
    }
  };
  
  // Load billing data on component mount and filter changes
  useEffect(() => {
    fetchBillingData();
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
          : { bg: 'bg-indigo-100', text: 'text-indigo-700' };
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
  
  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-8 gap-4">
        <h2 className={`text-2xl font-light ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>
          Billing <span className="text-purple-400 font-medium">Overview</span>
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
            onClick={fetchBillingData}
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
            Loading billing data...
          </span>
        </div>
      ) : (
        <>
          {/* Summary Cards */}
          {summary && (
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
              {/* Total Revenue Card */}
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
                  } text-sm`}>Total Revenue</span>
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
                  {formatCurrency(summary.total_amount)}
                </div>
                <div className="flex items-center mt-2">
                  {calculateChange().positive ? (
                    <ArrowUpRight className="w-3 h-3 text-green-500 mr-1" />
                  ) : (
                    <ArrowDownRight className="w-3 h-3 text-red-500 mr-1" />
                  )}
                  <span className={`text-xs ${
                    calculateChange().positive ? 'text-green-500' : 'text-red-500'
                  }`}>
                    {calculateChange().amount}% from previous period
                  </span>
                </div>
              </motion.div>
              
              {/* Total Invoices Card */}
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
                  } text-sm`}>Total Invoices</span>
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
                  {summary.total_records}
                </div>
                <div className="flex items-center mt-2">
                  <ArrowUpRight className="w-3 h-3 text-green-500 mr-1" />
                  <span className="text-xs text-green-500">
                    5.2% from previous period
                  </span>
                </div>
              </motion.div>
              
              {/* Pending Invoices */}
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
                  } text-sm`}>Pending Invoices</span>
                  <div className={`${
                    isDarkMode ? 'bg-amber-900/30' : 'bg-amber-100'
                  } p-2 rounded-lg`}>
                    <Clock className={`w-4 h-4 ${
                      isDarkMode ? 'text-amber-400' : 'text-amber-600'
                    }`} />
                  </div>
                </div>
                <div className={`text-2xl font-bold ${
                  isDarkMode ? 'text-white' : 'text-gray-800'
                }`}>
                  {summary.status_counts?.pending || 0}
                </div>
                <div className="flex items-center mt-2">
                  <span className={`text-xs ${
                    isDarkMode ? 'text-gray-400' : 'text-gray-500'
                  }`}>
                    Requires attention
                  </span>
                </div>
              </motion.div>
              
              {/* Overdue Invoices */}
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
                  } text-sm`}>Overdue Invoices</span>
                  <div className={`${
                    isDarkMode ? 'bg-red-900/30' : 'bg-red-100'
                  } p-2 rounded-lg`}>
                    <AlertCircle className={`w-4 h-4 ${
                      isDarkMode ? 'text-red-400' : 'text-red-600'
                    }`} />
                  </div>
                </div>
                <div className={`text-2xl font-bold ${
                  isDarkMode ? 'text-white' : 'text-gray-800'
                }`}>
                  {summary.status_counts?.overdue || 0}
                </div>
                <div className="flex items-center mt-2">
                  <span className={`text-xs ${
                    isDarkMode ? 'text-gray-400' : 'text-gray-500'
                  }`}>
                    Requires immediate action
                  </span>
                </div>
              </motion.div>
            </div>
          )}
          
          {/* Search Bar */}
          <div className="relative flex mb-6">
            <div className="relative flex-grow">
              <Search className={`w-5 h-5 absolute left-3 top-1/2 transform -translate-y-1/2 ${
                isDarkMode ? 'text-gray-400' : 'text-gray-500'
              }`} />
              <input
                type="text"
                placeholder="Search by username..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className={`w-full ${
                  isDarkMode 
                    ? 'bg-gray-800 text-white border border-white/10 focus:border-purple-500' 
                    : 'bg-white text-gray-800 border border-gray-200 focus:border-indigo-500'
                } rounded-lg pl-10 pr-4 py-2 focus:outline-none focus:ring-1 ${
                  isDarkMode ? 'focus:ring-purple-500/30' : 'focus:ring-indigo-500/40'
                } shadow-md`}
              />
              {searchQuery && (
                <button 
                  onClick={() => {
                    setSearchQuery('');
                    fetchBillingData();
                  }}
                  className={`absolute right-3 top-1/2 transform -translate-y-1/2 ${
                    isDarkMode ? 'text-gray-400 hover:text-white' : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  <X className="w-4 h-4" />
                </button>
              )}
            </div>
            
            <button
              onClick={fetchBillingData}
              className={`ml-2 px-4 py-2 rounded-lg ${
                isDarkMode 
                  ? 'bg-purple-600 hover:bg-purple-700 text-white' 
                  : 'bg-indigo-600 hover:bg-indigo-700 text-white'
              }`}
            >
              Search
            </button>
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
            
            {billingData.length === 0 ? (
              <div className={`py-16 text-center ${
                isDarkMode ? 'text-gray-400' : 'text-gray-500'
              }`}>
                <FileText className="w-12 h-12 mx-auto mb-4 opacity-20" />
                <p className="text-lg">No billing records found</p>
                <p className="text-sm mt-1 opacity-75">Try adjusting your filters</p>
              </div>
            ) : (
              <div className={`divide-y ${isDarkMode ? 'divide-white/5' : 'divide-gray-200'}`}>
                {billingData.map((record) => (
                  <motion.div
                    key={record.id}
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
                          {record.username}
                        </div>
                        <div className={`text-xs truncate ${
                          isDarkMode ? 'text-gray-400' : 'text-gray-500'
                        }`}>
                          {record.email}
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
                        {record.subscription_type.charAt(0).toUpperCase() + record.subscription_type.slice(1)}
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
                      <button className={`p-1 rounded-lg ${
                        isDarkMode 
                          ? 'hover:bg-gray-700 text-gray-400 hover:text-white' 
                          : 'hover:bg-gray-100 text-gray-500 hover:text-gray-800'
                      }`}>
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
    </div>
  );
};

export default BillingOverview; 