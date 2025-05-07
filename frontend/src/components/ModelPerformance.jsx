import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Loader2, ChevronDown, ChevronUp, AlertCircle, RefreshCw,
  BarChart as BarChartIcon, LineChart as LineChartIcon,
  PieChart as PieChartIcon, TrendingUp, Clock, 
  Shield, Download, Activity
} from 'lucide-react';
import { 
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, 
  Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell,
  Area, AreaChart
} from 'recharts';
import { useAuth } from '../context/AuthContext';

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
  }
};

const chartContainerVariant = {
  hidden: { opacity: 0, height: 0 },
  visible: { 
    opacity: 1, 
    height: 'auto',
    transition: { 
      duration: 0.4,
      ease: "easeInOut"
    }
  },
  exit: {
    opacity: 0,
    height: 0,
    transition: { 
      duration: 0.3,
      ease: "easeInOut"
    }
  }
};

const COLORS = ['#8b5cf6', '#ec4899', '#10b981', '#f59e0b', '#3b82f6', '#ef4444'];

const ModelPerformance = ({ isDarkMode }) => {
  const { token } = useAuth(); // Get token from auth context
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [metrics, setMetrics] = useState({
    summary: {
      avg_confidence: 0,
      avg_latency: 0,
      avg_frame_rate: 0,
      avg_stability: 0,
      total_entries: 0
    },
    by_workout_type: [],
    trend: []
  });
  const [timeRange, setTimeRange] = useState('week');
  const [workoutType, setWorkoutType] = useState('all');
  const [expandedSection, setExpandedSection] = useState('all');
  const [isRefreshing, setIsRefreshing] = useState(false);

  useEffect(() => {
    if (token) {
      fetchMetrics();
    }
  }, [timeRange, workoutType, token]);

  // Function to convert timeRange to actual dates
  const getDateRange = () => {
    const now = new Date();
    let startDate = null;
    
    switch(timeRange) {
      case 'day':
        startDate = new Date(now);
        startDate.setDate(startDate.getDate() - 1);
        break;
      case 'week':
        startDate = new Date(now);
        startDate.setDate(startDate.getDate() - 7);
        break;
      case 'month':
        startDate = new Date(now);
        startDate.setMonth(startDate.getMonth() - 1);
        break;
      case 'year':
        startDate = new Date(now);
        startDate.setFullYear(startDate.getFullYear() - 1);
        break;
      default:
        startDate = new Date(now);
        startDate.setDate(startDate.getDate() - 7); // default to week
    }
    
    return {
      start_date: startDate.toISOString().split('T')[0],
      end_date: now.toISOString().split('T')[0]
    };
  };

  const fetchMetrics = async () => {
    setLoading(true);
    setIsRefreshing(true);
    setError(null);
    try {
      // Get date ranges based on selected time range
      const dateRange = getDateRange();
      
      // Prepare query parameters
      const params = {
        ...dateRange
      };
      
      // Add workout type if specific one is selected
      if (workoutType !== 'all') {
        params.workout_type = workoutType;
      }
      
      const response = await axios.get('http://localhost:8000/api/model-performance/summary/', {
        params,
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      console.log('ModelPerformance API response:', response.data);
      setMetrics(response.data);
    } catch (err) {
      console.error('Error fetching model performance metrics:', err);
      setError('Failed to load performance metrics. Please try again later.');
    } finally {
      setLoading(false);
      // Set a slight delay for the refreshing indicator to make it smoother
      setTimeout(() => {
        setIsRefreshing(false);
      }, 300);
    }
  };

  const toggleSection = (section) => {
    setExpandedSection(expandedSection === section ? 'all' : section);
  };

  const formatTime = (ms) => {
    // Round the milliseconds to nearest integer
    ms = Math.round(ms);
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  };

  // Format workout type data for the bar chart
  const prepareWorkoutTypeData = () => {
    if (!metrics.by_workout_type || !Array.isArray(metrics.by_workout_type)) {
      return [];
    }
    
    return metrics.by_workout_type.map(item => {
      // Get workout name from workout type index
      let name = workoutMap[item.workout_type] || `Type ${item.workout_type}`;
      
      return {
        name: name,
        accuracy: item.avg_confidence,
        latency: item.avg_latency,
        count: item.count
      };
    });
  };

  // Format trend data for the line charts
  const prepareTrendData = () => {
    if (!metrics.trend || !Array.isArray(metrics.trend)) {
      return [];
    }
    
    return metrics.trend.map(item => {
      // Use the pre-formatted day value from the backend, or format timestamp if provided
      let formattedDate;
      
      if (item.day) {
        formattedDate = item.day;
      } else if (item.time_period) {
        // Convert ISO date to readable format
        const date = new Date(item.time_period);
        formattedDate = date.toLocaleTimeString(undefined, { 
          hour: '2-digit', 
          minute: '2-digit'
        });
      } else {
        // Fallback - should not happen with updated backend
        const date = new Date();
        formattedDate = date.toLocaleString();
      }
      
      return {
        name: formattedDate,
        // Store the timestamp for proper sorting if available
        timestamp: item.time_period_timestamp || null,
        accuracy: item.avg_confidence,
        latency: item.avg_latency,
        count: item.count
      };
    });
  };

  // Helper for workout type names
  const workoutMap = { 
    0: "Barbell Bicep Curl", 
    1: "Bench Press", 
    2: "Chest Fly Machine", 
    3: "Deadlift",
    4: "Decline Bench Press", 
    5: "Hammer Curl", 
    6: "Hip Thrust", 
    7: "Incline Bench Press", 
    8: "Lat Pulldown", 
    9: "Lateral Raises", 
    10: "Leg Extensions", 
    11: "Leg Raises",
    12: "Plank", 
    13: "Pull Up", 
    14: "Push Ups", 
    15: "Romanian Deadlift", 
    16: "Russian Twist", 
    17: "Shoulder Press", 
    18: "Squat", 
    19: "T Bar Row", 
    20: "Tricep Dips", 
    21: "Tricep Pushdown"
  };

  // Render loading state with animation
  if (loading && !isRefreshing) {
    return (
      <motion.div 
        className="flex flex-col items-center justify-center py-20"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
      >
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
          className={`w-12 h-12 rounded-full border-t-2 border-b-2 ${
            isDarkMode ? 'border-purple-400' : 'border-purple-600'
          } mb-4`}
        />
        <p className={isDarkMode ? 'text-gray-300' : 'text-gray-600'}>
          Loading model performance metrics...
        </p>
      </motion.div>
    );
  }

  // Render error state with animation
  if (error) {
    return (
      <motion.div 
        className={`p-6 rounded-xl ${
          isDarkMode 
            ? 'bg-red-900/20 backdrop-blur-md border border-red-800/30' 
            : 'bg-red-50 border border-red-200'
        }`}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="flex items-center text-red-500">
          <AlertCircle className="w-5 h-5 mr-2 flex-shrink-0" />
          <span>{error}</span>
        </div>
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={fetchMetrics}
          className={`mt-4 px-4 py-2 rounded-lg flex items-center space-x-2 ${
            isDarkMode 
              ? 'bg-red-500/20 hover:bg-red-500/30 text-red-300' 
              : 'bg-red-100 hover:bg-red-200 text-red-700'
          }`}
        >
          <RefreshCw className="w-4 h-4" />
          <span>Try Again</span>
        </motion.button>
      </motion.div>
    );
  }

  // If we don't have any data, show a message
  const workoutTypeData = prepareWorkoutTypeData();
  const trendData = prepareTrendData();
  
  if ((!metrics.summary || metrics.summary.total_entries === 0) && 
      workoutTypeData.length === 0 && trendData.length === 0) {
    return (
      <motion.div 
        className="space-y-8"
        initial="hidden"
        animate="visible"
        variants={fadeIn}
      >
        <div className="flex flex-col md:flex-row justify-between space-y-4 md:space-y-0 md:items-center">
          <h1 className={`text-2xl font-light ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>
            Model <span className="text-purple-500 font-medium">Performance Analytics</span>
          </h1>
          
          <div className="flex items-center space-x-4">
            <div className="flex space-x-4">
              <select 
                className={`px-3 py-2 rounded-lg ${
                  isDarkMode 
                    ? 'bg-gray-800/80 backdrop-blur-md border border-white/5 text-white' 
                    : 'bg-white border border-gray-200 text-gray-700'
                }`}
                value={timeRange}
                onChange={(e) => setTimeRange(e.target.value)}
              >
                <option value="day">Last 24 Hours</option>
                <option value="week">Last Week</option>
                <option value="month">Last Month</option>
                <option value="year">Last Year</option>
              </select>
              
              <select 
                className={`px-3 py-2 rounded-lg ${
                  isDarkMode 
                    ? 'bg-gray-800/80 backdrop-blur-md border border-white/5 text-white' 
                    : 'bg-white border border-gray-200 text-gray-700'
                }`}
                value={workoutType}
                onChange={(e) => setWorkoutType(e.target.value)}
              >
                <option value="all">All Workouts</option>
                <option value="0">Barbell Bicep Curl</option>
                <option value="1">Bench Press</option>
                <option value="12">Plank</option>
                <option value="18">Squat</option>
              </select>
            </div>
            
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={fetchMetrics}
              className={`p-2 rounded-lg ${
                isDarkMode 
                  ? 'bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 border border-purple-500/30' 
                  : 'bg-purple-100 hover:bg-purple-200 text-purple-700 border border-purple-200'
              } transition-colors duration-200 shadow-sm`}
              title="Refresh Data"
            >
              <RefreshCw className="w-5 h-5" />
            </motion.button>
          </div>
        </div>
        
        <motion.div 
          className={`p-6 rounded-xl ${
            isDarkMode 
              ? 'bg-blue-900/20 backdrop-blur-md border border-blue-800/30' 
              : 'bg-blue-50 border border-blue-200'
          }`}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="flex items-center text-blue-500">
            <AlertCircle className="w-5 h-5 mr-2 flex-shrink-0" />
            <span>No performance data available for the selected filters. Try changing the time range or workout type.</span>
          </div>
        </motion.div>
      </motion.div>
    );
  }

  return (
    <motion.div 
      className="space-y-8"
      initial="hidden"
      animate="visible"
      variants={staggerContainer}
    >
      <motion.div 
        className="flex flex-col md:flex-row justify-between space-y-4 md:space-y-0 md:items-center"
        variants={fadeIn}
      >
        <h1 className={`text-2xl font-light ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>
          Model <span className="text-purple-500 font-medium">Performance Analytics</span>
        </h1>
        
        <div className="flex items-center space-x-4">
          <div className="flex flex-col sm:flex-row space-y-3 sm:space-y-0 sm:space-x-4">
            <select 
              className={`px-3 py-2 rounded-lg transition-colors ${
                isDarkMode 
                  ? 'bg-gray-800/80 backdrop-blur-md border border-white/5 text-white hover:bg-gray-700/80' 
                  : 'bg-white border border-gray-200 text-gray-700 hover:bg-gray-50'
              } shadow-sm`}
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
            >
              <option value="day">Last 24 Hours</option>
              <option value="week">Last Week</option>
              <option value="month">Last Month</option>
              <option value="year">Last Year</option>
            </select>
            
            <select 
              className={`px-3 py-2 rounded-lg transition-colors ${
                isDarkMode 
                  ? 'bg-gray-800/80 backdrop-blur-md border border-white/5 text-white hover:bg-gray-700/80' 
                  : 'bg-white border border-gray-200 text-gray-700 hover:bg-gray-50'
              } shadow-sm`}
              value={workoutType}
              onChange={(e) => setWorkoutType(e.target.value)}
            >
              <option value="all">All Workouts</option>
              {Object.entries(workoutMap).map(([id, name]) => (
                <option key={id} value={id}>{name}</option>
              ))}
            </select>
          </div>
          
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={fetchMetrics}
            className={`p-2 rounded-lg shadow-sm ${
              isDarkMode 
                ? 'bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 border border-purple-500/30' 
                : 'bg-purple-100 hover:bg-purple-200 text-purple-700 border border-purple-200'
            } transition-colors duration-200`}
            title="Refresh Data"
            disabled={isRefreshing}
          >
            <RefreshCw className={`w-5 h-5 ${isRefreshing ? 'animate-spin' : ''}`} />
          </motion.button>
        </div>
      </motion.div>

      {/* Summary Cards */}
      <motion.div 
        className="grid grid-cols-1 md:grid-cols-3 gap-5"
        variants={staggerContainer}
      >
        <motion.div 
          className={`p-5 rounded-xl ${
            isDarkMode 
              ? 'bg-gray-800/80 backdrop-blur-md border border-white/5' 
              : 'bg-white backdrop-blur-md border border-gray-100'
          } shadow-lg overflow-hidden relative`}
          variants={cardVariant}
          whileHover="hover"
        >
          <div className="absolute top-0 right-0 w-24 h-24 bg-purple-500/5 rounded-full -mr-10 -mt-10"></div>
          <div className="flex items-center mb-3">
            <Activity className={`w-5 h-5 mr-2 ${
              isDarkMode ? 'text-purple-400' : 'text-purple-600'
            }`} />
            <h3 className={`text-sm uppercase font-medium ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              Average Confidence
            </h3>
          </div>
          <div className="flex items-baseline">
            <span className={`text-3xl font-light ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
              {metrics.summary ? (metrics.summary.avg_confidence * 100).toFixed(1) : 0}%
            </span>
            <span className={`ml-2 text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              accuracy
            </span>
          </div>
          <div className="mt-4 relative z-10">
            <div className={`w-full ${
              isDarkMode ? 'bg-gray-700' : 'bg-gray-200'
            } rounded-full h-1.5 overflow-hidden`}>
              <div 
                className={`${isDarkMode ? 'bg-purple-500' : 'bg-purple-600'} h-1.5 rounded-full`}
                style={{ width: `${(metrics.summary?.avg_confidence || 0) * 100}%` }}
              ></div>
            </div>
          </div>
        </motion.div>

        <motion.div 
          className={`p-5 rounded-xl ${
            isDarkMode 
              ? 'bg-gray-800/80 backdrop-blur-md border border-white/5' 
              : 'bg-white backdrop-blur-md border border-gray-100'
          } shadow-lg overflow-hidden relative`}
          variants={cardVariant}
          whileHover="hover"
        >
          <div className="absolute top-0 right-0 w-24 h-24 bg-pink-500/5 rounded-full -mr-10 -mt-10"></div>
          <div className="flex items-center mb-3">
            <Clock className={`w-5 h-5 mr-2 ${
              isDarkMode ? 'text-pink-400' : 'text-pink-600'
            }`} />
            <h3 className={`text-sm uppercase font-medium ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              Average Response Time
            </h3>
          </div>
          <div className="flex items-baseline">
            <span className={`text-3xl font-light ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
              {metrics.summary ? formatTime(Math.round(metrics.summary.avg_latency)) : '0ms'}
            </span>
            <span className={`ml-2 text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              latency
            </span>
          </div>
          <div className="mt-4 relative z-10">
            <div className={`w-full ${
              isDarkMode ? 'bg-gray-700' : 'bg-gray-200'
            } rounded-full h-1.5 overflow-hidden`}>
              <div 
                className={`${isDarkMode ? 'bg-pink-500' : 'bg-pink-600'} h-1.5 rounded-full`}
                style={{ 
                  width: `${Math.min((Math.round(metrics.summary?.avg_latency) || 0) / 200 * 100, 100)}%` 
                }}
              ></div>
            </div>
          </div>
        </motion.div>

        <motion.div 
          className={`p-5 rounded-xl ${
            isDarkMode 
              ? 'bg-gray-800/80 backdrop-blur-md border border-white/5' 
              : 'bg-white backdrop-blur-md border border-gray-100'
          } shadow-lg overflow-hidden relative`}
          variants={cardVariant}
          whileHover="hover"
        >
          <div className="absolute top-0 right-0 w-24 h-24 bg-green-500/5 rounded-full -mr-10 -mt-10"></div>
          <div className="flex items-center mb-3">
            <Shield className={`w-5 h-5 mr-2 ${
              isDarkMode ? 'text-green-400' : 'text-green-600'
            }`} />
            <h3 className={`text-sm uppercase font-medium ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              Model Stability
            </h3>
          </div>
          <div className="flex items-baseline">
            <span className={`text-3xl font-light ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
              {metrics.summary ? (metrics.summary.avg_stability * 100).toFixed(1) : 0}%
            </span>
            <span className={`ml-2 text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              consistency
            </span>
          </div>
          <div className="mt-4 relative z-10">
            <div className={`w-full ${
              isDarkMode ? 'bg-gray-700' : 'bg-gray-200'
            } rounded-full h-1.5 overflow-hidden`}>
              <div 
                className={`${isDarkMode ? 'bg-green-500' : 'bg-green-600'} h-1.5 rounded-full`}
                style={{ width: `${(metrics.summary?.avg_stability || 0) * 100}%` }}
              ></div>
            </div>
          </div>
        </motion.div>
      </motion.div>

      {/* Accuracy Over Time */}
      <motion.div 
        className={`rounded-xl ${
          isDarkMode 
            ? 'bg-gray-800/80 backdrop-blur-md border border-white/5' 
            : 'bg-white backdrop-blur-md border border-gray-100'
        } shadow-lg overflow-hidden`}
        variants={fadeIn}
      >
        <div className="p-5">
          <motion.div 
            className="flex justify-between items-center cursor-pointer" 
            onClick={() => toggleSection('accuracy')}
            whileHover={{ x: 5 }}
          >
            <div className="flex items-center">
              <LineChartIcon className={`w-5 h-5 mr-2 ${
                isDarkMode ? 'text-purple-400' : 'text-purple-600'
              }`} />
              <h2 className={`text-lg font-medium ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>
                Prediction Accuracy Over Time
              </h2>
            </div>
            <motion.div
              whileHover={{ scale: 1.2 }}
              whileTap={{ scale: 0.9 }}
            >
              {expandedSection === 'accuracy' ? 
                <ChevronUp className={`w-5 h-5 ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-600'
                }`} /> : 
                <ChevronDown className={`w-5 h-5 ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-600'
                }`} />
              }
            </motion.div>
          </motion.div>
        </div>
        
        <AnimatePresence initial={false}>
          {(expandedSection === 'accuracy' || expandedSection === 'all') && trendData.length > 0 && (
            <motion.div 
              className="px-5 pb-5"
              variants={chartContainerVariant}
              initial="hidden"
              animate="visible"
              exit="exit"
            >
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart
                    data={trendData.sort((a, b) => {
                      // Sort by timestamp if available, otherwise by name
                      if (a.timestamp && b.timestamp) {
                        return a.timestamp - b.timestamp;
                      }
                      return a.name.localeCompare(b.name);
                    })}
                    margin={{ top: 5, right: 30, left: 20, bottom: 20 }}
                  >
                    <defs>
                      <linearGradient id="accuracyGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0.1}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid 
                      strokeDasharray="3 3" 
                      stroke={isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)'} 
                      vertical={false} 
                    />
                    <XAxis 
                      dataKey="name" 
                      stroke={isDarkMode ? 'rgba(255, 255, 255, 0.6)' : 'rgba(0, 0, 0, 0.6)'} 
                      tick={{ fill: isDarkMode ? 'rgba(255, 255, 255, 0.8)' : 'rgba(0, 0, 0, 0.8)' }}
                      angle={-30}
                      textAnchor="end"
                      height={60}
                      minTickGap={10}
                    />
                    <YAxis 
                      stroke={isDarkMode ? 'rgba(255, 255, 255, 0.6)' : 'rgba(0, 0, 0, 0.6)'} 
                      tick={{ fill: isDarkMode ? 'rgba(255, 255, 255, 0.8)' : 'rgba(0, 0, 0, 0.8)' }} 
                      domain={[0, 1]}
                      tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: isDarkMode ? 'rgba(31, 41, 55, 0.9)' : 'rgba(255, 255, 255, 0.9)',
                        borderColor: isDarkMode ? 'rgba(55, 65, 81, 0.5)' : 'rgba(229, 231, 235, 0.8)',
                        borderRadius: '0.5rem',
                        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
                        color: isDarkMode ? '#fff' : '#000'
                      }}
                      formatter={(value) => [`${(value * 100).toFixed(1)}%`, 'Accuracy']}
                      labelFormatter={(label) => `Time: ${label}`}
                    />
                    <Legend />
                    <Area 
                      type="monotone" 
                      dataKey="accuracy" 
                      stroke="#8b5cf6" 
                      fillOpacity={1}
                      fill="url(#accuracyGradient)"
                      strokeWidth={2}
                      dot={{ r: 4, fill: '#8b5cf6', strokeWidth: 2, stroke: isDarkMode ? '#1f2937' : '#ffffff' }}
                      activeDot={{ r: 6, fill: '#8b5cf6', stroke: isDarkMode ? '#1f2937' : '#ffffff', strokeWidth: 2 }}
                      connectNulls={true}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* Latency Over Time */}
      <motion.div 
        className={`rounded-xl ${
          isDarkMode 
            ? 'bg-gray-800/80 backdrop-blur-md border border-white/5' 
            : 'bg-white backdrop-blur-md border border-gray-100'
        } shadow-lg overflow-hidden`}
        variants={fadeIn}
      >
        <div className="p-5">
          <motion.div 
            className="flex justify-between items-center cursor-pointer" 
            onClick={() => toggleSection('latency')}
            whileHover={{ x: 5 }}
          >
            <div className="flex items-center">
              <TrendingUp className={`w-5 h-5 mr-2 ${
                isDarkMode ? 'text-pink-400' : 'text-pink-600'
              }`} />
              <h2 className={`text-lg font-medium ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>
                Response Latency Over Time
              </h2>
            </div>
            <motion.div
              whileHover={{ scale: 1.2 }}
              whileTap={{ scale: 0.9 }}
            >
              {expandedSection === 'latency' ? 
                <ChevronUp className={`w-5 h-5 ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-600'
                }`} /> : 
                <ChevronDown className={`w-5 h-5 ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-600'
                }`} />
              }
            </motion.div>
          </motion.div>
        </div>
        
        <AnimatePresence initial={false}>
          {(expandedSection === 'latency' || expandedSection === 'all') && trendData.length > 0 && (
            <motion.div 
              className="px-5 pb-5"
              variants={chartContainerVariant}
              initial="hidden"
              animate="visible"
              exit="exit"
            >
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart
                    data={trendData.sort((a, b) => {
                      // Sort by timestamp if available, otherwise by name
                      if (a.timestamp && b.timestamp) {
                        return a.timestamp - b.timestamp;
                      }
                      return a.name.localeCompare(b.name);
                    })}
                    margin={{ top: 5, right: 30, left: 20, bottom: 20 }}
                  >
                    <defs>
                      <linearGradient id="latencyGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#ec4899" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#ec4899" stopOpacity={0.1}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid 
                      strokeDasharray="3 3" 
                      stroke={isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)'} 
                      vertical={false} 
                    />
                    <XAxis 
                      dataKey="name" 
                      stroke={isDarkMode ? 'rgba(255, 255, 255, 0.6)' : 'rgba(0, 0, 0, 0.6)'} 
                      tick={{ fill: isDarkMode ? 'rgba(255, 255, 255, 0.8)' : 'rgba(0, 0, 0, 0.8)' }}
                      angle={-30}
                      textAnchor="end"
                      height={60}
                      minTickGap={10}
                    />
                    <YAxis 
                      stroke={isDarkMode ? 'rgba(255, 255, 255, 0.6)' : 'rgba(0, 0, 0, 0.6)'} 
                      tick={{ fill: isDarkMode ? 'rgba(255, 255, 255, 0.8)' : 'rgba(0, 0, 0, 0.8)' }} 
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: isDarkMode ? 'rgba(31, 41, 55, 0.9)' : 'rgba(255, 255, 255, 0.9)',
                        borderColor: isDarkMode ? 'rgba(55, 65, 81, 0.5)' : 'rgba(229, 231, 235, 0.8)',
                        borderRadius: '0.5rem',
                        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
                        color: isDarkMode ? '#fff' : '#000'
                      }}
                      formatter={(value) => [`${Math.round(value)}ms`, 'Latency']}
                      labelFormatter={(label) => `Time: ${label}`}
                    />
                    <Legend />
                    <Area 
                      type="monotone" 
                      dataKey="latency" 
                      stroke="#ec4899" 
                      fillOpacity={1}
                      fill="url(#latencyGradient)"
                      strokeWidth={2}
                      dot={{ r: 4, fill: '#ec4899', strokeWidth: 2, stroke: isDarkMode ? '#1f2937' : '#ffffff' }}
                      activeDot={{ r: 6, fill: '#ec4899', stroke: isDarkMode ? '#1f2937' : '#ffffff', strokeWidth: 2 }}
                      connectNulls={true}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* Accuracy by Workout Type */}
      <motion.div 
        className={`rounded-xl ${
          isDarkMode 
            ? 'bg-gray-800/80 backdrop-blur-md border border-white/5' 
            : 'bg-white backdrop-blur-md border border-gray-100'
        } shadow-lg overflow-hidden`}
        variants={fadeIn}
      >
        <div className="p-5">
          <motion.div 
            className="flex justify-between items-center cursor-pointer" 
            onClick={() => toggleSection('workout')}
            whileHover={{ x: 5 }}
          >
            <div className="flex items-center">
              <BarChartIcon className={`w-5 h-5 mr-2 ${
                isDarkMode ? 'text-green-400' : 'text-green-600'
              }`} />
              <h2 className={`text-lg font-medium ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>
                Performance by Workout Type
              </h2>
            </div>
            <motion.div
              whileHover={{ scale: 1.2 }}
              whileTap={{ scale: 0.9 }}
            >
              {expandedSection === 'workout' ? 
                <ChevronUp className={`w-5 h-5 ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-600'
                }`} /> : 
                <ChevronDown className={`w-5 h-5 ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-600'
                }`} />
              }
            </motion.div>
          </motion.div>
        </div>
        
        <AnimatePresence initial={false}>
          {(expandedSection === 'workout' || expandedSection === 'all') && workoutTypeData.length > 0 && (
            <motion.div 
              className="px-5 pb-5"
              variants={chartContainerVariant}
              initial="hidden"
              animate="visible"
              exit="exit"
            >
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={workoutTypeData}
                    margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    layout="vertical"
                  >
                    <defs>
                      <linearGradient id="performanceGradient" x1="0" y1="0" x2="1" y2="0">
                        <stop offset="0%" stopColor="#10b981" stopOpacity={0.8}/>
                        <stop offset="100%" stopColor="#059669" stopOpacity={1}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid 
                      strokeDasharray="3 3" 
                      stroke={isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)'} 
                      horizontal={true}
                      vertical={false}
                    />
                    <XAxis 
                      type="number" 
                      domain={[0, 1]}
                      tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                      stroke={isDarkMode ? 'rgba(255, 255, 255, 0.6)' : 'rgba(0, 0, 0, 0.6)'} 
                      tick={{ fill: isDarkMode ? 'rgba(255, 255, 255, 0.8)' : 'rgba(0, 0, 0, 0.8)' }} 
                    />
                    <YAxis 
                      type="category"
                      dataKey="name" 
                      stroke={isDarkMode ? 'rgba(255, 255, 255, 0.6)' : 'rgba(0, 0, 0, 0.6)'} 
                      tick={{ fill: isDarkMode ? 'rgba(255, 255, 255, 0.8)' : 'rgba(0, 0, 0, 0.8)' }} 
                      width={150}
                      tickFormatter={(value) => {
                        // Shorten long workout names
                        return value.length > 20 ? value.substring(0, 20) + '...' : value;
                      }}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: isDarkMode ? 'rgba(31, 41, 55, 0.9)' : 'rgba(255, 255, 255, 0.9)',
                        borderColor: isDarkMode ? 'rgba(55, 65, 81, 0.5)' : 'rgba(229, 231, 235, 0.8)',
                        borderRadius: '0.5rem',
                        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
                        color: isDarkMode ? '#fff' : '#000'
                      }}
                      formatter={(value) => [`${(value * 100).toFixed(1)}%`, 'Accuracy']}
                    />
                    <Legend />
                    <Bar 
                      dataKey="accuracy" 
                      name="Accuracy"
                      fill="url(#performanceGradient)" 
                      radius={[0, 4, 4, 0]}
                      animationDuration={1000}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* Usage Count by Workout Type */}
      <motion.div 
        className={`rounded-xl ${
          isDarkMode 
            ? 'bg-gray-800/80 backdrop-blur-md border border-white/5' 
            : 'bg-white backdrop-blur-md border border-gray-100'
        } shadow-lg overflow-hidden`}
        variants={fadeIn}
      >
        <div className="p-5">
          <motion.div 
            className="flex justify-between items-center cursor-pointer" 
            onClick={() => toggleSection('distribution')}
            whileHover={{ x: 5 }}
          >
            <div className="flex items-center">
              <PieChartIcon className={`w-5 h-5 mr-2 ${
                isDarkMode ? 'text-blue-400' : 'text-blue-600'
              }`} />
              <h2 className={`text-lg font-medium ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>
                Usage by Workout Type
              </h2>
            </div>
            <motion.div
              whileHover={{ scale: 1.2 }}
              whileTap={{ scale: 0.9 }}
            >
              {expandedSection === 'distribution' ? 
                <ChevronUp className={`w-5 h-5 ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-600'
                }`} /> : 
                <ChevronDown className={`w-5 h-5 ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-600'
                }`} />
              }
            </motion.div>
          </motion.div>
        </div>
        
        <AnimatePresence initial={false}>
          {(expandedSection === 'distribution' || expandedSection === 'all') && workoutTypeData.length > 0 && (
            <motion.div 
              className="px-5 pb-5"
              variants={chartContainerVariant}
              initial="hidden"
              animate="visible"
              exit="exit"
            >
              <div className="h-96">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={workoutTypeData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      outerRadius={130}
                      innerRadius={45}
                      paddingAngle={2}
                      fill="#8884d8"
                      dataKey="count"
                      nameKey="name"
                      label={({
                        cx,
                        cy,
                        midAngle,
                        innerRadius,
                        outerRadius,
                        percent,
                        index,
                        name
                      }) => {
                        const RADIAN = Math.PI / 180;
                        // Calculate radius for label placement
                        const radius = outerRadius * 1.1;
                        // Calculate positioning
                        const x = cx + radius * Math.cos(-midAngle * RADIAN);
                        const y = cy + radius * Math.sin(-midAngle * RADIAN);
                        
                        // Only show label for segments that are significant enough (e.g., > 3%)
                        if (percent < 0.03) return null;
                        
                        // Truncate long workout names
                        const shortName = name.length > 12 ? name.substring(0, 10) + '...' : name;
                          
                        return (
                          <text
                            x={x}
                            y={y}
                            fill={isDarkMode ? 'white' : '#333'}
                            textAnchor={x > cx ? 'start' : 'end'}
                            dominantBaseline="central"
                            fontSize={12}
                          >
                            {`${shortName} (${(percent * 100).toFixed(0)}%)`}
                          </text>
                        );
                      }}
                    >
                      {workoutTypeData.map((entry, index) => (
                        <Cell 
                          key={`cell-${index}`} 
                          fill={COLORS[index % COLORS.length]} 
                          stroke={isDarkMode ? '#1f2937' : '#fff'} 
                          strokeWidth={2}
                        />
                      ))}
                    </Pie>
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: isDarkMode ? 'rgba(31, 41, 55, 0.9)' : 'rgba(255, 255, 255, 0.9)',
                        borderColor: isDarkMode ? 'rgba(55, 65, 81, 0.5)' : 'rgba(229, 231, 235, 0.8)',
                        borderRadius: '0.5rem',
                        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
                        color: isDarkMode ? '#fff' : '#000'
                      }}
                      formatter={(value, name) => [`${value} entries`, name]}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              
              {/* Export button */}
              <div className="flex justify-end mt-2">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className={`flex items-center space-x-1 px-3 py-1.5 text-sm rounded-lg ${
                    isDarkMode 
                      ? 'bg-gray-700 hover:bg-gray-600 text-white' 
                      : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
                  }`}
                >
                  <Download className="w-4 h-4" />
                  <span>Export Data</span>
                </motion.button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </motion.div>
  );
};

export default ModelPerformance;