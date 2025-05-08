import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import { 
  CheckCircle, XCircle, User, Mail, Calendar,
  Shield, UserCheck, AlertCircle, RefreshCw
} from 'lucide-react';
import { API_URL } from '../config';

const ApprovalRequests = ({ isDarkMode }) => {
  const [pendingUsers, setPendingUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [processingUser, setProcessingUser] = useState(null);
  
  // Get the role name from user object
  const getUserRole = (user) => {
    if (!user || !user.groups || !user.groups.length) return "No Role";
    return user.groups[0].name;
  };
  
  // Check if a user is an AI Engineer
  const isAIEngineer = (user) => {
    const role = getUserRole(user);
    return role === 'AI Engineer';
  };
  
  const fetchPendingUsers = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const token = localStorage.getItem('access_token');
      const response = await axios.get(`${API_URL}/api/approvals/pending/`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      
      console.log("All pending approval users:", response.data);
      
      // Filter to only show AI Engineers
      const aiEngineers = response.data.filter(isAIEngineer);
      console.log("Filtered AI Engineer approval requests:", aiEngineers);
      
      setPendingUsers(aiEngineers);
    } catch (err) {
      console.error("Error fetching pending users:", err);
      setError("Failed to fetch pending approval requests");
    } finally {
      setLoading(false);
    }
  };
  
  useEffect(() => {
    fetchPendingUsers();
  }, []);
  
  const handleApprove = async (userId) => {
    try {
      setProcessingUser(userId);
      
      const token = localStorage.getItem('access_token');
      await axios.post(`${API_URL}/api/approvals/${userId}/approve/`, {}, {
        headers: { Authorization: `Bearer ${token}` }
      });
      
      // Remove the approved user from the list
      setPendingUsers(pendingUsers.filter(user => user.id !== userId));
    } catch (err) {
      console.error(`Error approving user ${userId}:`, err);
      setError("Failed to approve user");
    } finally {
      setProcessingUser(null);
    }
  };
  
  const handleReject = async (userId) => {
    try {
      setProcessingUser(userId);
      
      const token = localStorage.getItem('access_token');
      await axios.post(`${API_URL}/api/approvals/${userId}/reject/`, {}, {
        headers: { Authorization: `Bearer ${token}` }
      });
      
      // Remove the rejected user from the list
      setPendingUsers(pendingUsers.filter(user => user.id !== userId));
    } catch (err) {
      console.error(`Error rejecting user ${userId}:`, err);
      setError("Failed to reject user");
    } finally {
      setProcessingUser(null);
    }
  };
  
  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <RefreshCw className={`w-6 h-6 ${isDarkMode ? 'text-purple-400' : 'text-indigo-500'} animate-spin`} />
        <span className={`ml-3 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
          Loading approval requests...
        </span>
      </div>
    );
  }
  
  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h2 className={`text-2xl font-semibold ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>
          AI Engineer Approval Requests
        </h2>
        <button
          onClick={fetchPendingUsers}
          className={`p-2 rounded-lg ${
            isDarkMode 
              ? 'bg-gray-700 hover:bg-gray-600 text-gray-200' 
              : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
          } transition-colors`}
        >
          <RefreshCw className="w-4 h-4" />
        </button>
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
      
      {pendingUsers.length === 0 && !loading && !error ? (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className={`p-6 rounded-lg text-center ${
            isDarkMode 
              ? 'bg-gray-700/50 text-gray-300' 
              : 'bg-gray-50 text-gray-600'
          }`}
        >
          <CheckCircle className="w-12 h-12 mx-auto mb-3 opacity-50" />
          <p className="text-lg font-medium">No pending AI Engineer requests</p>
          <p className="text-sm mt-1 opacity-75">All AI Engineer users have been processed</p>
        </motion.div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <AnimatePresence>
            {pendingUsers.map(user => (
              <motion.div
                key={user.id}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9, y: -10 }}
                className={`rounded-lg p-5 ${
                  isDarkMode 
                    ? 'bg-gray-700/70 border border-white/5' 
                    : 'bg-white border border-gray-200 shadow-sm'
                }`}
              >
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center">
                    <div className={`p-3 rounded-full ${
                      isDarkMode 
                        ? 'bg-purple-900/40 text-purple-300' 
                        : 'bg-indigo-50 text-indigo-600'
                    }`}>
                      <UserCheck className="w-5 h-5" />
                    </div>
                    <div className="ml-3">
                      <h3 className={`font-medium ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                        {user.username}
                      </h3>
                      <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                        {getUserRole(user)}
                      </p>
                    </div>
                  </div>
                  
                  <div className={`px-2 py-1 text-xs font-medium rounded-full ${
                    isDarkMode 
                      ? 'bg-amber-900/30 text-amber-300' 
                      : 'bg-amber-100 text-amber-700'
                  }`}>
                    Pending
                  </div>
                </div>
                
                <div className={`rounded-lg p-3 mb-4 ${
                  isDarkMode ? 'bg-gray-800/70' : 'bg-gray-50'
                }`}>
                  <div className="grid grid-cols-1 gap-2">
                    <div className="flex items-center">
                      <Mail className={`w-4 h-4 mr-2 ${
                        isDarkMode ? 'text-gray-400' : 'text-gray-500'
                      }`} />
                      <span className={`text-sm ${
                        isDarkMode ? 'text-gray-300' : 'text-gray-700'
                      }`}>
                        {user.email}
                      </span>
                    </div>
                    <div className="flex items-center">
                      <User className={`w-4 h-4 mr-2 ${
                        isDarkMode ? 'text-gray-400' : 'text-gray-500'
                      }`} />
                      <span className={`text-sm ${
                        isDarkMode ? 'text-gray-300' : 'text-gray-700'
                      }`}>
                        ID: {user.id}
                      </span>
                    </div>
                    <div className="flex items-center">
                      <Calendar className={`w-4 h-4 mr-2 ${
                        isDarkMode ? 'text-gray-400' : 'text-gray-500'
                      }`} />
                      <span className={`text-sm ${
                        isDarkMode ? 'text-gray-300' : 'text-gray-700'
                      }`}>
                        Joined recently
                      </span>
                    </div>
                  </div>
                </div>
                
                <div className="flex space-x-3">
                  <button
                    onClick={() => handleApprove(user.id)}
                    disabled={processingUser === user.id}
                    className={`flex-1 py-2 px-3 rounded-lg flex items-center justify-center ${
                      isDarkMode 
                        ? 'bg-green-600 hover:bg-green-700 text-white disabled:bg-green-800/40 disabled:text-green-300/50' 
                        : 'bg-green-600 hover:bg-green-700 text-white disabled:bg-green-300 disabled:text-green-800/50'
                    } transition-colors`}
                  >
                    {processingUser === user.id ? (
                      <RefreshCw className="w-4 h-4 animate-spin" />
                    ) : (
                      <>
                        <CheckCircle className="w-4 h-4 mr-2" />
                        <span>Approve</span>
                      </>
                    )}
                  </button>
                  
                  <button
                    onClick={() => handleReject(user.id)}
                    disabled={processingUser === user.id}
                    className={`flex-1 py-2 px-3 rounded-lg flex items-center justify-center ${
                      isDarkMode 
                        ? 'bg-red-600 hover:bg-red-700 text-white disabled:bg-red-800/40 disabled:text-red-300/50' 
                        : 'bg-red-600 hover:bg-red-700 text-white disabled:bg-red-300 disabled:text-red-800/50'
                    } transition-colors`}
                  >
                    {processingUser === user.id ? (
                      <RefreshCw className="w-4 h-4 animate-spin" />
                    ) : (
                      <>
                        <XCircle className="w-4 h-4 mr-2" />
                        <span>Reject</span>
                      </>
                    )}
                  </button>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
};

export default ApprovalRequests; 