import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Search, Filter, User, Mail, 
  ChevronRight, X, CheckCircle, SlashIcon, UserPlus, 
  ArrowUpDown, RefreshCw, AlertCircle
} from 'lucide-react';
import axios from 'axios';
import userService from '../services/userService';
import { API_URL } from '../config';

// Animation variants for table rows
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

const UserManagement = ({ isDarkMode = true, onSelectUser }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [isFiltering, setIsFiltering] = useState(false);
  const [filterRole, setFilterRole] = useState('All');
  const [filterStatus, setFilterStatus] = useState('All');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  
  const [users, setUsers] = useState([]);

  // Fetch users on component mount
  useEffect(() => {
    fetchUsers();
  }, []);

  const fetchUsers = async () => {
    try {
      setIsLoading(true);
      setError(null);
      console.log("Fetching users...");
      
      const response = await userService.getAllUsers();
      console.log("API response:", response);
      
      if (response) {
        // Process each user to ensure we have the correct approval status
        const formattedUsers = await Promise.all(response.map(async user => {
          let isApproved = false;
          
          // For AI Engineers, we need to check their specific approval status
          if (user.groups && user.groups.length > 0 && user.groups[0].name === 'AI Engineer') {
            try {
              // Get specific user role info to check approval status
              const token = localStorage.getItem('access_token');
              const roleResponse = await axios.get(`${API_URL}/api/role_info/${user.id}/user_approval_status/`, {
                headers: { Authorization: `Bearer ${token}` }
              });
              isApproved = roleResponse.data.is_approved;
            } catch (error) {
              console.error(`Error fetching approval status for user ${user.id}:`, error);
              // Default to backend value if available, otherwise false
              isApproved = user.profile?.is_approved || false;
            }
          } else {
            // Non-AI Engineers are considered approved by default
            isApproved = true;
          }
          
          // Get user profile information directly from the user object
          const title = user.title || '';
          const forename = user.forename || user.first_name || '';
          const surname = user.surname || user.last_name || '';
          
          // Create formatted full name with title if available
          let fullName = user.username;
          if (forename && surname) {
            fullName = title ? `${title} ${forename} ${surname}` : `${forename} ${surname}`;
          } else if (user.first_name && user.last_name) {
            fullName = `${user.first_name} ${user.last_name}`;
          }
          
          console.log("User data:", { 
            id: user.id, 
            username: user.username,
            title, forename, surname,
            fullName
          });
          
          return {
            id: user.id,
            username: user.username,
            fullName,
            email: user.email || 'No email',
            rolename: user.groups && user.groups.length > 0 ? user.groups[0].name : 'Customer',
            memberSince: new Date(user.date_joined).toLocaleDateString(),
            lastActive: 'Recently',
            status: user.is_active ? 'Active' : 'Inactive',
            isApproved: isApproved
          };
        }));
        
        setUsers(formattedUsers);
      }
    } catch (err) {
      console.error("Error fetching users:", err);
      setError("Failed to load users. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleToggleApproval = async (userId, isApproved) => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Determine which endpoint to call based on the target approval status
      const endpoint = isApproved ? 'approve' : 'reject';
      const action = isApproved ? 'approved' : 'rejected';
      
      const token = localStorage.getItem('access_token');
      await axios.post(`${API_URL}/api/approvals/${userId}/${endpoint}/`, {}, {
        headers: { Authorization: `Bearer ${token}` }
      });
      
      // Update local state
      setUsers(users.map(user => 
        user.id === userId ? { ...user, isApproved } : user
      ));
      
      console.log(`User ${action} successfully`);
      
    } catch (error) {
      console.error(`Error ${isApproved ? 'approving' : 'rejecting'} user:`, error);
      setError(`Failed to ${isApproved ? 'approve' : 'reject'} user. Please try again.`);
    } finally {
      setIsLoading(false);
    }
  };

  const refreshUsers = () => {
    fetchUsers();
  };

  const filteredUsers = users.filter(user => {
    // Apply search filter
    const matchesSearch = 
      user.fullName?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      user.username.toLowerCase().includes(searchTerm.toLowerCase()) ||
      user.email.toLowerCase().includes(searchTerm.toLowerCase());
    
    // Apply role filter
    const matchesRole = filterRole === 'All' || user.rolename === filterRole;
    
    // Apply status filter
    const matchesStatus = filterStatus === 'All' || user.status === filterStatus;
    
    return matchesSearch && matchesRole && matchesStatus;
  });

  const roles = ['All', ...new Set(users.map(user => user.rolename))];
  const statuses = ['All', ...new Set(users.map(user => user.status))];

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-8 gap-4">
        <h2 className={`text-2xl font-light ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>
          User <span className="text-purple-400 font-medium">Management</span>
        </h2>
        
        {/* <motion.button
          whileHover={{ scale: 1.05, y: -2 }}
          whileTap={{ scale: 0.95 }}
          className={`${
            isDarkMode 
              ? 'bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 border border-purple-500/20' 
              : 'bg-purple-50 hover:bg-purple-100 text-purple-600 border border-purple-200'
          } px-4 py-2 rounded-lg flex items-center space-x-2 text-sm shadow-md`}
        >
          <UserPlus className="w-4 h-4" />
          <span className="font-medium">Add User</span>
        </motion.button> */}
      </div>

      {error && (
        <div className={`p-4 mb-4 rounded-lg flex items-center ${
          isDarkMode 
            ? 'bg-red-500/10 text-red-400 border border-red-500/20' 
            : 'bg-red-50 text-red-600 border border-red-200'
        }`}>
          <AlertCircle className="w-5 h-5 mr-2 flex-shrink-0" />
          <span>{error}</span>
          <button 
            className="ml-auto"
            onClick={() => setError(null)}
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      <div className="flex flex-col md:flex-row gap-4 mb-6">
        <div className="relative flex-grow">
          <Search className={`w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 ${
            isDarkMode ? 'text-gray-400' : 'text-gray-500'
          }`} />
          <input
            type="text"
            placeholder="Search users..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className={`w-full ${
              isDarkMode 
                ? 'bg-gray-800 text-white border border-white/10 focus:border-purple-500/50' 
                : 'bg-white text-gray-800 border border-gray-200 focus:border-purple-500'
            } rounded-lg pl-10 pr-4 py-2.5 focus:outline-none focus:ring-1 ${
              isDarkMode ? 'focus:ring-purple-500/30' : 'focus:ring-purple-500/50'
            } shadow-md`}
          />
          {searchTerm && (
            <button 
              onClick={() => setSearchTerm('')}
              className={`absolute right-3 top-1/2 transform -translate-y-1/2 ${
                isDarkMode ? 'text-gray-400 hover:text-white' : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
        
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => setIsFiltering(!isFiltering)}
          className={`${
            isDarkMode 
              ? 'bg-gray-800/80 backdrop-blur-md border border-white/5 text-white' 
              : 'bg-white border border-gray-200 text-gray-700'
          } px-4 py-2.5 rounded-lg flex items-center space-x-2 shadow-md ${
            isFiltering ? (isDarkMode ? 'ring-2 ring-purple-500/30' : 'ring-2 ring-purple-500/20') : ''
          }`}
        >
          <Filter className="w-4 h-4" />
          <span className="text-sm font-medium">Filter</span>
          <span className={`px-1.5 py-0.5 rounded-full text-xs ${
            isDarkMode 
              ? 'bg-purple-500/20 text-purple-400' 
              : 'bg-purple-100 text-purple-600'
          }`}>
            {(filterRole !== 'All' ? 1 : 0) + (filterStatus !== 'All' ? 1 : 0)}
          </span>
        </motion.button>
        
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={refreshUsers}
          disabled={isLoading}
          className={`${
            isDarkMode 
              ? 'bg-gray-800/80 backdrop-blur-md border border-white/5 text-white' 
              : 'bg-white border border-gray-200 text-gray-700'
          } px-4 py-2.5 rounded-lg flex items-center space-x-2 shadow-md`}
        >
          {isLoading ? (
            <RefreshCw className="w-4 h-4 animate-spin" />
          ) : (
            <RefreshCw className="w-4 h-4" />
          )}
          <span className="text-sm font-medium">Refresh</span>
        </motion.button>
      </div>

      <AnimatePresence>
        {isFiltering && (
          <motion.div 
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className={`${
              isDarkMode 
                ? 'bg-gray-800 border border-white/10' 
                : 'bg-gray-50 border border-gray-200'
            } rounded-lg p-4 mb-6 overflow-hidden shadow-md`}
          >
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className={`block text-sm font-medium mb-2 ${
                  isDarkMode ? 'text-gray-300' : 'text-gray-700'
                }`}>
                  Role
                </label>
                <select 
                  value={filterRole}
                  onChange={(e) => setFilterRole(e.target.value)}
                  className={`w-full ${
                    isDarkMode 
                      ? 'bg-gray-700 text-white border border-gray-600' 
                      : 'bg-white text-gray-800 border border-gray-300'
                  } rounded-lg px-3 py-2 focus:outline-none focus:ring-1 ${
                    isDarkMode ? 'focus:ring-purple-500/30 focus:border-purple-500/50' : 'focus:ring-purple-500/40 focus:border-purple-500'
                  }`}
                >
                  {roles.map(role => (
                    <option key={role} value={role}>{role}</option>
                  ))}
                </select>
              </div>
              
              <div>
                <label className={`block text-sm font-medium mb-2 ${
                  isDarkMode ? 'text-gray-300' : 'text-gray-700'
                }`}>
                  Status
                </label>
                <select 
                  value={filterStatus}
                  onChange={(e) => setFilterStatus(e.target.value)}
                  className={`w-full ${
                    isDarkMode 
                      ? 'bg-gray-700 text-white border border-gray-600' 
                      : 'bg-white text-gray-800 border border-gray-300'
                  } rounded-lg px-3 py-2 focus:outline-none focus:ring-1 ${
                    isDarkMode ? 'focus:ring-purple-500/30 focus:border-purple-500/50' : 'focus:ring-purple-500/40 focus:border-purple-500'
                  }`}
                >
                  {statuses.map(status => (
                    <option key={status} value={status}>{status}</option>
                  ))}
                </select>
              </div>
            </div>
            
            <div className="flex justify-end mt-4">
              <button 
                onClick={() => {
                  setFilterRole('All');
                  setFilterStatus('All');
                }}
                className={`text-sm ${
                  isDarkMode ? 'text-purple-400 hover:text-purple-300' : 'text-purple-600 hover:text-purple-700'
                } mr-4`}
              >
                Reset Filters
              </button>
              
              <button 
                onClick={() => setIsFiltering(false)}
                className={`text-sm ${
                  isDarkMode ? 'text-gray-400 hover:text-gray-300' : 'text-gray-600 hover:text-gray-700'
                }`}
              >
                Close
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className={`${
        isDarkMode 
          ? 'bg-gray-800 border border-white/10' 
          : 'bg-white border border-gray-200'
      } rounded-xl overflow-hidden shadow-lg`}>
        <div className={`grid grid-cols-12 gap-4 px-6 py-4 border-b ${
          isDarkMode ? 'border-white/5 bg-black/20' : 'border-gray-200 bg-gray-50'
        } text-xs uppercase tracking-wider font-medium`}>
          <div className="col-span-4 sm:col-span-3 flex items-center space-x-1">
            <span className={isDarkMode ? 'text-purple-400/70' : 'text-purple-600/70'}>Username</span>
            <ArrowUpDown className="w-3 h-3 opacity-50" />
          </div>
          <div className="col-span-5 sm:col-span-4 flex items-center space-x-1">
            <span className={isDarkMode ? 'text-purple-400/70' : 'text-purple-600/70'}>Email</span>
            <ArrowUpDown className="w-3 h-3 opacity-50" />
          </div>
          <div className="hidden sm:flex sm:col-span-2 items-center space-x-1">
            <span className={isDarkMode ? 'text-purple-400/70' : 'text-purple-600/70'}>Role</span>
            <ArrowUpDown className="w-3 h-3 opacity-50" />
          </div>
          <div className="col-span-2 flex items-center space-x-1">
            <span className={isDarkMode ? 'text-purple-400/70' : 'text-purple-600/70'}>Status</span>
            <ArrowUpDown className="w-3 h-3 opacity-50" />
          </div>
          <div className="col-span-1"></div>
        </div>
        
        <AnimatePresence>
          {isLoading ? (
            <motion.div 
              initial={{ opacity: 1 }}
              animate={{ opacity: 1 }}
              className="flex justify-center items-center py-20"
            >
              <RefreshCw className={`w-8 h-8 animate-spin ${
                isDarkMode ? 'text-purple-400' : 'text-purple-600'
              }`} />
              <span className={`ml-3 ${
                isDarkMode ? 'text-gray-300' : 'text-gray-600'
              }`}>Loading users...</span>
            </motion.div>
          ) : filteredUsers.length > 0 ? (
            <motion.div 
              className={`divide-y ${isDarkMode ? 'divide-white/5' : 'divide-gray-200'}`}
              initial="hidden"
              animate="visible"
              variants={{
                visible: {
                  transition: {
                    staggerChildren: 0.05
                  }
                }
              }}
            >
              {filteredUsers.map((user) => (
                <motion.div
                  key={user.id}
                  variants={tableRowVariant}
                  whileHover={{ 
                    backgroundColor: isDarkMode ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.02)'
                  }}
                  className="grid grid-cols-12 gap-4 px-6 py-4 items-center cursor-pointer"
                  onClick={(e) => {
                    // Only navigate to user details if we didn't click on a control button
                    if (e.target.closest('.control-button')) return;
                    onSelectUser && onSelectUser(user);
                  }}
                >
                  <div className="col-span-4 sm:col-span-3 flex items-center space-x-3">
                    <div className={`${
                      isDarkMode ? 'bg-purple-500/20' : 'bg-purple-100'
                    } p-1.5 rounded-full`}>
                      <User className={`w-4 h-4 ${
                        isDarkMode ? 'text-purple-400' : 'text-purple-600'
                      }`} />
                    </div>
                    <div>
                      <div className={`font-medium ${
                        isDarkMode ? 'text-white' : 'text-gray-800'
                      }`}>
                        {user.fullName}
                      </div>
                      <div className={`text-xs ${
                        isDarkMode ? 'text-gray-400' : 'text-gray-500'
                      }`}>
                        @{user.username}
                      </div>
                    </div>
                  </div>
                  
                  <div className="col-span-5 sm:col-span-4 flex items-center space-x-2">
                    <Mail className={`w-3 h-3 flex-shrink-0 ${
                      isDarkMode ? 'text-gray-400' : 'text-gray-500'
                    }`} />
                    <span className={`truncate text-sm ${
                      isDarkMode ? 'text-gray-300' : 'text-gray-600'
                    }`}>
                      {user.email}
                    </span>
                  </div>
                  
                  <div className="hidden sm:flex sm:col-span-2 items-center">
                    <span className={`text-sm px-2 py-1 rounded-full ${
                      user.rolename === 'Admin' 
                        ? isDarkMode 
                          ? 'bg-purple-500/20 text-purple-400' 
                          : 'bg-purple-100 text-purple-700'
                        : user.rolename === 'AI Engineer' 
                          ? isDarkMode 
                            ? 'bg-blue-500/20 text-blue-400' 
                            : 'bg-blue-100 text-blue-700'
                          : isDarkMode 
                            ? 'bg-gray-700 text-gray-300' 
                            : 'bg-gray-100 text-gray-700'
                    }`}>
                      {user.rolename}
                      {user.rolename === 'AI Engineer' && (
                        <span 
                          className={`ml-1.5 inline-block w-2 h-2 rounded-full ${
                            user.isApproved
                              ? isDarkMode
                                ? 'bg-green-400'
                                : 'bg-green-500'
                              : isDarkMode
                                ? 'bg-amber-400'
                                : 'bg-amber-500'
                          }`}
                          title={user.isApproved ? 'Access Granted' : 'Access Restricted'}
                        />
                      )}
                    </span>
                  </div>
                  
                  <div className="col-span-2 flex items-center">
                    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs ${
                      user.status === 'Active' 
                        ? isDarkMode 
                          ? 'bg-green-500/20 text-green-400' 
                          : 'bg-green-100 text-green-700'
                        : isDarkMode 
                          ? 'bg-gray-700 text-gray-400' 
                          : 'bg-gray-100 text-gray-700'
                    }`}>
                      {user.status === 'Active' 
                        ? <CheckCircle className="w-3 h-3 mr-1" /> 
                        : <SlashIcon className="w-3 h-3 mr-1" />
                      }
                      {user.status}
                    </span>
                    
                    {/* Quick access toggle approval button for AI Engineers */}
                    {user.rolename === 'AI Engineer' && (
                      <button 
                        className={`ml-2 control-button px-1.5 py-0.5 rounded text-xs ${
                          user.isApproved
                            ? isDarkMode
                              ? 'bg-amber-500/20 hover:bg-amber-500/30 text-amber-400'
                              : 'bg-amber-50 hover:bg-amber-100 text-amber-600'
                            : isDarkMode
                              ? 'bg-green-500/20 hover:bg-green-500/30 text-green-400'
                              : 'bg-green-50 hover:bg-green-100 text-green-600'
                        }`}
                        onClick={(e) => {
                          e.stopPropagation();
                          handleToggleApproval(user.id, !user.isApproved);
                        }}
                      >
                        {user.isApproved ? 'Revoke' : 'Grant'}
                      </button>
                    )}
                  </div>
                  
                  <div className="col-span-1 flex justify-end">
                    <button
                      className={`control-button mr-2 ${
                        isDarkMode ? 'text-gray-400 hover:text-purple-400' : 'text-gray-500 hover:text-purple-600'
                      }`}
                      onClick={(e) => {
                        e.stopPropagation();
                        handleToggleApproval(user.id, !user.isApproved);
                      }}
                      title={user.isApproved ? 'Revoke Access' : 'Grant Access'}
                    >
                      {user.isApproved ? <SlashIcon className="w-4 h-4" /> : <CheckCircle className="w-4 h-4" />}
                    </button>
                    <ChevronRight className={`w-4 h-4 ${
                      isDarkMode ? 'text-gray-400' : 'text-gray-500'
                    }`} />
                  </div>
                </motion.div>
              ))}
            </motion.div>
          ) : (
            <motion.div 
              initial={{ opacity: 1 }}
              animate={{ opacity: 1 }}
              className="flex justify-center items-center py-20"
            >
              <span className={`text-sm ${
                isDarkMode ? 'text-gray-300' : 'text-gray-600'
              }`}>No users found.</span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default UserManagement;