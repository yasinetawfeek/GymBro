import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  UserCircle, Mail, MapPin, Calendar,
  Ruler, Weight, Heart, Medal, Target, 
  Clock, Dumbbell, ActivitySquare, TrendingUp,
  Award, AlertCircle, X, RefreshCw, BarChart3, ChevronLeft,
  Shield, Lock
} from 'lucide-react';

import NavBar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import ProfileOverview from '../components/ProfileOverview';
import FitnessStats from '../components/FitnessStats';
import UserManagement from '../components/UserManagement';
import UserDetailsPage from '../components/UserDetailsPage';
import UserDetailModal from '../components/UserDetailModal';
import ApprovalRequests from '../components/ApprovalRequests';
import BillingOverview from '../components/BillingOverview';
import AdminBillingActivity from '../components/AdminBillingActivity';
import ModelPerformance from '../components/ModelPerformance';
import MLModels from '../components/MLModels';

import { useAuth } from '../context/AuthContext';
import { useNavigate, useLocation } from 'react-router-dom';
import axios from 'axios';
import userService from '../services/userService';

// Animation variants
const pageTransition = {
  hidden: { opacity: 0, y: 20 },
  visible: { 
    opacity: 1, 
    y: 0,
    transition: { duration: 0.4 }
  },
  exit: {
    opacity: 0,
    y: -20,
    transition: { duration: 0.2 }
  }
};

const errorNotificationVariants = {
  hidden: { opacity: 0, y: -20 },
  visible: { 
    opacity: 1, 
    y: 0,
    transition: { duration: 0.3, type: "spring", stiffness: 300, damping: 25 }
  },
  exit: { 
    opacity: 0, 
    y: -20,
    transition: { duration: 0.2 }
  }
};

const AccountManagement = () => {
  const [isDarkMode, setIsDarkMode] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [approvalStatus, setApprovalStatus] = useState(true);
  const [permissionError, setPermissionError] = useState(null);
  
  // Change this to match Navbar.jsx - don't destructure
  const auth = useAuth();
    
  // Check system preference on initial load
  useEffect(() => {
    const prefersDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
    setIsDarkMode(prefersDarkMode);
  }, []);
  
  // Update dark mode class on body
  useEffect(() => {
    if (isDarkMode) {
      document.body.classList.add('dark');
    } else {
      document.body.classList.remove('dark');
    }
  }, [isDarkMode]);
  
  // Fetch role and approval status
  useEffect(() => {
    const fetchRoleInfo = async () => {
      if (auth.token) {
        try {
          setLoading(true);
          const response = await axios.get('http://localhost:8000/api/role_info/', {
            headers: { Authorization: `Bearer ${auth.token}` }
          });
          
          console.log("Role info:", response.data);
          setApprovalStatus(response.data.is_approved);
          
          // If the user is not approved and is an AI Engineer, show a notification
          if (!response.data.is_approved && response.data.is_ai_engineer) {
            setError('Your account is awaiting admin approval. Some features are limited.');
          }
          
        } catch (error) {
          console.error("Error fetching role info:", error);
          setError('Failed to fetch role information');
        } finally {
          setLoading(false);
        }
      }
    };
    
    fetchRoleInfo();
  }, [auth.token]);
  
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [fitnessStats, setFitnessStats] = useState({
    recentWorkouts: [
      { date: '2025-03-10', type: 'Upper Body', duration: '45 min', intensity: 'High' },
      { date: '2025-03-08', type: 'Lower Body', duration: '50 min', intensity: 'Medium' },
      { date: '2025-03-06', type: 'Cardio', duration: '30 min', intensity: 'High' }
    ],
    progressMetrics: {
      monthlyWorkouts: 12,
      avgDuration: '45 min',
      weightProgress: '+2.5 kg',
      streakDays: 15
    }
  });
  const [selectedUser, setSelectedUser] = useState(null);
  const [userListRefreshTrigger, setUserListRefreshTrigger] = useState(0);

  // Parse URL parameters for page navigation
  const getPageFromUrl = () => {
    const searchParams = new URLSearchParams(location.search);
    return searchParams.get('page') || 'profile';
  };

  // Set active page from URL
  const [activePage, setActivePage] = useState(getPageFromUrl());

  // Update URL when changing pages
  const handlePageChange = (page) => {
    setPermissionError(null);
    
    // Check permission before changing the page
    if (hasPagePermission(page)) {
      navigate(`/settings?page=${page}`, { replace: true });
      setActivePage(page);
    } else {
      // Set permission error instead of changing the page
      setPermissionError(`You don't have permission to access the ${getPageLabel(page)} page.`);
    }
  };

  // Get a human-readable label for a page ID
  const getPageLabel = (pageId) => {
    const pageLabels = {
      'profile': 'Profile',
      'stats': 'Fitness Stats',
      'users': 'User Management',
      'approvals': 'Approval Requests',
      'billing': 'Billing',
      'billingActivity': 'Billing Activity',
      'userDetails': 'User Details',
      'performance': 'Model Performance',
      'models': 'ML Models'
    };
    
    return pageLabels[pageId] || pageId;
  };

  // Update activePage when URL changes
  useEffect(() => {
    const pageFromUrl = getPageFromUrl();
    
    // Check if the user has permission for the requested page
    if (hasPagePermission(pageFromUrl)) {
      setActivePage(pageFromUrl);
      setPermissionError(null);
    } else {
      // Instead of redirecting, set the permission error and keep the URL unchanged
      setPermissionError(`You don't have permission to access the ${getPageLabel(pageFromUrl)} page.`);
      // Keep using the page from URL so the URL stays unchanged
      setActivePage(pageFromUrl);
    }
  }, [location.search]);

  // Determine user role from backend data
  const getUserRole = () => {
    console.log("User object:", auth);
    console.log("User data:", auth?.user);
    
    if (!auth?.user) {
      console.log("No user data found, defaulting to 'user' role");
      return 'Customer';
    }
    
    // Check if the user has the is_admin field set to true
    if (auth.user.is_admin === true) {
      console.log("User is identified as admin via is_admin field");
      return 'Admin';
    }
    
    // Check if the user has admin permissions
    if (auth.user.basicInfo?.isAdmin || auth.user.basicInfo?.rolename === 'Admin' || 
        auth.user.rolename === 'Admin' || auth.user.basicInfo?.role === 'Admin') {
      console.log("User is identified as admin");
      return 'Admin';
    }
    
    // Check groups directly if available
    if (auth.user.groups && Array.isArray(auth.user.groups)) {
      const hasAdminGroup = auth.user.groups.some(g => {
        return (g && (g.name === 'Admin' || g === 'Admin'));
      });
      
      if (hasAdminGroup) {
        console.log("User is admin via groups array");
        return 'Admin';
      }
      
      const hasAIEngineerGroup = auth.user.groups.some(g => {
        return (g && (g.name === 'AI Engineer' || g === 'AI Engineer'));
      });
      
      if (hasAIEngineerGroup) {
        console.log("User is AI Engineer via groups array");
        return 'AI Engineer';
      }
    }
    
    console.log("User is not admin or AI Engineer, defaulting to 'Customer' role");
    return 'Customer';
  };
  
  const userRole = getUserRole();
  console.log("Determined user role:", userRole);

  // Define page permissions based on user role
  const hasPagePermission = (page) => {
    // Pages accessible by all roles
    if (['profile', 'stats', 'billing'].includes(page)) {
      return true;
    }
    
    // Admin-only pages
    if (['users', 'approvals', 'billingActivity'].includes(page)) {
      return userRole === 'Admin';
    }
    
    // AI Engineer pages (require approval)
    if (['models', 'performance'].includes(page)) {
      return userRole === 'Admin' || (userRole === 'AI Engineer' && approvalStatus);
    }
    
    return false;
  };

  // Set initial active page based on role and URL
  useEffect(() => {
    const pageFromUrl = getPageFromUrl();
    
    // If URL has a page parameter, try to use it
    if (pageFromUrl && pageFromUrl !== 'profile') {
      if (hasPagePermission(pageFromUrl)) {
        setActivePage(pageFromUrl);
      } else {
        // Set permission error but don't redirect
        setPermissionError(`You don't have permission to access the ${getPageLabel(pageFromUrl)} page.`);
        // Still set the active page to show the permission denied component
        setActivePage(pageFromUrl);
      }
    } else {
      // No page in URL, set default based on role
      const defaultPage = userRole === 'Admin' ? 'users' : 
                         userRole === 'AI Engineer' ? 'models' : 'profile';
      navigate(`/settings?page=${defaultPage}`, { replace: true });
    }
  }, [userRole]);

  // Adding navigation to fitness stats for non-admin users
  const navigateToStats = () => {
    handlePageChange('stats');
  };

  const icons = {
    fullName: UserCircle,
    email: Mail,
    location: MapPin,
    memberSince: Calendar,
    height: Ruler,
    weight: Weight,
    bodyFat: Heart,
    fitnessLevel: Medal,
    primaryGoal: Target,
    workoutFrequency: Calendar,
    preferredTime: Clock,
    focusAreas: Dumbbell,
    workoutsCompleted: ActivitySquare,
    daysStreak: TrendingUp,
    personalBests: Award,
    points: Medal
  };

  // Define the missing updateProfile function
  const updateProfile = async (updatedData) => {
    const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/';
    try {
      // Log the data being sent to the API for debugging
      console.log('Sending profile update:', updatedData);
      
      const response = await axios.patch(
        `${API_URL}api/my_account/`, 
        updatedData,
        {
          headers: { 
            Authorization: `Bearer ${localStorage.getItem('access_token')}`,
            'Content-Type': 'application/json'
          }
        }
      );
      
      console.log('Profile update response:', response.data);
      return response.data;
    } catch (error) {
      console.error('Profile update error:', error.response?.data || error.message);
      throw error;
    }
  };

  // Handle the save action when profile is edited
  const handleProfileUpdate = async (updatedData) => {
    try {
      setLoading(true);
      setError(null);
      
      console.log('ProfileUpdate: Original user data:', updatedData);
      console.log('ProfileUpdate: Phone number from form:', updatedData.basicInfo?.phoneNumber);
      
      // Transform the data to match backend format
      const formattedData = userService.transformUserDataForBackend(updatedData);
      console.log('ProfileUpdate: Transformed data for backend:', formattedData);
      console.log('ProfileUpdate: Phone number being sent to backend:', formattedData.phone_number);
      
      // Send the formatted data to the API
      const response = await updateProfile(formattedData);
      console.log('ProfileUpdate: API response:', response);
      
      // Update the user context with the new data - properly merge nested objects
      if (auth && auth.setUser) {
        console.log('ProfileUpdate: Updating user context');
        
        // Create a proper deep merge of the user data
        const updatedUserData = {
          ...auth.user,
          basicInfo: { ...(auth.user?.basicInfo || {}), ...(updatedData.basicInfo || {}) },
          fitnessProfile: { ...(auth.user?.fitnessProfile || {}), ...(updatedData.fitnessProfile || {}) },
          preferences: { ...(auth.user?.preferences || {}), ...(updatedData.preferences || {}) },
          achievements: { ...(auth.user?.achievements || {}), ...(updatedData.achievements || {}) }
        };
        
        console.log('ProfileUpdate: Updated user context data:', updatedUserData);
        console.log('ProfileUpdate: Phone number in updated context:', updatedUserData.basicInfo?.phoneNumber);
        
        auth.setUser(updatedUserData);
        
        // Also update localStorage to persist the changes
        localStorage.setItem('user', JSON.stringify(updatedUserData));
      } else {
        console.warn('ProfileUpdate: Unable to update user context - setUser not available');
      }
      
      setIsEditing(false);
      console.log('ProfileUpdate: Profile update successful');
    } catch (err) {
      console.error('Failed to update profile:', err);
      if (err.response) {
        console.error('Response status:', err.response.status);
        console.error('Response data:', err.response.data);
      }
      setError('Failed to update profile. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Clear error message
  const clearError = () => {
    setError(null);
  };

  // Clear permission error message
  const clearPermissionError = () => {
    setPermissionError(null);
  };

  // Show loading spinner while user data is being fetched
  if (loading) {
    return (
      <div className={`min-h-screen flex items-center justify-center ${
        isDarkMode 
          ? 'bg-gradient-to-br from-gray-900 via-gray-800 to-indigo-900' 
          : 'bg-gradient-to-br from-gray-50 via-white to-indigo-50'
      }`}>
        <div className="flex flex-col items-center">
          <RefreshCw className={`w-12 h-12 animate-spin mb-4 ${
            isDarkMode ? 'text-purple-400' : 'text-purple-600'
          }`} />
          <p className={isDarkMode ? 'text-gray-300' : 'text-gray-600'}>
            Loading your profile...
          </p>
        </div>
      </div>
    );
  }

  // Handle user selection from UserManagement
  const handleSelectUser = (user) => {
    console.log("Selected user:", user);
    setSelectedUser(user);
  };

  // Handle user deletion
  const handleDeleteUser = async (userId) => {
    console.log("Deleting user:", userId);
    try {
      setLoading(true);
      // Call the userService to delete the user
      await userService.deleteUser(userId);
      console.log("User deleted successfully");
      setSelectedUser(null);
      
      // If the active page is users, refresh the user list
      if (activePage === 'users') {
        // This will trigger the UserManagement component to refresh
        console.log("Refreshing user list after deletion");
        setUserListRefreshTrigger(prevTrigger => prevTrigger + 1);
      }
      
    } catch (error) {
      console.error("Failed to delete user:", error);
      setError("Failed to delete user. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  // Handle user update
  const handleSaveUser = (updatedUser) => {
    console.log("Updating user:", updatedUser);
    // You might want to refresh the user list after update
  };

  // Handle toggle approval for AI Engineers
  const handleToggleApproval = async (userId, isApproved) => {
    try {
      setLoading(true);
      
      const endpoint = isApproved ? 'approve' : 'reject';
      const action = isApproved ? 'approved' : 'rejected';
      
      const token = localStorage.getItem('access_token');
      await axios.post(`http://localhost:8000/api/approvals/${userId}/${endpoint}/`, {}, {
        headers: { Authorization: `Bearer ${token}` }
      });
      
      console.log(`User ${action} successfully`);
      
      // Update the selected user if it's the one we just modified
      if (selectedUser && selectedUser.id === userId) {
        setSelectedUser({
          ...selectedUser,
          isApproved: isApproved
        });
      }
      
      // Refresh user list to reflect changes
      setUserListRefreshTrigger(prev => prev + 1);
      
    } catch (error) {
      console.error(`Error ${isApproved ? 'approving' : 'rejecting'} user:`, error);
      setError(`Failed to ${isApproved ? 'approve' : 'reject'} user. Please try again.`);
    } finally {
      setLoading(false);
    }
  };

  // Permission Denied Component
  const PermissionDenied = () => (
    <div className="flex flex-col items-center justify-center py-12 text-center">
      <Lock className={`w-16 h-16 mb-4 ${isDarkMode ? 'text-red-400' : 'text-red-500'}`} />
      <h2 className={`text-2xl font-semibold mb-3 ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>
        Access Denied
      </h2>
      <p className={`max-w-md mb-6 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
        {permissionError || "You don't have permission to access this page."}
      </p>
      <button
        onClick={() => handlePageChange('profile')}
        className={`px-4 py-2 rounded-lg font-medium ${
          isDarkMode 
            ? 'bg-purple-600 hover:bg-purple-500 text-white' 
            : 'bg-purple-600 hover:bg-purple-500 text-white'
        }`}
      >
        Go to Profile
      </button>
    </div>
  );

  const renderActivePage = () => {
    console.log("Rendering active page:", activePage);
    
    // If there's a permission error, show the permission denied component
    if (permissionError) {
      return <PermissionDenied />;
    }
    
    // Otherwise, render the requested page if permission is granted
    if (!hasPagePermission(activePage)) {
      return <PermissionDenied />;
    }
    
    switch (activePage) {
      case 'profile':
        console.log("Rendering ProfileOverview component");
        return (
          <ProfileOverview 
            userData={auth.user}
            setUserData={handleProfileUpdate}
            isEditing={isEditing}
            setIsEditing={setIsEditing}
            icons={icons}
            isDarkMode={isDarkMode}
          />
        );
      case 'stats':
        console.log("Rendering FitnessStats component");
        return (
          <div>
            <div className="flex items-center justify-between mb-6">
              <h2 className={`text-2xl font-light ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>
                Your <span className="text-purple-400 font-medium">Fitness Analytics</span>
              </h2>
              <button
                onClick={() => handlePageChange('profile')}
                className={`p-2 rounded-lg ${
                  isDarkMode 
                    ? 'bg-gray-700 hover:bg-gray-600 text-gray-200' 
                    : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
                } transition-colors flex items-center gap-2`}
              >
                <ChevronLeft className="w-4 h-4" />
                <span>Back to Profile</span>
              </button>
            </div>
            <FitnessStats fitnessStats={fitnessStats} isDarkMode={isDarkMode} />
          </div>
        );
      case 'users':
        console.log("Rendering UserManagement component");
        return (
          <UserManagement 
            isDarkMode={isDarkMode}
            onSelectUser={handleSelectUser}
            onDeleteUser={handleDeleteUser}
            onSaveUser={handleSaveUser}
            onToggleApproval={handleToggleApproval}
            key={userListRefreshTrigger}
          />
        );
      case 'approvals':
        console.log("Rendering ApprovalRequests component");
        return <ApprovalRequests isDarkMode={isDarkMode} />;
      case 'billing':
        console.log("Rendering BillingOverview component");
        return <BillingOverview isDarkMode={isDarkMode} />;
      case 'billingActivity':
        console.log("Rendering AdminBillingActivity component");
        return <AdminBillingActivity isDarkMode={isDarkMode} />;
      case 'userDetails':
        console.log("Rendering UserDetailsPage component");
        return <UserDetailsPage isDarkMode={isDarkMode} />;
      case 'performance':
        console.log("Rendering ModelPerformance component");
        return <ModelPerformance isDarkMode={isDarkMode} />;
      case 'models':
        console.log("Rendering MLModels component");
        return <MLModels isDarkMode={isDarkMode} />;
      default:
        console.log("No matching component for active page:", activePage);
        return <div>Page not found</div>;
    }
  };

  // Add toggleDarkMode function
  const toggleDarkMode = () => {
    setIsDarkMode(!isDarkMode);
  };

  return (
    <div className={`min-h-screen ${
      isDarkMode 
        ? 'bg-gradient-to-br from-gray-900 via-gray-800 to-indigo-900 text-white' 
        : 'bg-gradient-to-br from-gray-50 via-white to-indigo-50 text-gray-900'
    }`}>
      <NavBar isDarkMode={isDarkMode} toggleDarkMode={toggleDarkMode} />
      
      {/* Approval Status Banner for AI Engineers */}
      {userRole === 'AI Engineer' && !approvalStatus && (
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className={`fixed top-16 left-0 right-0 z-40 p-3 ${
            isDarkMode 
              ? 'bg-amber-800/90 text-amber-100 border-b border-amber-700'
              : 'bg-amber-100 text-amber-800 border-b border-amber-200'
          } flex items-center justify-center gap-2`}
        >
          <AlertCircle className="w-4 h-4" />
          <span className="text-sm">Your account is awaiting admin approval. Some features are limited.</span>
        </motion.div>
      )}
      
      {/* Error Notification - Now fixed at the top, above all content */}
      <AnimatePresence>
        {error && (
          <motion.div 
            variants={errorNotificationVariants}
            initial="hidden"
            animate="visible"
            exit="exit"
            className="fixed top-20 left-0 right-0 mx-auto max-w-4xl px-4 z-50"
          >
            <div className={`${
              isDarkMode 
                ? 'bg-red-900/80 text-red-100 border border-red-800/50' 
                : 'bg-red-50/95 text-red-700 border border-red-200'
            } p-4 rounded-xl flex items-center justify-between shadow-xl backdrop-blur-md`}>
              <div className="flex items-center">
                <AlertCircle className="w-5 h-5 mr-3 flex-shrink-0" />
                <p>{error}</p>
              </div>
              <button 
                onClick={clearError}
                className={`p-1 rounded-full ${
                  isDarkMode 
                    ? 'hover:bg-red-800/50 text-red-200' 
                    : 'hover:bg-red-100 text-red-500'
                } transition-colors duration-200`}
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Permission Error Notification */}
      <AnimatePresence>
        {permissionError && (
          <motion.div 
            variants={errorNotificationVariants}
            initial="hidden"
            animate="visible"
            exit="exit"
            className="fixed top-28 left-0 right-0 mx-auto max-w-4xl px-4 z-50"
          >
            <div className={`${
              isDarkMode 
                ? 'bg-amber-900/80 text-amber-100 border border-amber-800/50' 
                : 'bg-amber-50/95 text-amber-700 border border-amber-200'
            } p-4 rounded-xl flex items-center justify-between shadow-xl backdrop-blur-md`}>
              <div className="flex items-center">
                <Shield className="w-5 h-5 mr-3 flex-shrink-0" />
                <p>{permissionError}</p>
              </div>
              <button 
                onClick={clearPermissionError}
                className={`p-1 rounded-full ${
                  isDarkMode 
                    ? 'hover:bg-amber-800/50 text-amber-200' 
                    : 'hover:bg-amber-100 text-amber-500'
                } transition-colors duration-200`}
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 relative">
        <div className="flex flex-col lg:flex-row gap-6">
          <Sidebar 
            isMenuOpen={isMenuOpen}
            setIsMenuOpen={setIsMenuOpen}
            userRole={userRole}
            activePage={activePage}
            setActivePage={handlePageChange}
            isDarkMode={isDarkMode}
            isApproved={approvalStatus}
          />

          <div className="lg:ml-72 flex-1">
            <AnimatePresence mode="wait">
              <motion.div 
                key={activePage}
                variants={pageTransition}
                initial="hidden"
                animate="visible"
                exit="exit"
                className={`rounded-xl ${
                  isDarkMode 
                    ? 'bg-gray-800 border border-white/10' 
                    : 'bg-white shadow-lg border border-gray-200'
                } p-6 sm:p-8`}
              >
                {renderActivePage()}
              </motion.div>
            </AnimatePresence>
            
            {/* Analytics shortcut button for regular users */}
            {userRole === 'Customer' && activePage === 'profile' && (
              <motion.button
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                onClick={navigateToStats}
                className={`mt-4 px-4 py-2 rounded-lg shadow-md flex items-center ${
                  isDarkMode 
                    ? 'bg-purple-700 hover:bg-purple-600 text-white' 
                    : 'bg-purple-600 hover:bg-purple-700 text-white'
                } transition-colors`}
              >
                <BarChart3 className="w-4 h-4 mr-2" />
                <span>View Fitness Analytics</span>
              </motion.button>
            )}
          </div>
        </div>
      </div>
      
      {/* User Detail Modal - Rendered at the page level */}
      <AnimatePresence>
        {selectedUser && (
          <UserDetailModal 
            user={selectedUser}
            onClose={() => setSelectedUser(null)}
            onDelete={handleDeleteUser}
            onSave={handleSaveUser}
            onToggleApproval={handleToggleApproval}
            isAdmin={userRole === 'Admin'}
            isDarkMode={isDarkMode}
          />
        )}
      </AnimatePresence>
      
      {/* Decorative elements */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden -z-10">
        <div className="absolute top-0 right-0 w-1/3 h-1/3 bg-purple-500/5 rounded-full blur-3xl transform translate-x-1/2 -translate-y-1/2"></div>
        <div className="absolute bottom-0 left-0 w-1/2 h-1/2 bg-indigo-500/5 rounded-full blur-3xl transform -translate-x-1/3 translate-y-1/3"></div>
      </div>
    </div>
  );
};

export default AccountManagement;