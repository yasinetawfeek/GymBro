import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Edit3, Save, Trash2, Eye } from 'lucide-react';
import ProfileSection from './ProfileSection';
import UserDetailModal from './UserDetailModal';

const ProfileOverview = ({ userData, setUserData, isEditing, setIsEditing, icons, isDarkMode = true }) => {
  const [showDetailModal, setShowDetailModal] = useState(false);
  const [localUserData, setLocalUserData] = useState({...userData});
  const [isSaving, setIsSaving] = useState(false);
  
  // Create user object format compatible with the UserDetailModal
  const userDetails = {
    id: userData?.id || 1,
    username: userData?.basicInfo?.username || userData?.basicInfo?.fullName?.split(' ')[0]?.toLowerCase() || 'user',
    email: userData?.basicInfo?.email || '',
    rolename: userData?.basicInfo?.role || 'User',
    memberSince: userData?.basicInfo?.memberSince || '',
    lastActive: 'Today',
    location: userData?.basicInfo?.location || '',
    phoneNumber: userData?.basicInfo?.phoneNumber || ''
  };

  const handleUserSave = (updatedUser) => {
    // Update the local user data with values from the user details modal
    const updatedUserData = {
      ...localUserData,
      basicInfo: {
        ...localUserData.basicInfo,
        fullName: updatedUser.fullName || updatedUser.username,
        email: updatedUser.email,
        location: updatedUser.location || localUserData.basicInfo.location,
      }
    };
    
    setLocalUserData(updatedUserData);
  };
  
  const handleSaveChanges = async () => {
    setIsSaving(true);
    try {
      // Call the parent component's function to save data to backend
      await setUserData(localUserData);
      setIsEditing(false);
    } catch (error) {
      console.error('Error saving profile:', error);
    } finally {
      setIsSaving(false);
    }
  };
  
  // Update local state when userData changes (e.g., from API)
  React.useEffect(() => {
    if (userData && !isEditing) {
      setLocalUserData({...userData});
    }
  }, [userData, isEditing]);

  // Handle field changes in the sections
  const handleFieldChange = (section, field, value) => {
    setLocalUserData(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [field]: value
      }
    }));
  };

  if (!userData) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className={`${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>User data is not available</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between mb-8">
        <h2 className="text-2xl font-light">Profile <span className="text-purple-400 font-medium">Overview</span></h2>
        <div className="flex space-x-2">
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setShowDetailModal(true)}
            className="bg-purple-500/10 hover:bg-purple-500/20 px-4 py-2 rounded-lg flex items-center space-x-2 text-sm"
          >
            <Eye className="w-4 h-4" />
            <span className="font-light">Details</span>
          </motion.button>
          {isEditing ? (
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleSaveChanges}
              disabled={isSaving}
              className="bg-green-500/10 hover:bg-green-500/20 px-4 py-2 rounded-lg flex items-center space-x-2 text-sm"
            >
              {isSaving ? (
                <div className="w-4 h-4 border-t-2 border-purple-400 rounded-full animate-spin mr-2"></div>
              ) : (
                <Save className="w-4 h-4" />
              )}
              <span className="font-light">Save</span>
            </motion.button>
          ) : (
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setIsEditing(true)}
              className="bg-purple-500/10 hover:bg-purple-500/20 px-4 py-2 rounded-lg flex items-center space-x-2 text-sm"
            >
              <Edit3 className="w-4 h-4" />
              <span className="font-light">Edit</span>
            </motion.button>
          )}
        </div>
      </div>

      <div className="space-y-6">
        <ProfileSection 
          title="BasicInfo" 
          data={localUserData.basicInfo} 
          icons={icons} 
          isEditing={isEditing}
          onFieldChange={(field, value) => handleFieldChange('basicInfo', field, value)}
          isDarkMode={isDarkMode}
        />
        <ProfileSection 
          title="FitnessProfile" 
          data={localUserData.fitnessProfile} 
          icons={icons} 
          isEditing={isEditing}
          onFieldChange={(field, value) => handleFieldChange('fitnessProfile', field, value)}
          isDarkMode={isDarkMode}
        />
        <ProfileSection 
          title="Preferences" 
          data={localUserData.preferences} 
          icons={icons} 
          isEditing={isEditing}
          onFieldChange={(field, value) => handleFieldChange('preferences', field, value)}
          isDarkMode={isDarkMode}
        />
        <ProfileSection 
          title="Achievements" 
          data={localUserData.achievements} 
          icons={icons} 
          isEditing={isEditing}
          onFieldChange={(field, value) => handleFieldChange('achievements', field, value)}
          isDarkMode={isDarkMode}
        />
      </div>

      <motion.div 
        className="flex justify-end pt-6 mt-6 border-t border-white/5"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <motion.button
          whileHover={{ scale: 1.02, backgroundColor: 'rgba(239, 68, 68, 0.2)' }}
          className="flex items-center space-x-2 text-red-400 px-4 py-2 rounded-lg bg-red-500/5 hover:bg-red-500/10 text-sm"
        >
          <Trash2 className="w-4 h-4" />
          <span className="font-light">Delete Account</span>
        </motion.button>
      </motion.div>
      
      <AnimatePresence>
        {showDetailModal && (
          <UserDetailModal 
            user={userDetails} 
            onClose={() => setShowDetailModal(false)}
            onSave={handleUserSave}
            onDelete={() => console.log("Delete not available for own account")}
            isAdmin={false}
          />
        )}
      </AnimatePresence>
    </div>
  );
};

export default ProfileOverview;