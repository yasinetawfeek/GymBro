import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { X, Trash2, Save, Edit3 } from 'lucide-react';

const UserDetailModal = ({ user, onClose, onDelete, onSave, isAdmin = false, isDarkMode = true }) => {
  const [isEditing, setIsEditing] = useState(false);
  const [editedUser, setEditedUser] = useState({ ...user });

  const handleInputChange = (field, value) => {
    setEditedUser(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleSave = () => {
    onSave(editedUser);
    setIsEditing(false);
  };

  const renderSection = (title, fields) => (
    <div className={`${
      isDarkMode 
        ? 'bg-gray-700 backdrop-blur-sm' 
        : 'bg-white shadow-sm border border-gray-200'
      } rounded-lg p-4 mb-4`}>
      <h3 className={`text-sm ${
        isDarkMode ? 'text-purple-400/60' : 'text-indigo-600'
      } uppercase tracking-wider mb-4`}>{title}</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {fields.map(field => (
          <div key={field} className={`${
            isDarkMode 
              ? 'bg-purple-500/5' 
              : 'bg-gray-50'
            } p-3 rounded-lg`}>
            <div className={`text-xs ${
              isDarkMode ? 'text-purple-400/60' : 'text-indigo-500'
            } uppercase mb-1`}>
              {field.replace(/([A-Z])/g, ' $1').trim()}
            </div>
            {isEditing ? (
              <input
                type="text"
                value={editedUser[field] || ''}
                onChange={(e) => handleInputChange(field, e.target.value)}
                className={`w-full ${
                  isDarkMode 
                    ? 'bg-purple-500/5 border-purple-500/10 focus:border-purple-500/20 text-white' 
                    : 'bg-white border-gray-300 focus:border-indigo-500 text-gray-800'
                  } border rounded-lg px-2 py-1 text-sm focus:outline-none mt-1`}
                disabled={field === 'id' || (!isAdmin && field === 'rolename')}
              />
            ) : (
              <div className={`font-light text-lg ${
                isDarkMode ? 'text-white' : 'text-gray-800'
              }`}>{user[field]}</div>
            )}
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 px-4 pt-16"
      onClick={onClose}
    >
      <motion.div 
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.9, opacity: 0 }}
        className={`${
          isDarkMode 
            ? 'bg-gray-800' 
            : 'bg-white'
          } rounded-xl p-6 max-w-3xl w-full max-h-[80vh] overflow-y-auto`}
        onClick={e => e.stopPropagation()}
      >
        <div className={`flex items-center justify-between mb-6 ${
          isDarkMode 
            ? 'bg-gray-800 border-white/10' 
            : 'bg-white border-gray-200'
          } pb-4 border-b`}>
          <h2 className={`text-2xl font-light ${
            isDarkMode ? 'text-white' : 'text-gray-800'
          }`}>User <span className={`${
            isDarkMode ? 'text-purple-400' : 'text-indigo-600'
          } font-medium`}>Details</span></h2>
          <div className="flex space-x-2">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={isEditing ? handleSave : () => setIsEditing(true)}
              className={`${
                isDarkMode 
                  ? 'bg-purple-500/10 hover:bg-purple-500/20 text-purple-400' 
                  : 'bg-indigo-50 hover:bg-indigo-100 text-indigo-600'
                } p-2 rounded-lg`}
            >
              {isEditing ? <Save className="w-5 h-5" /> : <Edit3 className="w-5 h-5" />}
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={onClose}
              className={`${
                isDarkMode 
                  ? 'bg-purple-500/10 hover:bg-purple-500/20 text-white' 
                  : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
                } p-2 rounded-lg`}
            >
              <X className="w-5 h-5" />
            </motion.button>
          </div>
        </div>
        
        {renderSection('Account Information', ['id', 'username', 'email', 'rolename'])}
        
        {user.memberSince && renderSection('Additional Details', [
          'memberSince', 
          'lastActive', 
          'location',
          'phoneNumber'
        ])}
        
        {isAdmin && (
          <div className={`flex justify-end mt-6 pt-4 border-t ${
            isDarkMode ? 'border-white/10' : 'border-gray-200'
          }`}>
            <motion.button
              whileHover={{ scale: 1.02, backgroundColor: 'rgba(239, 68, 68, 0.2)' }}
              whileTap={{ scale: 0.98 }}
              onClick={() => onDelete(user.id)}
              className="flex items-center space-x-2 text-red-400 px-4 py-2 rounded-lg bg-red-500/5 hover:bg-red-500/10"
            >
              <Trash2 className="w-4 h-4" />
              <span>Delete Account</span>
            </motion.button>
          </div>
        )}
      </motion.div>
    </motion.div>
  );
};

export default UserDetailModal;