import React from 'react';
import { motion } from 'framer-motion';

const ProfileSection = ({ title, data, icons, isEditing, onFieldChange, isDarkMode = true }) => {
  // Format field label for display
  const formatFieldLabel = (key) => {
    return key.replace(/([A-Z])/g, ' $1').trim();
  };

  return (
    <div className={`${
      isDarkMode 
        ? 'bg-gray-700 backdrop-blur-sm' 
        : 'bg-white shadow-sm border border-gray-200'
      } rounded-lg p-4`}
    >
      <h3 className={`text-sm ${
        isDarkMode ? 'text-purple-400/60' : 'text-indigo-600'
      } uppercase tracking-wider mb-4`}>{title}</h3>
      <div className="grid grid-cols-2 gap-4">
        {Object.entries(data || {}).map(([key, value]) => (
          <motion.div
            key={key}
            whileHover={{ scale: 1.02 }}
            className={`${
              isDarkMode 
                ? 'bg-purple-500/5' 
                : 'bg-gray-50'
              } p-3 rounded-lg`}
          >
            <div className="flex items-center space-x-2 mb-1">
              {icons[key] && React.createElement(icons[key], { 
                className: `w-4 h-4 ${isDarkMode ? 'text-purple-400' : 'text-indigo-500'}` 
              })}
              <span className={`text-xs ${
                isDarkMode ? 'text-purple-400/60' : 'text-indigo-500'
              } uppercase`}>
                {formatFieldLabel(key)}
              </span>
            </div>
            {isEditing ? (
              <input
                type="text"
                value={value || ''}
                onChange={(e) => onFieldChange(key, e.target.value)}
                className={`w-full ${
                  isDarkMode 
                    ? 'bg-purple-500/5 border-purple-500/10 focus:border-purple-500/20 text-white' 
                    : 'bg-white border-gray-300 focus:border-indigo-500 text-gray-800'
                  } border rounded-lg px-2 py-1 text-sm focus:outline-none mt-1`}
              />
            ) : (
              <div className={`font-light text-lg ${
                isDarkMode ? 'text-white' : 'text-gray-800'
              }`}>{value || 'Not set'}</div>
            )}
          </motion.div>
        ))}
      </div>
    </div>
  );
};

export default ProfileSection;