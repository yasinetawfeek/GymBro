import React from 'react';
import { motion } from 'framer-motion';
import { Users, UserCheck, Shield } from 'lucide-react';

const RoleSelection = ({ selectedRole, onRoleChange, isDarkMode }) => {
  const roles = [
    { 
      id: 'Customer', 
      name: 'Customer', 
      description: 'Access to ML prediction services',
      icon: Users
    },
    { 
      id: 'AI Engineer', 
      name: 'AI Engineer', 
      description: 'Can train and update ML models',
      icon: UserCheck,
      requiresApproval: true,
      emailDomain: 'ufcfur_15_3.com'
    },
    { 
      id: 'Admin', 
      name: 'Admin', 
      description: 'Full system access and user management',
      icon: Shield,
      requiresApproval: true,
      emailDomain: 'ufcfur_15_3.com'
    }
  ];

  return (
    <div className="space-y-4">
      <label className={`block text-sm font-medium ${
        isDarkMode ? 'text-gray-300' : 'text-gray-700'
      }`}>
        Select Your Role
      </label>
      
      <div className="grid grid-cols-1 gap-3">
        {roles.map((role) => (
          <motion.div
            key={role.id}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => onRoleChange(role.id)}
            className={`
              relative rounded-lg p-4 cursor-pointer border-2 transition-all duration-200
              ${selectedRole === role.id
                ? isDarkMode 
                  ? 'border-purple-500 bg-purple-900/30'
                  : 'border-indigo-500 bg-indigo-50'
                : isDarkMode
                  ? 'border-gray-700 hover:border-gray-600'
                  : 'border-gray-200 hover:border-gray-300'
              }
            `}
          >
            <div className="flex items-center">
              <div className={`
                flex-shrink-0 p-2 rounded-full
                ${selectedRole === role.id
                  ? isDarkMode ? 'bg-purple-900 text-purple-300' : 'bg-indigo-100 text-indigo-600'
                  : isDarkMode ? 'bg-gray-800 text-gray-400' : 'bg-gray-100 text-gray-500'
                }
              `}>
                {React.createElement(role.icon, { className: 'w-5 h-5' })}
              </div>
              
              <div className="ml-4">
                <h3 className={`font-medium ${
                  isDarkMode ? 'text-white' : 'text-gray-900'
                }`}>{role.name}</h3>
                <p className={`text-sm ${
                  isDarkMode ? 'text-gray-400' : 'text-gray-500'
                }`}>{role.description}</p>
                
                {role.requiresApproval && (
                  <div className={`text-xs mt-1 font-medium ${
                    isDarkMode ? 'text-amber-400' : 'text-amber-600'
                  }`}>
                    Requires admin approval
                  </div>
                )}
                
                {role.emailDomain && (
                  <div className={`text-xs mt-1 ${
                    isDarkMode ? 'text-gray-500' : 'text-gray-600'
                  }`}>
                    Required email domain: @{role.emailDomain}
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
};

export default RoleSelection; 