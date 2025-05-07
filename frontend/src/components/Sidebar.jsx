import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  User, Users as UsersIcon, Terminal, 
  Brain, ActivitySquare, TrendingUp,
  UserCircle, CheckCircle, Clock, AlertTriangle,
  CreditCard, Settings, FileText, Database,
  DollarSign, PieChart, BarChart
} from 'lucide-react';

const Sidebar = ({ isMenuOpen, userRole, activePage, setActivePage, setIsMenuOpen, isDarkMode = true, isApproved = true }) => {
  // Define role-specific pages
  const rolePages = {
    Customer: [
      { id: 'profile', label: 'My Profile', icon: User },
      { id: 'billing', label: 'Billing', icon: CreditCard },
      { id: 'settings', label: 'Settings', icon: Settings }
    ],
    Admin: [
      { id: 'profile', label: 'My Profile', icon: User },
      { id: 'users', label: 'User Management', icon: UsersIcon },
      { id: 'approvals', label: 'Approval Requests', icon: Clock },
      { id: 'billing', label: 'Billing Overview', icon: DollarSign },
      { id: 'billingActivity', label: 'Billing Activity', icon: BarChart },
      // { id: 'analytics', label: 'Analytics', icon: PieChart },
      { id: 'performance', label: 'Model Performance', icon: TrendingUp },
      { id: 'models', label: 'ML Models', icon: Brain },
      // { id: 'settings', label: 'System Settings', icon: Settings }
    ],
    'AI Engineer': [
      { id: 'profile', label: 'My Profile', icon: User },
      { id: 'models', label: 'ML Models', icon: Brain },
      { id: 'performance', label: 'Model Performance', icon: TrendingUp },
      // { id: 'data', label: 'Training Data', icon: Database },
      // { id: 'api', label: 'API Access', icon: Terminal },
      // { id: 'docs', label: 'Documentation', icon: FileText }
    ]
  };

  // Get pages for current role, default to Customer if role not found
  const currentRolePages = rolePages[userRole] || rolePages.Customer;

  // Handler for page navigation
  const handlePageSelect = (pageId) => {
    setActivePage(pageId);
    if (window.innerWidth < 1024) setIsMenuOpen(false);
  };

  return (
    <AnimatePresence>
      <motion.div 
        className={`lg:w-64 ${isMenuOpen ? 'block fixed top-20 left-4 right-4 z-40' : 'hidden'} 
                   ${isDarkMode 
                     ? 'backdrop-blur-md ' 
                     : 'backdrop-blur-md '
                   } 
                   ${isMenuOpen ? 'rounded-xl p-4' : ''}
                   lg:block lg:fixed lg:top-24 lg:bottom-6 lg:overflow-y-auto
                   scrollbar-thin scrollbar-thumb-purple-500/20 scrollbar-track-transparent`}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        {/* Approval Status Indicator */}
        {userRole === 'AI Engineer' && !isApproved && (
          <div className={`mb-4 p-3 rounded-lg flex items-center space-x-2 ${
            isDarkMode ? 'bg-amber-800/40 text-amber-200' : 'bg-amber-50 text-amber-800'
          }`}>
            <AlertTriangle size={16} className="flex-shrink-0" />
            <span className="text-xs">Awaiting admin approval</span>
          </div>
        )}
        
        {userRole === 'AI Engineer' && isApproved && (
          <div className={`mb-4 p-3 rounded-lg flex items-center space-x-2 ${
            isDarkMode ? 'bg-green-800/40 text-green-200' : 'bg-green-50 text-green-800'
          }`}>
            <CheckCircle size={16} className="flex-shrink-0" />
            <span className="text-xs">Account approved</span>
          </div>
        )}
        
        {/* Role Indicator */}
        <div className={`mb-4 px-4 py-2 rounded-lg text-center ${
          isDarkMode ? 'bg-gray-800/50 text-white' : 'bg-gray-100 text-gray-800'
        }`}>
          <span className="text-xs font-medium">{userRole} Dashboard</span>
        </div>
        
        <div className="space-y-2">
          {currentRolePages.map((page) => (
            <motion.div
              key={page.id}
              whileHover={{ scale: 1.01 }}
              onClick={() => handlePageSelect(page.id)}
              className={`flex items-center space-x-3 px-4 py-3 rounded-lg cursor-pointer
                ${activePage === page.id 
                  ? isDarkMode 
                    ? 'bg-purple-500/10 text-purple-400' 
                    : 'bg-indigo-50 text-indigo-600 shadow-sm'
                  : isDarkMode
                    ? 'bg-white/5 hover:bg-white/10 text-white'
                    : 'bg-gray-50 hover:bg-gray-100 text-gray-700 shadow-sm'
                }`}
            >
              <page.icon className="w-4 h-4" />
              <span className="text-sm font-light">{page.label}</span>
            </motion.div>
          ))}
        </div>
      </motion.div>
    </AnimatePresence>
  );
};

export default Sidebar;