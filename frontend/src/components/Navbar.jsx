import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Home, 
  Users, 
  Settings, 
  LogOut, 
  Sun,
  Moon,
  Bell,
  Menu,
  X,
  User,
  MessageSquare,
  FileText,
  ArrowRight,
  ChevronDown
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const NavBar = ({ isDarkMode, toggleDarkMode }) => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const navigate = useNavigate();
  const user = useAuth();
  
  const handleLogout = () => {
    user.logout();
    navigate('/auth');
    setIsProfileOpen(false);
    setIsMenuOpen(false);
  };
  
  // Loading state
  if (user.loading) {
    return (
      <header className={`
        ${isDarkMode ? 'bg-gray-900/95 border-gray-800' : 'bg-white/95 border-gray-200'}
        border-b backdrop-blur-md shadow-sm transition-colors duration-300 sticky top-0 z-50`}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <Home className={`w-6 h-6 ${
                isDarkMode ? 'text-purple-400' : 'text-indigo-600'
              } mr-2`} />
              <a href="/" className={`text-xl font-bold ${isDarkMode ? 'text-white' : 'text-indigo-600'}`}>
                GymTracker
              </a>
            </div>
            <div className="flex items-center">
              <motion.div 
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                className="rounded-full h-4 w-4 border-t-2 border-b-2 border-purple-500 mr-2"
              />
              <span className={`text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                Loading...
              </span>
            </div>
          </div>
        </div>
      </header>
    );
  }

  return (
    <header className={`
      ${isDarkMode ? 'bg-gray-900/95 border-gray-800' : 'bg-white/95 border-gray-200'}
      border-b backdrop-blur-md shadow-sm transition-colors duration-300 sticky top-0 z-50`}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center py-4">
          <motion.div 
            className="flex items-center"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Home className={`w-6 h-6 ${
              isDarkMode ? 'text-purple-400' : 'text-indigo-600'
            } mr-2`} />
            <a href="/" className={`text-xl font-bold ${isDarkMode ? 'text-white' : 'text-indigo-600'}`}>
              GymTracker
            </a>
          </motion.div>
          
          <div className="hidden md:flex space-x-6">
            <motion.a 
              href="#features" 
              className={`${isDarkMode ? 'text-white hover:text-purple-400' : 'text-gray-600 hover:text-indigo-500'} 
                font-medium text-sm transition-colors`}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Features
            </motion.a>
            <motion.a 
              href="#how-it-works" 
              className={`${isDarkMode ? 'text-white hover:text-purple-400' : 'text-gray-600 hover:text-indigo-500'} 
                font-medium text-sm transition-colors`}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              How It Works
            </motion.a>
          </div>
          
          <div className="flex items-center space-x-4">
            <motion.button 
              onClick={toggleDarkMode}
              className={`p-2 rounded-full ${
                isDarkMode ? 'bg-gray-800 text-purple-400' : 'bg-gray-100 text-indigo-600'
              }`}
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
            >
              {isDarkMode ? <Sun size={18} /> : <Moon size={18} />}
            </motion.button>
            
            {user.user ? (
              <div className="hidden md:flex items-center space-x-4 relative">
                <motion.button 
                  onClick={() => setIsProfileOpen(!isProfileOpen)} 
                  className={`bg-gradient-to-r ${
                    isDarkMode ? 'from-purple-500 to-indigo-600' : 'from-indigo-500 to-purple-600'
                  } text-white font-medium py-2 px-5 rounded-lg text-sm transition duration-300 flex items-center space-x-2`}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <User size={16} className="mr-1" />
                  <span>{user.user?.username || 'User'}</span>
                  <ChevronDown size={14} />
                </motion.button>
                
                <AnimatePresence>
                  {isProfileOpen && (
                    <motion.div
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: 10 }}
                      className={`absolute right-0 top-12 w-48 p-2 rounded-lg shadow-lg z-50 ${
                        isDarkMode ? 'bg-gray-800 border border-gray-700' : 'bg-white border border-gray-200'
                      }`}
                    >
                      <div 
                        onClick={() => navigate("/settings?page=profile")}
                        className={`flex items-center space-x-2 px-4 py-2 rounded-md cursor-pointer ${
                          isDarkMode ? 'hover:bg-gray-700 text-white' : 'hover:bg-gray-100 text-gray-700'
                        }`}
                      >
                        <Settings size={16} />
                        <span>Settings</span>
                      </div>
                      <div 
                        onClick={handleLogout}
                        className={`flex items-center space-x-2 px-4 py-2 rounded-md cursor-pointer ${
                          isDarkMode ? 'hover:bg-gray-700 text-white' : 'hover:bg-gray-100 text-gray-700'
                        }`}
                      >
                        <LogOut size={16} />
                        <span>Logout</span>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
                
                {/* <motion.button 
                  onClick={() => navigate("/settings")}
                  className={`${
                    isDarkMode ? 'bg-gray-800 text-white hover:bg-gray-700' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  } font-medium py-2 px-4 rounded-lg text-sm transition duration-300`}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Settings size={18} />
                </motion.button> */}
              </div>
            ) : (
              <motion.button 
                onClick={() => navigate("/auth")} 
                className={`hidden md:flex bg-gradient-to-r ${
                  isDarkMode ? 'from-purple-500 to-indigo-600' : 'from-indigo-500 to-purple-600'
                } text-white font-medium py-2 px-5 rounded-lg text-sm transition duration-300 items-center space-x-2`}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <span>Log in</span>
                <ArrowRight size={14} />
              </motion.button>
            )}

            <motion.button 
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="md:hidden p-2 rounded-full bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-white"
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
            >
              {isMenuOpen ? <X size={20} /> : <Menu size={20} />}
            </motion.button>
          </div>
        </div>
      </div>
      
      <AnimatePresence>
        {isMenuOpen && (
          <motion.div 
            className={`md:hidden fixed inset-x-4 top-20 z-40 ${
              isDarkMode ? 'bg-gray-900/95 border-gray-800' : 'bg-white/95 border-gray-200'
            } backdrop-blur-md rounded-xl p-4 border shadow-lg`}
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.2 }}
          >
            <div className="flex flex-col space-y-4">
              <a 
                href="#features" 
                className={`${isDarkMode ? 'text-white hover:text-purple-400' : 'text-gray-600 hover:text-indigo-500'} 
                  font-medium text-sm py-2 px-3 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors`}
                onClick={() => setIsMenuOpen(false)}
              >
                Features
              </a>
              <a 
                href="#how-it-works" 
                className={`${isDarkMode ? 'text-white hover:text-purple-400' : 'text-gray-600 hover:text-indigo-500'} 
                  font-medium text-sm py-2 px-3 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors`}
                onClick={() => setIsMenuOpen(false)}
              >
                How It Works
              </a>
              <a
                href="/workout" 
                className={`${isDarkMode ? 'text-white hover:text-purple-400' : 'text-gray-600 hover:text-indigo-500'} 
                  font-medium text-sm py-2 px-3 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors`}
                onClick={() => setIsMenuOpen(false)}
              >
                Workout
              </a>
              
              {user.user ? (
                <div className="space-y-2 pt-2 border-t border-gray-200 dark:border-gray-700">
                  <div className={`w-full ${
                    isDarkMode ? 'bg-gray-800' : 'bg-gray-100'
                  } p-3 rounded-lg flex items-center space-x-2`}>
                    <User size={16} className={isDarkMode ? 'text-purple-400' : 'text-indigo-600'} />
                    <span className={isDarkMode ? 'text-white' : 'text-gray-700'}>
                      {user.user?.username || 'User'}
                    </span>
                  </div>
                  <button 
                    onClick={() => {
                      navigate("/settings?page=profile");
                      setIsMenuOpen(false);
                    }} 
                    className={`w-full ${
                      isDarkMode ? 'bg-gray-800 text-white hover:bg-gray-700' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    } font-medium py-2 px-4 rounded-lg text-sm transition duration-300 flex items-center justify-between`}
                  >
                    <span>Settings</span>
                    <Settings size={14} />
                  </button>
                  <button 
                    onClick={handleLogout} 
                    className={`w-full bg-gradient-to-r ${
                      isDarkMode ? 'from-red-500 to-red-600' : 'from-red-500 to-red-600'
                    } text-white font-medium py-2 px-4 rounded-lg text-sm transition duration-300 flex items-center justify-between`}
                  >
                    <span>Logout</span>
                    <LogOut size={14} />
                  </button>
                </div>
              ) : (
                <button 
                  onClick={() => {
                    navigate("/auth");
                    setIsMenuOpen(false);
                  }} 
                  className={`w-full bg-gradient-to-r ${
                    isDarkMode ? 'from-purple-500 to-indigo-600' : 'from-indigo-500 to-purple-600'
                  } text-white font-medium py-2 px-4 rounded-lg text-sm transition duration-300 flex items-center justify-between`}
                >
                  <span>Log in</span>
                  <ArrowRight size={14} />
                </button>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
};

export default NavBar;