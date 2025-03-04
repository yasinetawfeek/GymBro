import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  AtSign, 
  Lock, 
  User, 
  LogIn, 
  LogOut, 
  Users,
  Sun,
  Moon 
} from 'lucide-react';

const AuthPage = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [username, setUsername] = useState('');
  const [isDarkMode, setIsDarkMode] = useState(false);

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

  const handleSubmit = (e) => {
    e.preventDefault();
    // Implement authentication logic here
    console.log(isLogin ? 'Login' : 'Register', { email, password, username });
  };

  const toggleDarkMode = () => {
    setIsDarkMode(!isDarkMode);
  };

  return (
    <div className={`fixed inset-0 ${isDarkMode 
        ? 'bg-gradient-to-br from-gray-900 to-gray-800 text-gray-100' 
        : 'bg-gradient-to-br from-indigo-50 to-purple-100 text-gray-900'
      } transition-colors duration-300 overflow-y-auto flex items-center justify-center p-4`}>
      {/* Dark Mode Toggle */}
      <button 
        onClick={toggleDarkMode}
        className={`fixed top-4 right-4 p-2 rounded-full transition-all duration-300 ${
          isDarkMode 
            ? 'bg-gray-700 text-yellow-400 hover:bg-gray-600' 
            : 'bg-gray-200 text-indigo-600 hover:bg-gray-300'
        }`}
        aria-label="Toggle Dark Mode"
      >
        {isDarkMode ? <Sun className="w-6 h-6" /> : <Moon className="w-6 h-6" />}
      </button>

      <motion.div 
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.3 }}
        className={`w-full max-w-md mx-4 ${
          isDarkMode 
            ? 'bg-gray-800 border border-gray-700 shadow-2xl' 
            : 'bg-white shadow-2xl'
        } rounded-2xl overflow-hidden`}
      >
        <div className="p-8">
          {/* Logo and Title */}
          <div className="flex items-center justify-center mb-6">
            <Users className={`w-12 h-12 ${
              isDarkMode ? 'text-purple-400' : 'text-indigo-600'
            } mr-3`} />
            <h1 className={`text-3xl font-bold ${
              isDarkMode ? 'text-gray-100' : 'text-gray-800'
            }`}>
              {isLogin ? 'Welcome Back' : 'Create Account'}
            </h1>
          </div>

          {/* Auth Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            {!isLogin && (
              <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="relative"
              >
                <label className={`block mb-2 flex items-center ${
                  isDarkMode ? 'text-gray-300' : 'text-gray-700'
                }`}>
                  <User className={`w-5 h-5 mr-2 ${
                    isDarkMode ? 'text-purple-400' : 'text-indigo-500'
                  }`} />
                  Username
                </label>
                <input 
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className={`w-full px-4 py-2 rounded-lg focus:outline-none focus:ring-2 ${
                    isDarkMode 
                      ? 'bg-gray-700 text-gray-100 border-gray-600 focus:ring-purple-500' 
                      : 'bg-white text-gray-900 border-gray-300 focus:ring-indigo-500'
                  }`}
                  placeholder="Choose a username"
                  required={!isLogin}
                />
              </motion.div>
            )}

            <div className="relative">
              <label className={`block mb-2 flex items-center ${
                isDarkMode ? 'text-gray-300' : 'text-gray-700'
              }`}>
                <AtSign className={`w-5 h-5 mr-2 ${
                  isDarkMode ? 'text-purple-400' : 'text-indigo-500'
                }`} />
                Email
              </label>
              <input 
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className={`w-full px-4 py-2 rounded-lg focus:outline-none focus:ring-2 ${
                  isDarkMode 
                    ? 'bg-gray-700 text-gray-100 border-gray-600 focus:ring-purple-500' 
                    : 'bg-white text-gray-900 border-gray-300 focus:ring-indigo-500'
                }`}
                placeholder="Enter your email"
                required
              />
            </div>

            <div className="relative">
              <label className={`block mb-2 flex items-center ${
                isDarkMode ? 'text-gray-300' : 'text-gray-700'
              }`}>
                <Lock className={`w-5 h-5 mr-2 ${
                  isDarkMode ? 'text-purple-400' : 'text-indigo-500'
                }`} />
                Password
              </label>
              <input 
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className={`w-full px-4 py-2 rounded-lg focus:outline-none focus:ring-2 ${
                  isDarkMode 
                    ? 'bg-gray-700 text-gray-100 border-gray-600 focus:ring-purple-500' 
                    : 'bg-white text-gray-900 border-gray-300 focus:ring-indigo-500'
                }`}
                placeholder="Enter your password"
                required
              />
            </div>

            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              type="submit"
              className={`w-full py-3 rounded-lg transition-colors flex items-center justify-center ${
                isDarkMode
                  ? 'bg-purple-600 text-white hover:bg-purple-700'
                  : 'bg-indigo-600 text-white hover:bg-indigo-700'
              }`}
            >
              {isLogin ? 'Log In' : 'Sign Up'}
              {isLogin ? <LogIn className="ml-2" /> : <Users className="ml-2" />}
            </motion.button>
          </form>

          {/* Toggle between Login and Register */}
          <div className="text-center mt-4">
            <button 
              onClick={() => setIsLogin(!isLogin)}
              className={`hover:underline ${
                isDarkMode 
                  ? 'text-purple-400 hover:text-purple-300 bg-gray-800' 
                  : 'text-blue-600 hover:text-blue-500 bg-white'
                }`}
            >
              {isLogin 
                ? 'Need an account? Sign Up' 
                : 'Already have an account? Log In'}
            </button>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default AuthPage;