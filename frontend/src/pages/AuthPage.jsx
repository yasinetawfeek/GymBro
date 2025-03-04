// src/pages/AuthPage.jsx
import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { AtSign, Lock, User, Users, Sun, Moon, Palette } from 'lucide-react';

const colorThemes = [
  {
    name: 'Red',
    primary: 'red',
    gradient: { light: 'from-red-200 to-red-300', dark: 'from-red-700 to-red-800' }
  },
  {
    name: 'Orange',
    primary: 'orange',
    gradient: { light: 'from-orange-200 to-orange-300', dark: 'from-orange-700 to-orange-800' }
  },
  {
    name: 'Yellow',
    primary: 'yellow',
    gradient: { light: 'from-yellow-200 to-yellow-300', dark: 'from-yellow-700 to-yellow-800' }
  },
  {
    name: 'Green',
    primary: 'green',
    gradient: { light: 'from-green-200 to-green-300', dark: 'from-green-700 to-green-800' }
  },
  {
    name: 'Blue',
    primary: 'blue',
    gradient: { light: 'from-blue-200 to-blue-300', dark: 'from-blue-700 to-blue-800' }
  },
  {
    name: 'Indigo',
    primary: 'indigo',
    gradient: { light: 'from-indigo-200 to-indigo-300', dark: 'from-indigo-700 to-indigo-800' }
  },
  {
    name: 'Violet',
    primary: 'purple',
    gradient: { light: 'from-purple-200 to-purple-300', dark: 'from-purple-700 to-purple-800' }
  }
];

export default function AuthPage() {
  const { login } = useAuth();
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [username, setUsername] = useState('');
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [currentTheme, setCurrentTheme] = useState(colorThemes[0]);
  const [isThemeMenuOpen, setIsThemeMenuOpen] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const prefersDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
    setIsDarkMode(prefersDarkMode);
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      if (isLogin) {
        await login({ email, password });
        navigate('/dashboard');
      } else {
        navigate('/dashboard');
      }
    } catch (error) {
      console.error('Authentication error:', error);
    }
  };

  const NeuInput = ({ icon: Icon, type, value, onChange, placeholder, required }) => (
    <div className="relative mb-4">
      <div className={`absolute left-3 top-1/2 -translate-y-1/2 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
        <Icon className="w-5 h-5" />
      </div>
      <input
        type={type}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        required={required}
        className={`w-full pl-10 pr-4 py-3 rounded-2xl outline-none transition-all duration-300
          ${isDarkMode
            ? 'bg-gray-800 text-gray-200 shadow-[inset_2px_2px_5px_rgba(0,0,0,0.4),inset_-2px_-2px_5px_rgba(255,255,255,0.05)]'
            : 'bg-gray-100 text-gray-800 shadow-[inset_2px_2px_5px_rgba(0,0,0,0.1),inset_-2px_-2px_5px_rgba(255,255,255,0.9)]'
          }
          focus:ring-2 focus:ring-${currentTheme.primary}-500`}
      />
    </div>
  );

  return (
    <div className={`min-h-screen flex items-center justify-center p-4 transition-colors duration-300
      ${isDarkMode
        ? 'bg-gradient-to-br from-gray-900 to-gray-800 text-gray-100'
        : 'bg-gradient-to-br from-gray-100 to-gray-200 text-gray-900'
      }`}>

      <div className="absolute top-6 right-6 flex items-center space-x-4">
        <div className="relative">
          <button
            onClick={() => setIsThemeMenuOpen(!isThemeMenuOpen)}
            className={`w-12 h-12 rounded-full flex items-center justify-center
              transition-all duration-300 focus:outline-none
              ${isDarkMode
                ? 'shadow-[4px_4px_6px_rgba(0,0,0,0.4),-4px_-4px_6px_rgba(255,255,255,0.05)] bg-gray-900 text-gray-400'
                : 'shadow-[4px_4px_6px_rgba(0,0,0,0.1),-4px_-4px_6px_rgba(255,255,255,0.9)] bg-gray-100 text-gray-600'
              }`}>
            <Palette className="w-6 h-6" />
          </button>

          {isThemeMenuOpen && (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              className={`absolute top-full right-0 mt-2 p-2 rounded-lg flex space-x-2
                ${isDarkMode ? 'bg-gray-800 border border-gray-700' : 'bg-white shadow-lg'}`}>
              {colorThemes.map((theme) => (
                <button
                  key={theme.name}
                  onClick={() => setCurrentTheme(theme)}
                  className={`w-8 h-8 rounded-full
                    bg-gradient-to-br ${isDarkMode ? theme.gradient.dark : theme.gradient.light}
                    ${currentTheme.name === theme.name ? 'ring-2 ring-offset-2 ring-gray-500' : ''}`}
                />
              ))}
            </motion.div>
          )}
        </div>

        <button
          onClick={() => setIsDarkMode(!isDarkMode)}
          className={`w-12 h-12 rounded-full flex items-center justify-center
            transition-all duration-300 focus:outline-none
            ${isDarkMode
              ? 'shadow-[4px_4px_6px_rgba(0,0,0,0.4),-4px_-4px_6px_rgba(255,255,255,0.05)] bg-gray-900 text-yellow-400'
              : 'shadow-[4px_4px_6px_rgba(0,0,0,0.1),-4px_-4px_6px_rgba(255,255,255,0.9)] bg-gray-100 text-indigo-600'
            }`}>
          {isDarkMode ? <Sun className="w-6 h-6" /> : <Moon className="w-6 h-6" />}
        </button>
      </div>

      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        className={`w-full max-w-md p-8 rounded-2xl transition-all duration-300
          ${isDarkMode
            ? 'shadow-[8px_8px_12px_rgba(0,0,0,0.4),-8px_-8px_12px_rgba(255,255,255,0.05)] bg-gray-900'
            : 'shadow-[8px_8px_12px_rgba(0,0,0,0.1),-8px_-8px_12px_rgba(255,255,255,0.9)] bg-gray-100'
          }`}>
        <div className="text-center mb-8">
          <div className={`mx-auto w-20 h-20 rounded-full flex items-center justify-center mb-4
            ${isDarkMode
              ? 'shadow-[4px_4px_6px_rgba(0,0,0,0.4),-4px_-4px_6px_rgba(255,255,255,0.05)] bg-gray-900'
              : 'shadow-[4px_4px_6px_rgba(0,0,0,0.1),-4px_-4px_6px_rgba(255,255,255,0.9)] bg-gray-100'
            }`}>
            <Users className={`w-10 h-10 text-${currentTheme.primary}-500`} />
          </div>
          <h2 className={`text-3xl font-semibold ${isDarkMode ? 'text-gray-200' : 'text-gray-800'}`}>
            {isLogin ? 'Welcome Back' : 'Create Account'}
          </h2>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {!isLogin && (
            <NeuInput
              icon={User}
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Username"
              required={!isLogin}
            />
          )}

          <NeuInput
            icon={AtSign}
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Email"
            required
          />

          <NeuInput
            icon={Lock}
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Password"
            required
          />

          <button
            type="submit"
            className={`w-full py-3 rounded-lg transition-all duration-300 focus:outline-none
              bg-gradient-to-br ${isDarkMode ? currentTheme.gradient.dark : currentTheme.gradient.light}
              text-white hover:opacity-90 active:opacity-100 ${isDarkMode ? 'opacity-80' : ''}`}>
            {isLogin ? 'Log In' : 'Sign Up'}
          </button>
        </form>

        <div className="text-center mt-6">
          <button
            onClick={() => setIsLogin(!isLogin)}
            className={`text-sm transition-colors text-${currentTheme.primary}-600 hover:text-${currentTheme.primary}-700`}>
            {isLogin ? 'Need an account? Sign Up' : 'Already have an account? Log In'}
          </button>
        </div>
      </motion.div>
    </div>
  );
}
