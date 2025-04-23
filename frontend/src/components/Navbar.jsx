import React from 'react';
import { motion } from 'framer-motion';
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
  ArrowRight
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { useEffect, useRef, useState } from "react";

const NavBar = ({ isDarkMode,}) => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const navigate = useNavigate()
  const user = useAuth()
  // console.log('user:')
  // console.log(user)

  return (
    <header className={`${ 
      isDarkMode 
      ? 'bg-gray-800 border-gray-700' 
      : 'bg-white border-gray-200'
      } border-b shadow-sm transition-colors duration-300`}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
            <Home className={`w-8 h-8 ${
            isDarkMode ? 'text-purple-400' : 'text-indigo-600'
          } mr-2`} />
          <h1 className={`text-xl font-bold hidden sm:block ${isDarkMode?'text-white':'text-indigo-600'}`}>GymTracker </h1>
            </div>
            
            <div className="hidden md:flex space-x-8">
              <a href="#features" className={`${ isDarkMode? 'text-white hover:text-indigo-500' : 'text-gray-400 hover:text-indigo-500'} font-medium`}>Features</a>
              <a href="#how-it-works" className={`${ isDarkMode? 'text-white hover:text-indigo-500' : 'text-gray-400 hover:text-indigo-500'} font-medium`}>How It Works</a>
            </div>
            
            { user.user ? (
                <div className='md:flex space-x-8'>
                  <div>
                  <button onClick={()=>navigate("/dashboard")} className="hidden md:block bg-indigo-500 hover:bg-indigo-800 text-white font-semibold py-2 px-6 rounded-full transition duration-300">
                    Dashboard
                  </button>
                  </div>
                  <div>
                  <button onClick={()=>navigate("/account")} className="hidden md:block bg-indigo-500 hover:bg-indigo-800 text-white font-semibold py-2 px-6 rounded-full transition duration-300">
                    My Account
                  </button>
                  </div>
                </div>
                ) : (
                  <button onClick={()=>navigate("/auth")} className="hidden md:block bg-indigo-500 hover:bg-indigo-800 text-white font-semibold py-2 px-6 rounded-full transition duration-300">
                  Log in
                </button>
                )}

            <div className="md:hidden">
              <button 
                onClick={() => setIsMenuOpen(!isMenuOpen)}
                className="text-gray-500 hover:text-gray-700 focus:outline-none"
              >
                {isMenuOpen ? (
                  <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                ) : (
                  <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                  </svg>
                )}
              </button>
            </div>
          </div>
          
          {isMenuOpen && (
            <div className="md:hidden py-4">
              <div className="flex flex-col space-y-4">
                <a href="#features" className={`${ isDarkMode? 'text-white hover:text-indigo-500' : 'text-gray-400 hover:text-indigo-500'} font-medium`}>Features</a>
                <a href="#how-it-works" className={`${ isDarkMode? 'text-white hover:text-indigo-500' : 'text-gray-400 hover:text-indigo-500'} font-medium`}>How It Works</a>
                { user.user ? (
                <div className="flex flex-col space-y-4">
                <button className="bg-indigo-500 hover:bg-indigo-800 text-white font-semibold py-2 px-6 rounded-full transition duration-300 w-full">
                  My Account
                </button>
                <button className="bg-indigo-500 hover:bg-indigo-800 text-white font-semibold py-2 px-6 rounded-full transition duration-300 w-full">
                  Dashboard
                </button>
                </div>
                ) : (
                <button className="bg-indigo-500 hover:bg-indigo-800 text-white font-semibold py-2 px-6 rounded-full transition duration-300 w-full">
                  Log in
                </button> 
                )}
              </div>
            </div>
          )}
        </div>
      </header>
  );
};

export default NavBar;