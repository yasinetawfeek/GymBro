import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  UserCircle, Mail, MapPin, Calendar,
  Ruler, Weight, Heart, Medal, Target, 
  Clock, Dumbbell, ActivitySquare, TrendingUp,
  Award
} from 'lucide-react';

import NavBar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import ProfileOverview from '../components/ProfileOverview';
import FitnessStats from '../components/FitnessStats';
import UserManagement from '../components/UserManagement';
import UserDetailsPage from '../components/UserDetailsPage';

import { useAuth } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';

const AccountManagement = () => {
  const [isDarkMode, setIsDarkMode] = useState(false);
    
  // Check system preference on initial load
  useEffect(() => {
    const prefersDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
    setIsDarkMode(prefersDarkMode);
    
  }, [navigate]);
  
  // Update dark mode class on body
  useEffect(() => {
    if (isDarkMode) {
      document.body.classList.add('dark');
    } else {
      document.body.classList.remove('dark');
    }
  }, [isDarkMode]);
  
  const toggleDarkMode = () => {
    setIsDarkMode(!isDarkMode);
  };

  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [activePage, setActivePage] = useState('profile');
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

  // Get user data and functions from auth context
  const { user} = useAuth();
  const navigate = useNavigate();

  // Determine user role from backend data
  const getUserRole = () => {
    if (!user) return 'user';
    
    // Check if the user has admin permissions
    // This might need to be adjusted based on how roles are stored in your backend
    if (user.basicInfo?.isAdmin || user.basicInfo?.role === 'Admin') {
      return 'admin';
    }
    return 'user';
  };
  
  const userRole = getUserRole();

  // Set initial active page based on role
  useEffect(() => {
    if (userRole === 'admin') {
      setActivePage('users');
    } else {
      setActivePage('profile');
    }
  }, [userRole]);

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

  // Handle the save action when profile is edited
  const handleProfileUpdate = async (updatedData) => {
    try {
      await updateProfile(updatedData);
      setIsEditing(false);
    } catch (err) {
      console.error('Failed to update profile:', err);
    }
  };

  // Show loading spinner while user data is being fetched
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-900">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-purple-500"></div>
      </div>
    );
  }

  const renderActivePage = () => {
    switch (activePage) {
      case 'profile':
        return (
          <ProfileOverview 
            userData={user}
            setUserData={handleProfileUpdate}
            isEditing={isEditing}
            setIsEditing={setIsEditing}
            icons={icons}
          />
        );
      case 'stats':
        return <FitnessStats fitnessStats={fitnessStats} />;
      case 'users':
        return <UserManagement />;
      case 'userDetails':
        return <UserDetailsPage />;
      default:
        return <div>Page not found</div>;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white">
      <NavBar isDarkMode={isDarkMode} />
      <div className="max-w-6xl mx-auto px-4 py-6 relative">
        {error && (
          <div className="mb-4 bg-red-500/20 text-red-300 p-3 rounded-lg">
            {error}
          </div>
        )}

        <div className="flex flex-col lg:flex-row gap-6">
          <Sidebar 
            isMenuOpen={isMenuOpen}
            setIsMenuOpen={setIsMenuOpen}
            userRole={userRole}
            activePage={activePage}
            setActivePage={setActivePage}
          />

          <div className="lg:ml-72 flex-1">
            <motion.div 
              className="rounded-xl backdrop-blur-sm bg-white/5 border border-white/5 p-6"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              key={activePage}
            >
              {renderActivePage()}
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AccountManagement;