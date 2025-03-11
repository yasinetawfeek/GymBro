import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Menu, X, User, Users as UsersIcon, 
  Terminal, Edit3, Trash2, Save,
  Brain, ChevronRight, ActivitySquare,
  TrendingUp, Calendar, Weight, Timer,
  Mail, Target, Award, MapPin,
  UserCircle, Dumbbell, Medal, 
  Clock, Ruler, Heart
} from 'lucide-react';

const AccountManagement = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [userRole] = useState('user');
  const [activePage, setActivePage] = useState('profile');
  const [isEditing, setIsEditing] = useState(false);

  const rolePages = {
    user: [
      { id: 'profile', label: 'My Profile', icon: User },
      { id: 'stats', label: 'Fitness Stats', icon: ActivitySquare }
    ],
    admin: [
      { id: 'users', label: 'User Management', icon: UsersIcon },
      { id: 'stats', label: 'Analytics', icon: TrendingUp }
    ],
    engineer: [
      { id: 'models', label: 'ML Models', icon: Brain },
      { id: 'api', label: 'API Access', icon: Terminal }
    ]
  };

  const [userData, setUserData] = useState({
    basicInfo: {
      fullName: 'Alex Johnson',
      email: 'alex@example.com',
      location: 'New York, USA',
      memberSince: 'March 2024'
    },
    fitnessProfile: {
      height: '175 cm',
      weight: '70 kg',
      bodyFat: '15%',
      fitnessLevel: 'Intermediate'
    },
    preferences: {
      primaryGoal: 'Build Muscle',
      workoutFrequency: '4x/week',
      preferredTime: 'Morning',
      focusAreas: 'Full Body'
    },
    achievements: {
      workoutsCompleted: '48',
      daysStreak: '15',
      personalBests: '12',
      points: '2,450'
    }
  });

  const [fitnessStats] = useState({
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

  const renderProfileSection = (title, data, icons) => (
    <div className="bg-gray-700 backdrop-blur-sm rounded-lg p-4">
      <h3 className="text-sm text-purple-400/60 uppercase tracking-wider mb-4">{title}</h3>
      <div className="grid grid-cols-2 gap-4">
        {Object.entries(data).map(([key, value]) => (
          <motion.div
            key={key}
            whileHover={{ scale: 1.02 }}
            className="bg-purple-500/5 p-3 rounded-lg"
          >
            <div className="flex items-center space-x-2 mb-1">
              {icons[key] && React.createElement(icons[key], { className: "w-4 h-4 text-purple-400" })}
              <span className="text-xs text-purple-400/60 uppercase">
                {key.replace(/([A-Z])/g, ' $1').trim()}
              </span>
            </div>
            {isEditing ? (
              <input
                type="text"
                value={value}
                onChange={(e) => {
                  const newData = { ...userData };
                  newData[title.toLowerCase()][key] = e.target.value;
                  setUserData(newData);
                }}
                className="w-full bg-purple-500/5 border border-purple-500/10 rounded-lg px-2 py-1 text-sm focus:outline-none focus:border-purple-500/20 mt-1"
              />
            ) : (
              <div className="font-light text-lg">{value}</div>
            )}
          </motion.div>
        ))}
      </div>
    </div>
  );

  const renderStatsPage = () => (
    <div className="space-y-6">
      <div className="flex items-center justify-between mb-8">
        <h2 className="text-2xl font-light">Fitness <span className="text-purple-400 font-medium">Overview</span></h2>
        <div className="flex items-center space-x-2 text-purple-400/60 text-sm">
          <Calendar className="w-4 h-4" />
          <span>March 2025</span>
        </div>
      </div>
      
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {Object.entries(fitnessStats.progressMetrics).map(([key, value]) => (
          <motion.div
            key={key}
            whileHover={{ scale: 1.02 }}
            className="bg-gray-700 backdrop-blur-sm p-4 rounded-lg"
          >
            <div className="flex items-center space-x-2 mb-2">
              {key === 'monthlyWorkouts' && <Calendar className="w-4 h-4 text-purple-400" />}
              {key === 'avgDuration' && <Timer className="w-4 h-4 text-purple-400" />}
              {key === 'weightProgress' && <Weight className="w-4 h-4 text-purple-400" />}
              {key === 'streakDays' && <TrendingUp className="w-4 h-4 text-purple-400" />}
            </div>
            <div className="text-2xl font-light">{value}</div>
            <div className="text-xs text-purple-400/60 uppercase tracking-wider mt-1">
              {key.replace(/([A-Z])/g, ' $1').trim()}
            </div>
          </motion.div>
        ))}
      </div>

      <div className="bg-gray-700 backdrop-blur-sm rounded-lg overflow-hidden">
        <div className="p-4">
          <h3 className="text-lg font-light mb-4">Recent Activity</h3>
          <div className="space-y-3">
            {fitnessStats.recentWorkouts.map((workout, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className="flex items-center justify-between p-3 bg-purple-500/5 rounded-lg hover:bg-purple-500/10 transition-colors"
              >
                <div className="flex items-center space-x-3">
                  <div className="bg-purple-500/20 p-1.5 rounded-lg">
                    <ActivitySquare className="w-4 h-4 text-purple-400" />
                  </div>
                  <div>
                    <div className="font-light">{workout.type}</div>
                    <div className="text-xs text-gray-400">{workout.date}</div>
                  </div>
                </div>
                <div className="text-right text-sm">
                  <div className="text-purple-400">{workout.duration}</div>
                  <div className="text-xs text-gray-400">{workout.intensity}</div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );

  const renderProfilePage = () => {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between mb-8">
          <h2 className="text-2xl font-light">Profile <span className="text-purple-400 font-medium">Overview</span></h2>
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setIsEditing(!isEditing)}
            className="bg-purple-500/10 hover:bg-purple-500/20 px-4 py-2 rounded-lg flex items-center space-x-2 text-sm"
          >
            {isEditing ? <Save className="w-4 h-4" /> : <Edit3 className="w-4 h-4" />}
            <span className="font-light">{isEditing ? 'Save' : 'Edit'}</span>
          </motion.button>
        </div>

        <div className="space-y-6">
          {renderProfileSection('BasicInfo', userData.basicInfo, icons)}
          {renderProfileSection('FitnessProfile', userData.fitnessProfile, icons)}
          {renderProfileSection('Preferences', userData.preferences, icons)}
          {renderProfileSection('Achievements', userData.achievements, icons)}
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
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white">
      <nav className="backdrop-blur-md bg-black/10 border-b border-white/5 sticky top-0 z-50">
        <div className="max-w-6xl mx-auto flex justify-between items-center p-4">
          <span className="text-lg font-light">gym<span className="text-purple-400">tracker</span></span>
          <motion.button 
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setIsMenuOpen(!isMenuOpen)}
            className="lg:hidden p-1.5 rounded-lg bg-white/5"
          >
            {isMenuOpen ? <X size={20} /> : <Menu size={20} />}
          </motion.button>
        </div>
      </nav>

      <div className="max-w-6xl mx-auto px-4 py-6 relative">
        <div className="flex flex-col lg:flex-row gap-6">
          <AnimatePresence>
            <motion.div 
              className={`lg:w-64 ${isMenuOpen ? 'block fixed top-20 left-4 right-4 z-40 bg-gray-900/95 backdrop-blur-md rounded-xl p-4 border border-white/5' : 'hidden'} 
                         lg:block lg:fixed lg:top-24 lg:bottom-6 lg:overflow-y-auto
                         scrollbar-thin scrollbar-thumb-purple-500/20 scrollbar-track-transparent`}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              <div className="space-y-2">
                {rolePages[userRole].map((page) => (
                  <motion.div
                    key={page.id}
                    whileHover={{ scale: 1.01 }}
                    onClick={() => {
                      setActivePage(page.id);
                      if (window.innerWidth < 1024) setIsMenuOpen(false);
                    }}
                    className={`flex items-center space-x-3 px-4 py-3 rounded-lg cursor-pointer
                      ${activePage === page.id ? 'bg-purple-500/10 text-purple-400' : 'bg-white/5 hover:bg-white/10'}`}
                  >
                    <page.icon className="w-4 h-4" />
                    <span className="text-sm font-light">{page.label}</span>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          </AnimatePresence>

          <div className="lg:ml-72 flex-1">
            <motion.div 
              className="rounded-xl backdrop-blur-sm bg-white/5 border border-white/5 p-6"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              {activePage === 'profile' ? renderProfilePage() : renderStatsPage()}
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AccountManagement;
