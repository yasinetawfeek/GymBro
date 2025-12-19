import { Platform } from 'react-native';

// API Configuration
const defaultHost = Platform.OS === 'android' ? 'http://10.0.2.2:8000' : 'http://localhost:8000';
export const API_URL = process.env.API_URL || `${defaultHost}/api`;
export const AI_URL = process.env.AI_URL || (Platform.OS === 'android' ? 'http://10.0.2.2:5000' : 'http://localhost:5000');

// App Configuration
export const APP_CONFIG = {
  name: 'GymBro',
  version: '1.0.0',
  description: 'AI-Powered Fitness Tracking App',
};

// Feature Flags
export const FEATURES = {
  POSE_DETECTION: true,
  WORKOUT_CLASSIFICATION: true,
  MUSCLE_GROUP_DETECTION: true,
  REAL_TIME_FEEDBACK: true,
  WORKOUT_HISTORY: true,
  PROGRESS_TRACKING: true,
  PUSH_NOTIFICATIONS: true,
};

// Camera Configuration
export const CAMERA_CONFIG = {
  quality: 0.8,
  aspect: [16, 9],
  facing: 'front',
  flashMode: 'off',
  autoFocus: true,
  whiteBalance: 'auto',
  zoom: 0,
};

// Workout Types Mapping
export const WORKOUT_MAP = {
  0: "Barbell Bicep Curl",
  1: "Bench Press",
  2: "Chest Fly Machine",
  3: "Deadlift",
  4: "Decline Bench Press",
  5: "Hammer Curl",
  6: "Hip Thrust",
  7: "Incline Bench Press",
  8: "Lat Pulldown",
  9: "Lateral Raises",
  10: "Leg Extensions",
  11: "Leg Raises",
  12: "Plank",
  13: "Pull Up",
  14: "Push Ups",
  15: "Romanian Deadlift",
  16: "Russian Twist",
  17: "Shoulder Press",
  18: "Squat",
  19: "T Bar Row",
  20: "Tricep Dips",
  21: "Tricep Pushdown"
};

// Palette (match web: indigo/purple)
export const COLORS = {
  primary: '#4f46e5',      // indigo-600
  secondary: '#9333ea',    // purple-600
  success: '#10b981',      // emerald-500
  warning: '#f59e0b',      // amber-500
  error: '#ef4444',        // red-500
  info: '#6366f1',         // indigo-500
  light: {
    background: '#ffffff',
    surface: '#ffffff',
    text: '#111827',        // gray-900
    textSecondary: '#6b7280', // gray-500
    border: '#e5e7eb',      // gray-200
  },
  dark: {
    background: '#111827',  // gray-900
    surface: '#1f2937',     // gray-800
    text: '#f9fafb',        // gray-50
    textSecondary: '#9ca3af', // gray-400
    border: '#374151',      // gray-700
  },
};

// Animation Configuration
export const ANIMATION_CONFIG = {
  duration: 300,
  easing: 'ease-in-out',
};

// Storage Keys
export const STORAGE_KEYS = {
  AUTH_TOKEN: 'authToken',
  USER_DATA: 'user',
  THEME: 'theme',
  WORKOUT_HISTORY: 'workoutHistory',
  SETTINGS: 'settings',
  LAST_WORKOUT: 'lastWorkout',
};

// Notification Configuration
export const NOTIFICATION_CONFIG = {
  workoutReminder: {
    title: 'Time to Work Out! 💪',
    body: 'Your body is ready for a great workout session.',
    sound: 'default',
    priority: 'high',
  },
  achievement: {
    title: 'Achievement Unlocked! 🏆',
    body: 'Great job! You\'ve reached a new milestone.',
    sound: 'default',
    priority: 'high',
  },
  formCorrection: {
    title: 'Form Correction Needed',
    body: 'Check your form to prevent injury and maximize results.',
    sound: 'default',
    priority: 'normal',
  },
}; 