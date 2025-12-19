import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_URL } from '../config';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: API_URL,
  timeout: 10000,
});

// Request interceptor to add auth token
api.interceptors.request.use(
  async (config) => {
    try {
      const token = await AsyncStorage.getItem('authToken');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
    } catch (error) {
      console.error('Error getting auth token:', error);
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor to handle auth errors
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.response?.status === 401) {
      // Token expired or invalid
      try {
        await AsyncStorage.removeItem('authToken');
        await AsyncStorage.removeItem('user');
      } catch (storageError) {
        console.error('Error clearing auth data:', storageError);
      }
    }
    return Promise.reject(error);
  }
);

export const authService = {
  // Login user
  login: async (credentials) => {
    return api.post('/auth/login/', credentials);
  },

  // Register user
  register: async (userData) => {
    return api.post('/auth/register/', userData);
  },

  // Verify token
  verifyToken: async (token) => {
    return api.post('/auth/verify/', { token });
  },

  // Refresh token
  refreshToken: async (refreshToken) => {
    return api.post('/auth/refresh/', { refresh: refreshToken });
  },

  // Logout user
  logout: async () => {
    return api.post('/auth/logout/');
  },

  // Get user profile
  getProfile: async () => {
    return api.get('/auth/profile/');
  },

  // Update user profile
  updateProfile: async (userData) => {
    return api.put('/auth/profile/', userData);
  },

  // Change password
  changePassword: async (passwordData) => {
    return api.post('/auth/change-password/', passwordData);
  },

  // Request password reset
  requestPasswordReset: async (email) => {
    return api.post('/auth/password-reset/', { email });
  },

  // Reset password with token
  resetPassword: async (token, newPassword) => {
    return api.post('/auth/password-reset/confirm/', {
      token,
      new_password: newPassword,
    });
  },
}; 