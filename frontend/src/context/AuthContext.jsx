// src/context/AuthContext.jsx
import { createContext, useContext, useState, useEffect } from 'react';
import authService from '../services/authService';
import userService from '../services/userService';

const AuthContext = createContext();

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Load user on initial render if token exists
  useEffect(() => {
    const loadUser = async () => {
      try {
        if (authService.isAuthenticated()) {
          const userData = await authService.getCurrentUser();
          if (userData) {
            // Transform backend user data to frontend format
            const transformedData = userService.transformUserData(userData);
            setUser(transformedData);
          }
        }
      } catch (err) {
        console.error('Failed to load user', err);
        setError('Failed to authenticate user');
        authService.logout();
      } finally {
        setLoading(false);
      }
    };

    loadUser();
  }, []);

  const login = async (username, password) => {
    setError(null);
    try {
      await authService.login(username, password);
      const userData = await authService.getCurrentUser();
      const transformedData = userService.transformUserData(userData);
      setUser(transformedData);
      return true;
    } catch (err) {
      console.error('Login failed', err);
      setError(err.response?.data?.detail || 'Login failed. Please check your credentials.');
      return false;
    }
  };

  const register = async (username, email, password) => {
    setError(null);
    try {
      await authService.register(username, email, password);
      return true;
    } catch (err) {
      console.error('Registration failed', err);
      setError(err.response?.data || 'Registration failed. Please try again.');
      return false;
    }
  };

  const logout = () => {
    authService.logout();
    setUser(null);
  };

  const updateProfile = async (userData) => {
    setError(null);
    try {
      // Transform frontend data to backend format
      const backendData = userService.transformUserDataForBackend(userData);
      
      // Send update to API
      const updated = await authService.updateProfile(backendData);
      
      // Transform the response back to frontend format
      const transformedData = userService.transformUserData(updated);
      setUser(transformedData);
      
      return true;
    } catch (err) {
      console.error('Update profile failed', err);
      setError(err.response?.data || 'Failed to update profile. Please try again.');
      return false;
    }
  };

  const value = {
    user,
    loading,
    error,
    login,
    register,
    logout,
    updateProfile,
    isAuthenticated: authService.isAuthenticated
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}