"""
Authentication API service for the GymBro frontend.
"""
import apiClient from './client';

class AuthService {
  /**
   * Login with username and password
   * @param {string} username - User's username
   * @param {string} password - User's password
   * @returns {Promise<Object>} Authentication response with tokens
   */
  async login(username, password) {
    try {
      const response = await apiClient.post('/auth/jwt/create/', {
        username,
        password,
      });
      
      const { access, refresh } = response.data;
      
      // Store tokens in localStorage
      localStorage.setItem('access_token', access);
      localStorage.setItem('refresh_token', refresh);
      
      return { success: true, data: response.data };
    } catch (error) {
      console.error('Login error:', error);
      return { 
        success: false, 
        error: error.response?.data?.detail || 'Login failed' 
      };
    }
  }

  /**
   * Register a new user
   * @param {Object} userData - User registration data
   * @returns {Promise<Object>} Registration response
   */
  async register(userData) {
    try {
      const response = await apiClient.post('/auth/users/', userData);
      return { success: true, data: response.data };
    } catch (error) {
      console.error('Registration error:', error);
      return { 
        success: false, 
        error: error.response?.data || 'Registration failed' 
      };
    }
  }

  /**
   * Logout the current user
   */
  logout() {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    window.location.href = '/auth';
  }

  /**
   * Get current user information
   * @returns {Promise<Object>} Current user data
   */
  async getCurrentUser() {
    try {
      const response = await apiClient.get('/api/my_account/');
      return { success: true, data: response.data };
    } catch (error) {
      console.error('Get current user error:', error);
      return { 
        success: false, 
        error: error.response?.data || 'Failed to get user data' 
      };
    }
  }

  /**
   * Update current user information
   * @param {Object} userData - Updated user data
   * @returns {Promise<Object>} Update response
   */
  async updateUser(userData) {
    try {
      const response = await apiClient.patch('/api/my_account/', userData);
      return { success: true, data: response.data };
    } catch (error) {
      console.error('Update user error:', error);
      return { 
        success: false, 
        error: error.response?.data || 'Failed to update user' 
      };
    }
  }

  /**
   * Check if user is authenticated
   * @returns {boolean} Authentication status
   */
  isAuthenticated() {
    return !!localStorage.getItem('access_token');
  }

  /**
   * Get user role information
   * @returns {Promise<Object>} Role information
   */
  async getRoleInfo() {
    try {
      const response = await apiClient.get('/api/role_info/');
      return { success: true, data: response.data };
    } catch (error) {
      console.error('Get role info error:', error);
      return { 
        success: false, 
        error: error.response?.data || 'Failed to get role info' 
      };
    }
  }
}

export default new AuthService();