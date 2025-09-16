"""
Admin API service for the GymBro frontend.
"""
import apiClient from './client';

class AdminService {
  /**
   * Get all users (admin only)
   * @returns {Promise<Object>} Users data
   */
  async getUsers() {
    try {
      const response = await apiClient.get('/api/manage_accounts/');
      return { success: true, data: response.data };
    } catch (error) {
      console.error('Get users error:', error);
      return { 
        success: false, 
        error: error.response?.data || 'Failed to get users' 
      };
    }
  }

  /**
   * Create a new user (admin only)
   * @param {Object} userData - User data
   * @returns {Promise<Object>} Create user response
   */
  async createUser(userData) {
    try {
      const response = await apiClient.post('/api/manage_accounts/', userData);
      return { success: true, data: response.data };
    } catch (error) {
      console.error('Create user error:', error);
      return { 
        success: false, 
        error: error.response?.data || 'Failed to create user' 
      };
    }
  }

  /**
   * Update a user (admin only)
   * @param {number} userId - User ID
   * @param {Object} userData - Updated user data
   * @returns {Promise<Object>} Update user response
   */
  async updateUser(userId, userData) {
    try {
      const response = await apiClient.patch(`/api/manage_accounts/${userId}/`, userData);
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
   * Delete a user (admin only)
   * @param {number} userId - User ID
   * @returns {Promise<Object>} Delete user response
   */
  async deleteUser(userId) {
    try {
      const response = await apiClient.delete(`/api/manage_accounts/${userId}/`);
      return { success: true, data: response.data };
    } catch (error) {
      console.error('Delete user error:', error);
      return { 
        success: false, 
        error: error.response?.data || 'Failed to delete user' 
      };
    }
  }

  /**
   * Get user approval requests (admin only)
   * @returns {Promise<Object>} Approval requests data
   */
  async getApprovalRequests() {
    try {
      const response = await apiClient.get('/api/approvals/');
      return { success: true, data: response.data };
    } catch (error) {
      console.error('Get approval requests error:', error);
      return { 
        success: false, 
        error: error.response?.data || 'Failed to get approval requests' 
      };
    }
  }

  /**
   * Approve a user (admin only)
   * @param {number} userId - User ID
   * @returns {Promise<Object>} Approve user response
   */
  async approveUser(userId) {
    try {
      const response = await apiClient.patch(`/api/approvals/${userId}/approve/`);
      return { success: true, data: response.data };
    } catch (error) {
      console.error('Approve user error:', error);
      return { 
        success: false, 
        error: error.response?.data || 'Failed to approve user' 
      };
    }
  }

  /**
   * Reject a user (admin only)
   * @param {number} userId - User ID
   * @returns {Promise<Object>} Reject user response
   */
  async rejectUser(userId) {
    try {
      const response = await apiClient.patch(`/api/approvals/${userId}/reject/`);
      return { success: true, data: response.data };
    } catch (error) {
      console.error('Reject user error:', error);
      return { 
        success: false, 
        error: error.response?.data || 'Failed to reject user' 
      };
    }
  }

  /**
   * Get ML models (AI Engineers only)
   * @returns {Promise<Object>} ML models data
   */
  async getMLModels() {
    try {
      const response = await apiClient.get('/api/ml-models/');
      return { success: true, data: response.data };
    } catch (error) {
      console.error('Get ML models error:', error);
      return { 
        success: false, 
        error: error.response?.data || 'Failed to get ML models' 
      };
    }
  }

  /**
   * Create a new ML model (AI Engineers only)
   * @param {Object} modelData - Model data
   * @returns {Promise<Object>} Create model response
   */
  async createMLModel(modelData) {
    try {
      const response = await apiClient.post('/api/ml-models/', modelData);
      return { success: true, data: response.data };
    } catch (error) {
      console.error('Create ML model error:', error);
      return { 
        success: false, 
        error: error.response?.data || 'Failed to create ML model' 
      };
    }
  }

  /**
   * Deploy an ML model (AI Engineers only)
   * @param {number} modelId - Model ID
   * @returns {Promise<Object>} Deploy model response
   */
  async deployMLModel(modelId) {
    try {
      const response = await apiClient.patch(`/api/ml-models/${modelId}/deploy/`);
      return { success: true, data: response.data };
    } catch (error) {
      console.error('Deploy ML model error:', error);
      return { 
        success: false, 
        error: error.response?.data || 'Failed to deploy ML model' 
      };
    }
  }

  /**
   * Undeploy an ML model (AI Engineers only)
   * @param {number} modelId - Model ID
   * @returns {Promise<Object>} Undeploy model response
   */
  async undeployMLModel(modelId) {
    try {
      const response = await apiClient.patch(`/api/ml-models/${modelId}/undeploy/`);
      return { success: true, data: response.data };
    } catch (error) {
      console.error('Undeploy ML model error:', error);
      return { 
        success: false, 
        error: error.response?.data || 'Failed to undeploy ML model' 
      };
    }
  }

  /**
   * Get deployed models (AI Engineers only)
   * @returns {Promise<Object>} Deployed models data
   */
  async getDeployedModels() {
    try {
      const response = await apiClient.get('/api/ml-models/deployed/');
      return { success: true, data: response.data };
    } catch (error) {
      console.error('Get deployed models error:', error);
      return { 
        success: false, 
        error: error.response?.data || 'Failed to get deployed models' 
      };
    }
  }
}

export default new AdminService();