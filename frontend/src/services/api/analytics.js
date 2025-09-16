"""
Analytics and usage tracking API service for the GymBro frontend.
"""
import apiClient from './client';

class AnalyticsService {
  /**
   * Start a new usage tracking session
   * @param {Object} sessionData - Session data
   * @returns {Promise<Object>} Start session response
   */
  async startSession(sessionData) {
    try {
      const response = await apiClient.post('/api/usage/start_session/', sessionData);
      return { success: true, data: response.data };
    } catch (error) {
      console.error('Start session error:', error);
      return { 
        success: false, 
        error: error.response?.data || 'Failed to start session' 
      };
    }
  }

  /**
   * End a usage tracking session
   * @param {number} sessionId - Session ID
   * @param {Object} sessionData - Session end data
   * @returns {Promise<Object>} End session response
   */
  async endSession(sessionId, sessionData) {
    try {
      const response = await apiClient.patch(`/api/usage/${sessionId}/end_session/`, sessionData);
      return { success: true, data: response.data };
    } catch (error) {
      console.error('End session error:', error);
      return { 
        success: false, 
        error: error.response?.data || 'Failed to end session' 
      };
    }
  }

  /**
   * Update session metrics
   * @param {number} sessionId - Session ID
   * @param {Object} metricsData - Metrics data
   * @returns {Promise<Object>} Update metrics response
   */
  async updateMetrics(sessionId, metricsData) {
    try {
      const response = await apiClient.patch(`/api/usage/${sessionId}/update_metrics/`, metricsData);
      return { success: true, data: response.data };
    } catch (error) {
      console.error('Update metrics error:', error);
      return { 
        success: false, 
        error: error.response?.data || 'Failed to update metrics' 
      };
    }
  }

  /**
   * Get usage records
   * @returns {Promise<Object>} Usage records data
   */
  async getUsageRecords() {
    try {
      const response = await apiClient.get('/api/usage/');
      return { success: true, data: response.data };
    } catch (error) {
      console.error('Get usage records error:', error);
      return { 
        success: false, 
        error: error.response?.data || 'Failed to get usage records' 
      };
    }
  }

  /**
   * Record model performance metrics (AI Engineers only)
   * @param {Object} metricsData - Performance metrics data
   * @returns {Promise<Object>} Record metrics response
   */
  async recordPerformanceMetrics(metricsData) {
    try {
      const response = await apiClient.post('/api/model-performance/record_metrics/', metricsData);
      return { success: true, data: response.data };
    } catch (error) {
      console.error('Record performance metrics error:', error);
      return { 
        success: false, 
        error: error.response?.data || 'Failed to record performance metrics' 
      };
    }
  }

  /**
   * Get performance analytics (AI Engineers only)
   * @returns {Promise<Object>} Performance analytics data
   */
  async getPerformanceAnalytics() {
    try {
      const response = await apiClient.get('/api/model-performance/analytics/');
      return { success: true, data: response.data };
    } catch (error) {
      console.error('Get performance analytics error:', error);
      return { 
        success: false, 
        error: error.response?.data || 'Failed to get performance analytics' 
      };
    }
  }

  /**
   * Update last viewed exercise
   * @param {Object} exerciseData - Exercise data
   * @returns {Promise<Object>} Update exercise response
   */
  async updateLastViewedExercise(exerciseData) {
    try {
      const response = await apiClient.post('/api/last-viewed-exercise/', exerciseData);
      return { success: true, data: response.data };
    } catch (error) {
      console.error('Update last viewed exercise error:', error);
      return { 
        success: false, 
        error: error.response?.data || 'Failed to update last viewed exercise' 
      };
    }
  }

  /**
   * Get last viewed exercise
   * @returns {Promise<Object>} Last viewed exercise data
   */
  async getLastViewedExercise() {
    try {
      const response = await apiClient.get('/api/last-viewed-exercise/');
      return { success: true, data: response.data };
    } catch (error) {
      console.error('Get last viewed exercise error:', error);
      return { 
        success: false, 
        error: error.response?.data || 'Failed to get last viewed exercise' 
      };
    }
  }
}

export default new AnalyticsService();