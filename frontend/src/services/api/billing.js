"""
Billing and subscription API service for the GymBro frontend.
"""
import apiClient from './client';

class BillingService {
  /**
   * Get user's subscriptions
   * @returns {Promise<Object>} Subscriptions data
   */
  async getSubscriptions() {
    try {
      const response = await apiClient.get('/api/subscriptions/');
      return { success: true, data: response.data };
    } catch (error) {
      console.error('Get subscriptions error:', error);
      return { 
        success: false, 
        error: error.response?.data || 'Failed to get subscriptions' 
      };
    }
  }

  /**
   * Create a new subscription
   * @param {Object} subscriptionData - Subscription data
   * @returns {Promise<Object>} Create subscription response
   */
  async createSubscription(subscriptionData) {
    try {
      const response = await apiClient.post('/api/subscriptions/', subscriptionData);
      return { success: true, data: response.data };
    } catch (error) {
      console.error('Create subscription error:', error);
      return { 
        success: false, 
        error: error.response?.data || 'Failed to create subscription' 
      };
    }
  }

  /**
   * Cancel a subscription
   * @param {number} subscriptionId - Subscription ID
   * @returns {Promise<Object>} Cancel subscription response
   */
  async cancelSubscription(subscriptionId) {
    try {
      const response = await apiClient.patch(`/api/subscriptions/${subscriptionId}/cancel/`);
      return { success: true, data: response.data };
    } catch (error) {
      console.error('Cancel subscription error:', error);
      return { 
        success: false, 
        error: error.response?.data || 'Failed to cancel subscription' 
      };
    }
  }

  /**
   * Get user's invoices
   * @returns {Promise<Object>} Invoices data
   */
  async getInvoices() {
    try {
      const response = await apiClient.get('/api/invoices/');
      return { success: true, data: response.data };
    } catch (error) {
      console.error('Get invoices error:', error);
      return { 
        success: false, 
        error: error.response?.data || 'Failed to get invoices' 
      };
    }
  }

  /**
   * Pay an invoice
   * @param {number} invoiceId - Invoice ID
   * @returns {Promise<Object>} Pay invoice response
   */
  async payInvoice(invoiceId) {
    try {
      const response = await apiClient.patch(`/api/invoices/${invoiceId}/pay/`);
      return { success: true, data: response.data };
    } catch (error) {
      console.error('Pay invoice error:', error);
      return { 
        success: false, 
        error: error.response?.data || 'Failed to pay invoice' 
      };
    }
  }

  /**
   * Get billing overview with statistics
   * @returns {Promise<Object>} Billing overview data
   */
  async getBillingOverview() {
    try {
      const response = await apiClient.get('/api/billing/overview/');
      return { success: true, data: response.data };
    } catch (error) {
      console.error('Get billing overview error:', error);
      return { 
        success: false, 
        error: error.response?.data || 'Failed to get billing overview' 
      };
    }
  }

  /**
   * Get billing records
   * @returns {Promise<Object>} Billing records data
   */
  async getBillingRecords() {
    try {
      const response = await apiClient.get('/api/billing/');
      return { success: true, data: response.data };
    } catch (error) {
      console.error('Get billing records error:', error);
      return { 
        success: false, 
        error: error.response?.data || 'Failed to get billing records' 
      };
    }
  }
}

export default new BillingService();