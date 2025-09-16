"""
WebSocket service for AI pose correction and workout classification.
"""
import { io } from 'socket.io-client';

class AIService {
  constructor() {
    this.socket = null;
    this.isConnected = false;
    this.callbacks = {
      onConnect: [],
      onDisconnect: [],
      onPoseCorrections: [],
      onError: [],
    };
  }

  /**
   * Connect to the AI WebSocket server
   * @param {string} token - Authentication token
   * @param {string} serverUrl - AI server URL
   */
  connect(token = null, serverUrl = import.meta.env.VITE_AI_URL || 'http://localhost:8001') {
    if (this.socket) {
      this.disconnect();
    }

    const url = new URL(serverUrl);
    if (token) {
      url.searchParams.set('token', token);
    }

    this.socket = io(url.toString(), {
      transports: ['websocket'],
      timeout: 10000,
    });

    this.setupEventListeners();
  }

  /**
   * Setup WebSocket event listeners
   */
  setupEventListeners() {
    this.socket.on('connect', () => {
      console.log('Connected to AI service');
      this.isConnected = true;
      this.callbacks.onConnect.forEach(callback => callback());
    });

    this.socket.on('disconnect', () => {
      console.log('Disconnected from AI service');
      this.isConnected = false;
      this.callbacks.onDisconnect.forEach(callback => callback());
    });

    this.socket.on('connected', (data) => {
      console.log('AI service connection confirmed:', data);
    });

    this.socket.on('pose_corrections', (data) => {
      this.callbacks.onPoseCorrections.forEach(callback => callback(data));
    });

    this.socket.on('error', (error) => {
      console.error('AI service error:', error);
      this.callbacks.onError.forEach(callback => callback(error));
    });
  }

  /**
   * Send pose data for analysis
   * @param {Object} poseData - Pose landmarks and metadata
   */
  sendPoseData(poseData) {
    if (this.socket && this.isConnected) {
      this.socket.emit('pose_data', {
        ...poseData,
        timestamp: Date.now(),
      });
    } else {
      console.warn('AI service not connected, cannot send pose data');
    }
  }

  /**
   * Disconnect from the AI service
   */
  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
      this.isConnected = false;
    }
  }

  /**
   * Add event callback
   * @param {string} event - Event name
   * @param {Function} callback - Callback function
   */
  on(event, callback) {
    if (this.callbacks[event]) {
      this.callbacks[event].push(callback);
    } else {
      console.warn(`Unknown event: ${event}`);
    }
  }

  /**
   * Remove event callback
   * @param {string} event - Event name
   * @param {Function} callback - Callback function to remove
   */
  off(event, callback) {
    if (this.callbacks[event]) {
      const index = this.callbacks[event].indexOf(callback);
      if (index > -1) {
        this.callbacks[event].splice(index, 1);
      }
    }
  }

  /**
   * Get connection status
   * @returns {boolean} Connection status
   */
  getConnectionStatus() {
    return this.isConnected;
  }
}

export default new AIService();