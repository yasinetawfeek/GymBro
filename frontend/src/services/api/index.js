"""
Centralized API service exports for the GymBro frontend.
"""
import authService from './auth';
import billingService from './billing';
import analyticsService from './analytics';
import adminService from './admin';

// Export all services
export {
  authService,
  billingService,
  analyticsService,
  adminService,
};

// Default export with all services
export default {
  auth: authService,
  billing: billingService,
  analytics: analyticsService,
  admin: adminService,
};