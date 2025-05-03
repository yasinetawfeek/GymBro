# Implementation Plan for Test Cases

## Feature 1: User Registration & Authentication
**Backend:**
- Add email validation for different domains (@ufcfur_15_3.com for company employees)
- Create API endpoints for registration with role assignment
- Implement JWT authentication with proper token handling

**Frontend:**
- Build registration form with role selection (hidden for public users)
- Create login page with proper error handling
- Implement persistent authentication with token storage

**Checks:**
- 1.1: Register Mr Edmond Hobbs as customer ✓
- 2.1: Register Ms Tensa Flow as AI engineer ✓
- 4.4: Unregistered user requires signup ✓

## Feature 2: Role-Based Access Control
**Backend:**
- Implement middleware to check user roles and permissions
- Create approval workflow for AI engineers
- Add is_approved flag to User model
- Secure routes based on role and approval status

**Frontend:**
- Enhance ProtectedRoute component with role checks
- Modify AccountManagement.jsx to show different dashboards based on user role
- Update Sidebar.jsx to display role-specific navigation options
- Add UI indicators for approval status in AccountManagement.jsx
- Implement redirect for unauthorized access attempts

**Checks:**
- 1.4: Edmond fails to access admin dashboard ✓
- 2.3: Tensa can't access AI dashboard until approved ✓
- 3.4: Dr First fails to access admin dashboard ✓
- 4.4: Unregistered users redirected to signup ✓

## Feature 3: Admin Dashboard
**Backend:**
- ✅ Create API for pending approvals listing
- ✅ Implement endpoints for approval/rejection actions
- ✅ Add user management APIs (list, update, delete)
- ✅ Create billing report endpoints with date filtering (TODO)

**Frontend:**
- ✅ Build approval management interface (implemented in ApprovalRequests.jsx)
- ✅ Create user listing with role filters and actions (implemented in UserManagement.jsx)
- ✅ Implement billing overview with date range selection (TODO)
- ✅ Add user access management controls (implemented in AccountManagement.jsx)

**Implementation Notes:**
- The Admin Dashboard is integrated into AccountManagement.jsx
- Navigation is handled through Sidebar.jsx which already has role-specific options
- ApprovalRequests.jsx handles the approval management interface
- UserManagement.jsx handles the user listing and management
- We need to add a billing overview component

**Checks:**
- 2.2: Admin approves Tensa as AI engineer ✓
- 4.1: Admin checks and approves/rejects pending requests ✓
- 4.2: Admin views billable activity by time period ✓ (TODO)
- 4.3: Admin can remove AI engineer access ✓

## Feature 4: Machine Learning Service
**Backend:**
- ~~Implement ML model integration~~ ✓ (Already implemented with workout_classifer_model.py)
- ~~Create prediction API endpoint~~ ✓ (Already implemented in pose_correction_server.py)
- Store model performance metrics
  - Create Django models for tracking confidence scores, response times, and accuracy
  - Implement periodic logging of model performance metrics
  - Add API endpoints to retrieve performance data
- Add usage tracking per user request
  - Create usage tracking models to record ML service requests
  - Track user ID, timestamp, model type, and request metadata
  - Implement rate limiting and quota system for AI feature access

**Frontend:**
- ~~Build file upload interface for data~~ ✓ (Already implemented in WorkoutPage.jsx)
- Create admin dashboard for ML metrics
  - Display usage statistics (requests per day/hour)
  - Show model performance trends over time
  - Provide user activity visualizations 
- Implement usage tracking display
  - Show personal usage statistics to users
  - Display quota information for AI Engineer and Admin roles

**Checks:**
- ~~1.2: Edmond uploads data and receives prediction~~ ✓ (Already implemented)
- 3.1: Dr First logs in to AI dashboard and views usage statistics
- 3.2: Dr First views system performance metrics and model accuracy
- 3.3: Admin can see user feedback on model predictions

## Feature 5: Model Management
**Backend:**
- Create API for model version control
- Implement model update endpoints
- Store and track model performance
- Add permission checks for model management

**Frontend:**
- Build model performance visualization
- Create model update interface for AI engineers
- Add version history display
- Implement model comparison tools

**Checks:**
- 2.4: Tensa views model performance ✓
- 3.2: Dr First views system performance ✓
- 3.3: Dr First updates aspects of ML model ✓

## Feature 6: Subscription & Billing
**Backend:**
- Create Subscription model with different plans
- Implement Invoice generation system
- Add usage tracking and quotas
- Create payment processing integration (Stripe)

**Frontend:**
- Build subscription selection interface
- Create invoice display and download
- Add payment method management
- Implement usage statistics visualization

**Checks:**
- 1.3: Edmond gets invoice to pay ✓
- 4.2: Admin views billable activity ✓