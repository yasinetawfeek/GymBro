# GymBro Frontend - React Application

The frontend is a modern React application built with Vite, providing a responsive user interface for the GymBro fitness platform. It features real-time pose analysis, workout tracking, and comprehensive user management.

## 🏗️ Architecture

The frontend follows a clean, component-based architecture with organized services and contexts:

```
frontend/
├── src/
│   ├── components/         # Reusable UI components
│   │   ├── common/         # Common components (LoadingSpinner, ErrorBoundary)
│   │   ├── layout/         # Layout components (Navbar, Sidebar)
│   │   └── features/       # Feature-specific components
│   ├── services/           # API and external service integrations
│   │   ├── api/           # Organized API services
│   │   │   ├── auth.js    # Authentication service
│   │   │   ├── billing.js # Billing and subscription service
│   │   │   ├── analytics.js # Analytics and usage tracking
│   │   │   └── admin.js   # Admin management service
│   │   └── websocket/     # WebSocket services
│   │       └── aiService.js # AI service WebSocket client
│   ├── contexts/          # React contexts for state management
│   │   └── AuthContext.jsx # Authentication context
│   ├── pages/            # Page components
│   │   ├── HomePage.jsx
│   │   ├── AuthPage.jsx
│   │   ├── Dashboard.jsx
│   │   ├── TrainingPage.jsx
│   │   └── WorkoutPage.jsx
│   ├── App.jsx           # Main application component
│   └── main.jsx          # Application entry point
├── Dockerfile.clean      # Production-ready Dockerfile
└── package.json         # Dependencies and scripts
```

## 🚀 Quick Start

### Development Setup

```bash
# Start development environment
make dev

# Or manually with Docker
docker-compose -f docker-compose.clean.yml --profile development up frontend-dev
```

### Production Setup

```bash
# Start production environment
make prod

# Or manually with Docker
docker-compose -f docker-compose.clean.yml up frontend
```

## 🛠️ Development

### Available Scripts

```bash
# Development
npm run dev          # Start development server
npm run build        # Build for production
npm run preview      # Preview production build
npm run lint         # Run ESLint

# Testing
npm test             # Run tests
npm run test:coverage # Run tests with coverage
```

### Environment Variables

```bash
# .env.local
VITE_API_URL=http://localhost:8000
VITE_AI_URL=http://localhost:8001
```

## 🎨 UI Components

### Common Components

#### LoadingSpinner
```jsx
import LoadingSpinner from './components/common/LoadingSpinner';

<LoadingSpinner 
  size="medium" 
  color="primary" 
  text="Loading..." 
/>
```

#### ErrorBoundary
```jsx
import ErrorBoundary from './components/common/ErrorBoundary';

<ErrorBoundary>
  <YourComponent />
</ErrorBoundary>
```

#### ProtectedRoute
```jsx
import ProtectedRoute from './components/common/ProtectedRoute';

<ProtectedRoute 
  requiredRoles={['admin']} 
  requireApproval={true}
>
  <AdminPanel />
</ProtectedRoute>
```

### Layout Components

#### Navbar
```jsx
import Navbar from './components/layout/Navbar';

function App() {
  return (
    <div>
      <Navbar />
      <main>
        {/* Your content */}
      </main>
    </div>
  );
}
```

## 🔌 API Services

### Authentication Service
```javascript
import { authService } from './services/api';

// Login
const result = await authService.login(username, password);

// Register
const result = await authService.register(userData);

// Get current user
const result = await authService.getCurrentUser();

// Update user
const result = await authService.updateUser(userData);

// Logout
authService.logout();

// Check authentication
const isAuth = authService.isAuthenticated();

// Get role info
const result = await authService.getRoleInfo();
```

### Billing Service
```javascript
import { billingService } from './services/api';

// Get subscriptions
const result = await billingService.getSubscriptions();

// Create subscription
const result = await billingService.createSubscription(subscriptionData);

// Cancel subscription
const result = await billingService.cancelSubscription(subscriptionId);

// Get invoices
const result = await billingService.getInvoices();

// Pay invoice
const result = await billingService.payInvoice(invoiceId);

// Get billing overview
const result = await billingService.getBillingOverview();
```

### Analytics Service
```javascript
import { analyticsService } from './services/api';

// Start session
const result = await analyticsService.startSession(sessionData);

// End session
const result = await analyticsService.endSession(sessionId, sessionData);

// Update metrics
const result = await analyticsService.updateMetrics(sessionId, metricsData);

// Get usage records
const result = await analyticsService.getUsageRecords();

// Record performance metrics (AI Engineers)
const result = await analyticsService.recordPerformanceMetrics(metricsData);

// Get performance analytics (AI Engineers)
const result = await analyticsService.getPerformanceAnalytics();
```

### Admin Service
```javascript
import { adminService } from './services/api';

// Get users (admin)
const result = await adminService.getUsers();

// Create user (admin)
const result = await adminService.createUser(userData);

// Update user (admin)
const result = await adminService.updateUser(userId, userData);

// Delete user (admin)
const result = await adminService.deleteUser(userId);

// Get approval requests (admin)
const result = await adminService.getApprovalRequests();

// Approve user (admin)
const result = await adminService.approveUser(userId);

// Reject user (admin)
const result = await adminService.rejectUser(userId);
```

## 🔌 WebSocket Services

### AI Service WebSocket
```javascript
import aiService from './services/websocket/aiService';

// Connect to AI service
aiService.connect(token, serverUrl);

// Send pose data
aiService.sendPoseData({
  landmarks: poseLandmarks,
  selected_workout: workoutType
});

// Listen for corrections
aiService.on('pose_corrections', (data) => {
  console.log('Pose corrections:', data);
});

// Listen for errors
aiService.on('error', (error) => {
  console.error('AI Service error:', error);
});

// Check connection status
const isConnected = aiService.getConnectionStatus();

// Disconnect
aiService.disconnect();
```

## 🎯 State Management

### Authentication Context
```javascript
import { useAuth } from './contexts/AuthContext';

function MyComponent() {
  const {
    user,
    isAuthenticated,
    isLoading,
    login,
    logout,
    hasRole,
    hasAnyRole
  } = useAuth();

  // Check user roles
  const isAdmin = hasRole('admin');
  const isAIEngineer = hasRole('ai_engineer');
  const isApproved = hasRole('approved');

  // Check multiple roles
  const canAccessAdmin = hasAnyRole(['admin', 'ai_engineer']);

  return (
    <div>
      {isAuthenticated ? (
        <p>Welcome, {user?.username}!</p>
      ) : (
        <p>Please log in</p>
      )}
    </div>
  );
}
```

## 🎨 Styling

### Tailwind CSS
The application uses Tailwind CSS for styling with a custom configuration:

```javascript
// tailwind.config.js
module.exports = {
  content: ['./src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#eff6ff',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
        },
      },
    },
  },
  plugins: [],
};
```

### Dark Mode Support
```javascript
// Toggle dark mode
const [isDarkMode, setIsDarkMode] = useState(false);

useEffect(() => {
  if (isDarkMode) {
    document.body.classList.add('dark');
  } else {
    document.body.classList.remove('dark');
  }
}, [isDarkMode]);
```

## 🧪 Testing

### Component Testing
```javascript
import { render, screen } from '@testing-library/react';
import { AuthProvider } from './contexts/AuthContext';
import MyComponent from './MyComponent';

test('renders component correctly', () => {
  render(
    <AuthProvider>
      <MyComponent />
    </AuthProvider>
  );
  
  expect(screen.getByText('Expected Text')).toBeInTheDocument();
});
```

### Service Testing
```javascript
import { authService } from './services/api';

// Mock API calls
jest.mock('./services/api/client');

test('login service', async () => {
  const mockResponse = { success: true, data: { user: 'test' } };
  apiClient.post.mockResolvedValue(mockResponse);
  
  const result = await authService.login('user', 'pass');
  expect(result.success).toBe(true);
});
```

## 🚀 Deployment

### Development Build
```bash
npm run build
npm run preview
```

### Production Build
```bash
# Build for production
npm run build

# Serve with nginx
docker build -f Dockerfile.clean -t gymbro-frontend .
docker run -p 80:80 gymbro-frontend
```

### Environment Configuration
The production build supports runtime environment variable substitution:

```javascript
// Runtime environment configuration
window._env_ = {
  VITE_API_URL: 'http://backend:8000',
  VITE_AI_URL: 'http://ai:8001'
};
```

## 🔒 Security

### Authentication
- JWT token management
- Automatic token refresh
- Secure token storage
- Role-based access control

### Input Validation
- Form validation with error handling
- XSS protection
- CSRF protection via SameSite cookies

### Content Security Policy
```html
<meta http-equiv="Content-Security-Policy" 
      content="default-src 'self'; 
               script-src 'self' 'unsafe-inline'; 
               style-src 'self' 'unsafe-inline';">
```

## 📱 Responsive Design

### Breakpoints
```css
/* Mobile First Approach */
.sm:min-width: 640px
.md:min-width: 768px
.lg:min-width: 1024px
.xl:min-width: 1280px
```

### Mobile Optimization
- Touch-friendly interfaces
- Optimized for mobile devices
- Progressive Web App features
- Offline capability

## 🎯 Performance

### Code Splitting
```javascript
// Lazy load components
const LazyComponent = React.lazy(() => import('./LazyComponent'));

function App() {
  return (
    <Suspense fallback={<LoadingSpinner />}>
      <LazyComponent />
    </Suspense>
  );
}
```

### Bundle Optimization
- Tree shaking
- Dead code elimination
- Asset optimization
- Compression

## 🤝 Contributing

### Development Workflow
1. Create feature branch
2. Make changes
3. Write tests
4. Run linting
5. Submit pull request

### Code Standards
- Follow ESLint configuration
- Use Prettier for formatting
- Write meaningful commit messages
- Add JSDoc comments

## 🆘 Troubleshooting

### Common Issues

**Build Errors**
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
npm run build
```

**API Connection Issues**
```bash
# Check environment variables
echo $VITE_API_URL
echo $VITE_AI_URL

# Test API connectivity
curl http://localhost:8000/health/
```

**WebSocket Connection Issues**
```javascript
// Check WebSocket connection
aiService.on('connect', () => console.log('Connected!'));
aiService.on('disconnect', () => console.log('Disconnected!'));
aiService.on('error', (error) => console.error('Error:', error));
```

### Debug Mode
```bash
# Enable debug logging
export NODE_ENV=development
npm run dev
```

---

**Frontend** - The face of the GymBro platform 🎨✨