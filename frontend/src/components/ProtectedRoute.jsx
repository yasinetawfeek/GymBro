// src/components/ProtectedRoute.jsx
import { Navigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function ProtectedRoute({ children, requireAdmin = false }) {
  const { user, loading, isAuthenticated } = useAuth();

  // Show loading indicator while checking authentication
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-900">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-purple-500"></div>
      </div>
    );
  }

  // Check if user is authenticated
  if (!isAuthenticated()) {
    return <Navigate to="/auth" replace />;
  }

  // Check if route requires admin privileges
  if (requireAdmin && user?.rolename !== 'Admin') {
    return <Navigate to="/dashboard" replace />;
  }

  return children;
}