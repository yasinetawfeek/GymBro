// src/components/ProtectedRoute.jsx
import { Navigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function ProtectedRoute({ children, requireAdmin = false, requireEngineer = false}) {
  const user = useAuth();

  //Check if user is authenticated
  if (user.user != null) {
    return <Navigate to="/auth" />;
  }

  // // Check if route requires admin privileges
  // if (requireAdmin && user?.rolename !== 'Admin') {
  //   return <Navigate to="/dashboard" replace />;
  // }

  //   // Check if route requires admin privileges
  // if (requireEngineer && user?.rolename !== 'AIEngineer') {
  //   return <Navigate to="/dashboard" replace />;
  // }

  return children;
}