// src/components/ProtectedRoute.jsx
import { useState, useEffect } from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function ProtectedRoute({ children, requireAdmin = false, requireEngineer = false}) {
  // Match Navbar.jsx's implementation exactly - note that Navbar doesn't destructure
  const auth = useAuth();
  const [isApproved, setIsApproved] = useState(null);
  const [checkingApproval, setCheckingApproval] = useState(false);

  // Check approval status for roles that require it
  useEffect(() => {
    const checkApproval = async () => {
      if (auth.user) {
        setCheckingApproval(true);
        const approved = await auth.checkApprovalStatus();
        setIsApproved(approved);
        setCheckingApproval(false);
      }
    };

    if (auth.user && (requireAdmin || requireEngineer)) {
      checkApproval();
    } else if (auth.user) {
      // Default approved for regular users
      setIsApproved(true);
    }
  }, [auth.user, requireAdmin, requireEngineer]);

  console.log('ProtectedRoute: Auth State', {
    isLoading: auth.loading,
    checkingApproval,
    hasUser: !!auth.user,
    isApproved,
    requireAdmin,
    requireEngineer
  });

  // Show loading state while authentication or approval is being determined
  if (auth.loading || checkingApproval) {
    console.log("ProtectedRoute: Authentication or approval check is still loading");
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-900">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-purple-500"></div>
        <p className="text-white ml-3">Verifying access...</p>
      </div>
    );
  }

  // Check for authentication exactly like Navbar does
  if (!auth.user) {
    console.log("ProtectedRoute: Not authenticated, redirecting to /auth");
    return <Navigate to="/auth" />;
  }

  // Check if user is approved when required
  if (isApproved === false) {
    console.log("ProtectedRoute: User not approved, redirecting to pending page");
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-gray-900 text-white p-6 text-center">
        <h1 className="text-3xl font-bold mb-4">Approval Required</h1>
        <p className="text-xl mb-6">Your account is waiting for administrator approval.</p>
        <p className="mb-4">You'll receive an email when your account is approved.</p>
        <button 
          onClick={() => auth.logout()}
          className="px-4 py-2 bg-purple-600 rounded hover:bg-purple-700 transition-colors"
        >
          Return to Login
        </button>
      </div>
    );
  }

  // Get user role to check permissions
  const userRole = auth.user?.groups?.[0]?.name || 
                   auth.user?.rolename || 
                   (auth.user?.is_admin ? 'Admin' : 'Customer');

  // Check role-specific permissions
  if (requireAdmin && userRole !== 'Admin') {
    console.log("ProtectedRoute: Admin access required, redirecting to dashboard");
    return <Navigate to="/dashboard" replace />;
  }

  if (requireEngineer && userRole !== 'AI Engineer') {
    console.log("ProtectedRoute: AI Engineer access required, redirecting to dashboard");
    return <Navigate to="/dashboard" replace />;
  }

  console.log("ProtectedRoute: User authenticated and approved, rendering children");
  return children;
}