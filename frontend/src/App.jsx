import { Routes, Route, Navigate } from 'react-router-dom';
import AuthPage from './pages/AuthPage';
import HomePage from './pages/HomePage';
import TrainingPage from './pages/TrainingPage';
import WorkoutPage from './pages/WorkoutPage';
import Dashboard from './pages/Dashboard';
import ProtectedRoute from './components/ProtectedRoute';
import AccountManagement from './pages/AccountManagement';
import Meeting from './pages/Meeting';
import ErrorBoundary from './components/ErrorBoundary';
import PredictionPage from './pages/PredictionPage';
import ModelManagement from './pages/ModelManagement';
import InvoiceDetail from './components/InvoiceDetail';

import './App.css';

export default function App() {
  return (
      <Routes>
        {/* Public routes */}
        <Route path="/" element={<HomePage/>} />
        <Route path="/auth" element={<AuthPage />} />
        
        {/* Protected routes for all authenticated users */}
        <Route path="/settings" element={<ProtectedRoute><AccountManagement /></ProtectedRoute>} />
        <Route path="/dashboard" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
        <Route path="/invoices/:invoiceId" element={<ProtectedRoute><InvoiceDetail /></ProtectedRoute>} />
        
        {/* Customer and authorized user routes */}
        <Route path="/training" element={<ProtectedRoute><TrainingPage /></ProtectedRoute>} />
        <Route path="/workout" element={<ProtectedRoute><WorkoutPage /></ProtectedRoute>} />
        <Route path="/predict" element={<ProtectedRoute><PredictionPage /></ProtectedRoute>} />
        
        {/* Admin-only routes */}
        <Route path="/admin" element={<ProtectedRoute requireAdmin={true}><AccountManagement /></ProtectedRoute>} />
        <Route path="/admin/users" element={<ProtectedRoute requireAdmin={true}><AccountManagement /></ProtectedRoute>} />
        <Route path="/admin/approvals" element={<ProtectedRoute requireAdmin={true}><AccountManagement /></ProtectedRoute>} />
        <Route path="/admin/billing" element={<ProtectedRoute requireAdmin={true}><AccountManagement /></ProtectedRoute>} />
        
        {/* AI Engineer routes */}
        <Route path="/engineer" element={<ProtectedRoute requireEngineer={true}><AccountManagement /></ProtectedRoute>} />
        <Route path="/engineer/models" element={<ProtectedRoute requireEngineer={true}><ModelManagement /></ProtectedRoute>} />
        
        {/* Video streaming route */}
        <Route path="/live" element={<ErrorBoundary><Meeting /></ErrorBoundary>} />
        
        {/* Fallback route */}
        <Route path="*" element={<Navigate to="/" />} />
      </Routes>
  );
}
