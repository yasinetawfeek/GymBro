import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  UserCircle, Mail, MapPin, Calendar,
  Ruler, Weight, Heart, Medal, Target, 
  Clock, Dumbbell, ActivitySquare, TrendingUp,
  Award, AlertCircle, X, RefreshCw, BarChart3, ChevronLeft
} from 'lucide-react';

import NavBar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import ProfileOverview from '../components/ProfileOverview';
import FitnessStats from '../components/FitnessStats';
import UserManagement from '../components/UserManagement';
import UserDetailsPage from '../components/UserDetailsPage';
import UserDetailModal from '../components/UserDetailModal';
import ApprovalRequests from '../components/ApprovalRequests';
import BillingOverview from '../components/BillingOverview';
import AdminBillingActivity from '../components/AdminBillingActivity';
import ModelPerformance from '../components/ModelPerformance';
import MLModels from '../components/MLModels';

import { useAuth } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import userService from '../services/userService';

const AccountManagement = () => {
  const [activePage, setActivePage] = useState('overview');

  const handlePageChange = (page) => {
    setActivePage(page);
  };

  return (
    <div className="flex h-screen">
      <Sidebar activePage={activePage} onPageChange={handlePageChange} />
      <div className="flex-1 p-8">
        {activePage === 'overview' && <ProfileOverview />}
        {activePage === 'fitness' && <FitnessStats />}
        {activePage === 'user' && <UserManagement />}
        {activePage === 'details' && <UserDetailsPage />}
        {activePage === 'details' && <UserDetailModal />}
        {activePage === 'approval' && <ApprovalRequests />}
        {activePage === 'billing' && <BillingOverview />}
        {activePage === 'billing' && <AdminBillingActivity />}
        {activePage === 'models' && <MLModels />}
      </div>
    </div>
  );
};

export default AccountManagement; 