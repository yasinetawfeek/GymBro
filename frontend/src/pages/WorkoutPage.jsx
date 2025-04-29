import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import * as poseDetection from '@mediapipe/pose';
import { POSE_CONNECTIONS } from '@mediapipe/pose';
import { Camera } from '@mediapipe/camera_utils';
import io from 'socket.io-client';
// Import React Body Highlighter
import Model from 'react-body-highlighter';
import { MuscleType, ModelType } from 'react-body-highlighter';
import { useNavigate } from 'react-router-dom';

// Lucide icons for a more consistent look
import { 
  FullscreenIcon, 
  MinimizeIcon, 
  XCircle, 
  Check, 
  AlertTriangle, 
  Info, 
  Eye, 
  EyeOff, 
  RefreshCw, 
  ChevronDown, 
  ChevronUp, 
  Zap
} from 'lucide-react';

// Import NavBar
import NavBar from '../components/Navbar';

// Add workout mapping (matching the backend)
const workoutMap = { 
  0: "Barbell Bicep Curl", 
  1: "Bench Press", 
  2: "Chest Fly Machine", 
  3: "Deadlift",
  4: "Decline Bench Press", 
  5: "Hammer Curl", 
  6: "Hip Thrust", 
  7: "Incline Bench Press", 
  8: "Lat Pulldown", 
  9: "Lateral Raises", 
  10: "Leg Extensions", 
  11: "Leg Raises",
  12: "Plank", 
  13: "Pull Up", 
  14: "Push Ups", 
  15: "Romanian Deadlift", 
  16: "Russian Twist", 
  17: "Shoulder Press", 
  18: "Squat", 
  19: "T Bar Row", 
  20: "Tricep Dips", 
  21: "Tricep Pushdown"
};

// Define muscle group mapping
const muscleGroupMap = {
  1: "Shoulders",
  2: "Chest",
  3: "Biceps",
  4: "Core",
  5: "Triceps",
  6: "Legs",
  7: "Back"
};

// --- Utility functions ---
const getJointName = (index) => {
  switch(index) {
    case 11: return "Left Shoulder"; case 12: return "Right Shoulder"; case 13: return "Left Elbow";
    case 14: return "Right Elbow"; case 15: return "Left Wrist"; case 16: return "Right Wrist";
    case 23: return "Left Hip"; case 24: return "Right Hip"; case 25: return "Left Knee";
    case 26: return "Right Knee"; case 27: return "Left Ankle"; case 28: return "Right Ankle";
    default: return "";
  }
};

const getArrowColor = (distance) => {
  if (distance < 0.05) return '#eab308'; // yellow
  else if (distance < 0.1) return '#f97316'; // orange
  else return '#ef4444'; // red
};

const drawArrow = (ctx, fromX, fromY, toX, toY, color, lineWidth) => {
  const headLength = 15; const dx = toX - fromX; const dy = toY - fromY;
  const angle = Math.atan2(dy, dx); ctx.beginPath(); ctx.moveTo(fromX, fromY);
  ctx.lineTo(toX, toY); ctx.strokeStyle = color; ctx.lineWidth = lineWidth; ctx.stroke();
  ctx.beginPath(); ctx.moveTo(toX, toY);
  ctx.lineTo(toX - headLength * Math.cos(angle - Math.PI/6), toY - headLength * Math.sin(angle - Math.PI/6));
  ctx.lineTo(toX - headLength * Math.cos(angle + Math.PI/6), toY - headLength * Math.sin(angle + Math.PI/6));
  ctx.closePath(); ctx.fillStyle = color; ctx.fill();
};

// --- Drawing functions ---
const drawUserPose = (ctx, landmarks, canvasWidth, canvasHeight) => {
  if (!landmarks) return;
  for (let i = 11; i < landmarks.length; i++) {
    const landmark = landmarks[i];
    if (!landmark || landmark.x == null || landmark.y == null) continue;
    const x = landmark.x * canvasWidth; const y = landmark.y * canvasHeight;
    ctx.beginPath(); ctx.arc(x, y, 5, 0, 2 * Math.PI); ctx.fillStyle = '#8b5cf6'; ctx.fill();
  }
  POSE_CONNECTIONS.forEach(([i, j]) => {
    if (i >= 11 && j >= 11 && i < landmarks.length && j < landmarks.length) {
      const p1 = landmarks[i]; const p2 = landmarks[j];
      if (!p1 || !p2 || p1.x == null || p1.y == null || p2.x == null || p2.y == null) return;
      ctx.beginPath(); ctx.moveTo(p1.x * canvasWidth, p1.y * canvasHeight);
      ctx.lineTo(p2.x * canvasWidth, p2.y * canvasHeight);
      ctx.strokeStyle = '#a78bfa'; ctx.lineWidth = 2; ctx.stroke();
    }
  });
};

const findJointsToCorrect = (landmarks, corrections) => {
  if (!landmarks || !corrections || Object.keys(corrections).length === 0) { return []; }
  const correctJoints = [];
  Object.keys(corrections).forEach(indexStr => {
    const i = parseInt(indexStr); const correction = corrections[indexStr];
    if (correction && correction.x != null && correction.y != null &&
        (Math.abs(correction.x) > 0.01 || Math.abs(correction.y) > 0.01) &&
        i < landmarks.length && landmarks[i]) {
      const jointName = getJointName(i);
      if (jointName) { correctJoints.push({ name: jointName, index: i, correction: correction }); }
    }
  });
  return correctJoints;
};

const drawCorrectionArrows = (ctx, landmarks, corrections, jointsToCorrect, canvasWidth, canvasHeight) => {
  if (!landmarks || !corrections || jointsToCorrect.length === 0) { return; }
  jointsToCorrect.forEach(joint => {
    const i = joint.index;
    if (!landmarks[i] || landmarks[i].x == null || landmarks[i].y == null) { return; }
    const originalX = landmarks[i].x * canvasWidth; const originalY = landmarks[i].y * canvasHeight;
    const correction = corrections[String(i)];
    if (!correction || correction.x == null || correction.y == null) { return; }
    const vectorX = correction.x * canvasWidth; const vectorY = correction.y * canvasHeight;
    const targetX = originalX + vectorX; const targetY = originalY + vectorY;
    const extendedTargetX = originalX + vectorX * 1.5; const extendedTargetY = originalY + vectorY * 1.5;
    const correctionMagnitude = Math.sqrt(correction.x * correction.x + correction.y * correction.y);
    const arrowColor = getArrowColor(correctionMagnitude);
    drawArrow(ctx, originalX, originalY, extendedTargetX, extendedTargetY, arrowColor, 6);
  });
};

// --- UI Components ---

// Animation variants
const fadeIn = {
  hidden: { opacity: 0 },
  visible: { 
    opacity: 1,
    transition: { duration: 0.4 }
  },
  exit: { 
    opacity: 0,
    transition: { duration: 0.2 }
  }
};

const slideUp = {
  hidden: { opacity: 0, y: 20 },
  visible: { 
    opacity: 1, 
    y: 0,
    transition: { duration: 0.4 }
  },
  exit: { 
    opacity: 0, 
    y: 20,
    transition: { duration: 0.2 }
  }
};

const ConnectionStatus = ({ status }) => {
  const statusInfo = {
    connected: {
      color: "bg-green-500", 
      text: "Connected",
      icon: <Check className="w-3 h-3 text-green-500" />
    },
    connecting: {
      color: "bg-yellow-500", 
      text: "Connecting...",
      icon: <RefreshCw className="w-3 h-3 text-yellow-500 animate-spin" />
    },
    disconnected: {
      color: "bg-red-500", 
      text: "Disconnected",
      icon: <XCircle className="w-3 h-3 text-red-500" />
    }
  };
  
  const { color, text, icon } = statusInfo[status];
  
  return (
    <motion.div 
      className="absolute top-4 right-4 z-40"
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
    >
      <div className="flex items-center space-x-2 bg-black/40 backdrop-blur-sm text-white px-3 py-2 rounded-lg shadow-lg">
        <div className={`w-2 h-2 rounded-full ${color}`}></div>
        <span className="text-xs font-medium">{text}</span>
        <div className="ml-1">{icon}</div>
      </div>
    </motion.div>
  );
};

const FullscreenButton = ({ isFullscreen, toggleFullscreen }) => {
  return (
    <motion.button 
      onClick={toggleFullscreen}
      className="absolute top-4 left-4 z-40 bg-black/40 hover:bg-black/60 text-white p-2.5 rounded-lg transition-colors backdrop-blur-sm shadow-lg"
      title={isFullscreen ? "Exit Fullscreen" : "Enter Fullscreen"}
      whileHover={{ scale: 1.1 }}
      whileTap={{ scale: 0.9 }}
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
    >
      {isFullscreen ? <MinimizeIcon className="w-5 h-5" /> : <FullscreenIcon className="w-5 h-5" />}
    </motion.button>
  );
};

const WorkoutSelector = ({ selectedWorkout, onSelectWorkout }) => {
  const [isOpen, setIsOpen] = useState(false);
  
  return (
    <motion.div 
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: 0.1 }}
    >
      <div className="relative">
        <motion.button
          onClick={() => setIsOpen(!isOpen)}
          className="flex items-center space-x-2 bg-black/40 hover:bg-black/60 text-white px-4 py-2 rounded-lg transition-colors shadow-md backdrop-blur-sm border border-gray-600/30"
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <span>{workoutMap[selectedWorkout]}</span>
          {isOpen ? (
            <ChevronUp className="w-4 h-4" />
          ) : (
            <ChevronDown className="w-4 h-4" />
          )}
        </motion.button>
        
        <AnimatePresence>
          {isOpen && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="absolute mt-1 left-0 right-0 max-h-60 overflow-y-auto bg-black/80 text-white border border-gray-600/30 rounded-lg shadow-xl z-50 backdrop-blur-md"
            >
              {Object.entries(workoutMap).map(([id, name]) => (
                <div
                  key={id}
                  onClick={() => {
                    onSelectWorkout(Number(id));
                    setIsOpen(false);
                  }}
                  className={`px-4 py-2 cursor-pointer ${
                    Number(id) === selectedWorkout
                      ? 'bg-purple-500/30 text-purple-200'
                      : ''
                  } hover:bg-gray-700/50 transition-colors`}
                >
                  {name}
                </div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
};

const MuscleGroupVisualizer = ({ isVisible, muscleGroup }) => {
  // Get the appropriate muscle data based on the current muscle group
  let muscleData;
  
  // Use a safer approach - hardcode a valid array for the default case
  if (muscleGroup === 0 || muscleGroup > 7) {
    // Hardcoded safe default with a valid muscle to avoid library errors
    muscleData = [
      { name: 'Rest', muscles: [] }
    ];
  } else {
    // Function to map muscle group IDs to React Body Highlighter muscle types
    switch(muscleGroup) {
      case 1: // shoulders
        muscleData = [
          { name: 'Shoulder Press', muscles: [MuscleType.BACK_DELTOIDS, MuscleType.FRONT_DELTOIDS] }
        ];
        break;
      case 2: // chest
        muscleData = [
          { name: 'Bench Press', muscles: [MuscleType.CHEST] }
        ];
        break;
      case 3: // biceps
        muscleData = [
          { name: 'Bicep Curl', muscles: [MuscleType.BICEPS] }
        ];
        break;
      case 4: // core
        muscleData = [
          { name: 'Crunches', muscles: [MuscleType.ABS, MuscleType.OBLIQUES] }
        ];
        break;
      case 5: // triceps
        muscleData = [
          { name: 'Tricep Pushdown', muscles: [MuscleType.TRICEPS] }
        ];
        break;
      case 6: // legs
        muscleData = [
          { name: 'Squats', muscles: [MuscleType.QUADRICEPS, MuscleType.HAMSTRING, MuscleType.CALVES] },
          { name: 'Hip Thrust', muscles: [MuscleType.GLUTEAL] }
        ];
        break;
      case 7: // back
        muscleData = [
          { name: 'Lat Pulldown', muscles: [MuscleType.UPPER_BACK, MuscleType.LOWER_BACK] }
        ];
        break;
    }
  }
  
  // Now just check if we want to show the visualizer
  return (
    <AnimatePresence>
      {isVisible && (
        <>
          {/* Front view - Left side */}
          <motion.div 
            className="absolute bottom-20 left-8 z-30 pointer-events-none w-[120px] sm:w-[180px] lg:w-[200px] max-w-[30vw]"
            initial="hidden"
            animate="visible"
            exit="exit"
            variants={fadeIn}
          >
            <Model 
              data={muscleData}
              type={ModelType.ANTERIOR}
              highlightedColors={['#a855f7']}
              onClick={() => {}} // Empty handler to prevent errors
            />
          </motion.div>
          
          {/* Back view - Right side */}
          <motion.div 
            className="absolute bottom-20 right-8 z-30 pointer-events-none w-[120px] sm:w-[180px] lg:w-[200px] max-w-[30vw]"
            initial="hidden"
            animate="visible"
            exit="exit"
            variants={fadeIn}
          >
            <Model 
              data={muscleData}
              type={ModelType.POSTERIOR}
              highlightedColors={['#a855f7']}
              onClick={() => {}} // Empty handler to prevent errors
            />
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};

const MuscleVisualizerToggle = ({ isVisible, toggleVisibility }) => {
  return (
    <motion.button 
      onClick={toggleVisibility}
      className={`absolute bottom-4 right-8 z-40 ${
        isVisible 
          ? 'bg-purple-600/70 hover:bg-purple-700/80' 
          : 'bg-black/40 hover:bg-black/60'
      } text-white p-2.5 rounded-lg transition-colors backdrop-blur-sm shadow-lg`}
      title={isVisible ? "Hide Muscle Activation" : "Show Muscle Activation"}
      whileHover={{ scale: 1.1 }}
      whileTap={{ scale: 0.9 }}
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
    >
      {isVisible ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
    </motion.button>
  );
};

const CustomNotification = ({ type, message, onClose }) => {
  // Define color schemes based on notification type
  const notificationConfig = {
    success: {
      bg: "bg-green-500/90",
      border: "border-green-400",
      icon: <Check className="w-5 h-5" />
    },
    info: {
      bg: "bg-blue-500/90",
      border: "border-blue-400",
      icon: <Info className="w-5 h-5" />
    },
    warning: {
      bg: "bg-yellow-500/90",
      border: "border-yellow-400",
      icon: <AlertTriangle className="w-5 h-5" />
    },
    error: {
      bg: "bg-red-500/90",
      border: "border-red-400",
      icon: <XCircle className="w-5 h-5" />
    }
  };

  // Use info as default type
  const config = notificationConfig[type] || notificationConfig.info;
  
  return (
    <motion.div 
      className={`${config.bg} backdrop-blur-md text-white rounded-lg shadow-lg overflow-hidden mb-4 border-l-4 ${config.border}`}
      initial={{ opacity: 0, x: 50 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 50 }}
      transition={{ duration: 0.3 }}
    >
      <div className="flex items-start p-4">
        <div className="flex-shrink-0 mr-3">
          {config.icon}
        </div>
        <div className="flex-grow">
          <p className="font-medium">{message}</p>
        </div>
        <button 
          onClick={onClose}
          className="flex-shrink-0 ml-3 p-1 rounded-full hover:bg-white/20 transition-colors"
        >
          <XCircle className="w-5 h-5" />
        </button>
      </div>
    </motion.div>
  );
};

const InfoPanel = ({ 
  currentWorkout, 
  predictedWorkout, 
  predictionConfidence, 
  predictionThreshold,
  predictedMuscleGroup,
  muscleGroupConfidence,
  feedbackLatency,
  receivedCount,
  connectionStatus
}) => {
  return (
    <motion.div 
      className="bg-black/60 text-white px-5 py-3 rounded-xl backdrop-blur-md border border-white/10 shadow-lg"
      initial="hidden"
      animate="visible"
      variants={slideUp}
    >
      <div className="flex flex-col space-y-2">
        <div className="flex justify-between items-center">
          <div className="flex items-center">
            <span className="text-sm font-medium opacity-70">Current:</span>
            <span className="ml-2 font-semibold">{workoutMap[currentWorkout]}</span>
          </div>          
        </div>

        {predictedWorkout !== currentWorkout && (
            <div className="flex items-center">
              <span className="text-sm font-medium opacity-70">Suggested:</span>
              <span className={`ml-2 px-2 py-0.5 rounded-full text-sm ${
                predictionConfidence > predictionThreshold 
                  ? 'bg-purple-500/30 text-purple-200 border border-purple-500/30' 
                  : 'bg-gray-500/30 text-gray-200 border border-gray-500/30'
              }`}>
                {workoutMap[predictedWorkout]}
              </span>
              {predictionConfidence > 0 && (
                <div className="ml-2 flex items-center">
                  <div className="w-4 h-1 bg-gray-700/50 rounded-full overflow-hidden">
                    <div 
                      className={`h-full rounded-full ${
                        predictionConfidence > predictionThreshold ? 'bg-purple-500' : 'bg-gray-500'
                      }`}
                      style={{ width: `${predictionConfidence * 100}%` }}
                    ></div>
                  </div>
                  <span className={`ml-1 text-xs ${
                    predictionConfidence > predictionThreshold ? 'text-purple-300' : 'opacity-60'
                  }`}>
                    {Math.round(predictionConfidence * 100)}%
                  </span>
                </div>
              )}
            </div>
          )}
        
        {predictedMuscleGroup > 0 && (
          <div className="flex items-center">
            <span className="text-sm font-medium opacity-70">Muscle Group:</span>
            <div className="flex items-center ml-2">
              <Zap className={`w-4 h-4 mr-1 ${
                muscleGroupConfidence > predictionThreshold ? 'text-purple-400' : 'opacity-60'
              }`} />
              <span className="font-semibold">{muscleGroupMap[predictedMuscleGroup]}</span>
              {muscleGroupConfidence > 0 && (
                <div className="ml-2 flex items-center">
                  <div className="w-4 h-1 bg-gray-700/50 rounded-full overflow-hidden">
                    <div 
                      className={`h-full rounded-full ${
                        muscleGroupConfidence > predictionThreshold ? 'bg-purple-500' : 'bg-gray-500'
                      }`}
                      style={{ width: `${muscleGroupConfidence * 100}%` }}
                    ></div>
                  </div>
                  <span className={`ml-1 text-xs ${
                    muscleGroupConfidence > predictionThreshold ? 'text-purple-300' : 'opacity-60'
                  }`}>
                    {Math.round(muscleGroupConfidence * 100)}%
                  </span>
                </div>
              )}
            </div>
          </div>
        )}
        
        <div className="flex items-center justify-center text-xs opacity-60 space-x-4">
          {(feedbackLatency > 0 || receivedCount > 0) && (
            <>
              <span>Latency: {feedbackLatency > 0 ? `${feedbackLatency}ms` : 'N/A'}</span>
              <span>Corrections: {receivedCount}</span>
            </>
          )}
          <span className="flex items-center">
            <div className={`w-1.5 h-1.5 rounded-full mr-1 ${
              connectionStatus === 'connected' ? 'bg-green-500' : 
              connectionStatus === 'connecting' ? 'bg-yellow-500' : 
              'bg-red-500'
            }`}></div>
            {connectionStatus}
          </span>
        </div>
      </div>
    </motion.div>
  );
};

// --- Main TrainingPage Component ---
const TrainingPage = () => {
  const navigate = useNavigate();
  const [isDarkMode, setIsDarkMode] = useState(false);
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [userLandmarksForDrawing, setUserLandmarksForDrawing] = useState(null);
  const latestLandmarksRef = useRef(null);
  const [corrections, setCorrections] = useState({});
  const latestCorrectionsRef = useRef({});
  const socketRef = useRef(null);
  const [connectionStatus, setConnectionStatus] = useState("connecting");
  const lastLandmarkUpdateRef = useRef(0);
  const lastCorrectionTimeRef = useRef(0);
  const [feedbackLatency, setFeedbackLatency] = useState(0);
  const [receivedCount, setReceivedCount] = useState(0);
  const poseInstanceRef = useRef(null);
  const cameraInstanceRef = useRef(null);
  const sendIntervalRef = useRef(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  
  // Add state for muscle visualizer toggle
  const [showMuscleVisualizer, setShowMuscleVisualizer] = useState(true);
  
  // Modified workout prediction state
  const [predictedWorkout, setPredictedWorkout] = useState(12); // Default to plank (12)
  
  // Replace the selectedWorkout state with currentWorkout
  const [currentWorkout, setCurrentWorkout] = useState(12); // Default to plank (12)
  // Add ref to track current workout synchronously
  const currentWorkoutRef = useRef(12);
  
  // Add state for muscle group prediction
  const [predictedMuscleGroup, setPredictedMuscleGroup] = useState(0); // Default to none
  
  // Add state for muscle group prediction stability
  const [recentMuscleGroupPredictions, setRecentMuscleGroupPredictions] = useState([]);
  const [muscleGroupConfidence, setMuscleGroupConfidence] = useState(0);
  
  // Add state for workout prediction stability
  const [recentWorkoutPredictions, setRecentWorkoutPredictions] = useState([]);
  const PREDICTION_WINDOW_SIZE = 30; // Keep track of ~1.5 seconds of predictions (at 50ms intervals)
  const CONFIDENCE_THRESHOLD = 0.6; // 60% majority needed to change workout
  const lastStableWorkoutRef = useRef(12); // Track the last stable workout type
  const lastStableMuscleGroupRef = useRef(0); // Track the last stable muscle group
  // Add state to track overall confidence
  const [predictionConfidence, setPredictionConfidence] = useState(0);
  
  // Modify the notification system variables
  const SUGGESTION_THRESHOLD = 0.8; // Higher threshold for suggestions
  const AUTO_CHANGE_THRESHOLD = 0.95; // Very high threshold for auto-changing (increased)
  const AUTO_CHANGE_CONSECUTIVE_FRAMES = 40; // Need consistent prediction for ~2 seconds (increased)
  const MIN_SUGGESTION_COUNT = 15; // Require at least ~0.75 seconds of consistent detection before suggesting
  const [suggestionCounts, setSuggestionCounts] = useState({});
  const [lastSuggestedWorkout, setLastSuggestedWorkout] = useState(null);
  const [lastNotificationTime, setLastNotificationTime] = useState(0);
  // Add a ref to synchronously track the next allowed notification time
  const nextAllowedChangeTimeRef = useRef(0);
  const NOTIFICATION_COOLDOWN = 5000; // 20 seconds between notifications (doubled)
  const [declinedWorkouts, setDeclinedWorkouts] = useState({}); // Track declined suggestions
  const DECLINE_COOLDOWN = 60000; // 1 minute cooldown after declining a suggestion
  
  // Update the ratio window size from 15 seconds to 5 seconds
  const RATIO_WINDOW_SIZE = 100; // Track ~5 seconds (100 frames at 50ms intervals)
  const SUGGESTION_RATIO_THRESHOLD = 0.75; // Require 75% of recent predictions to be the same workout
  const AUTO_CHANGE_RATIO_THRESHOLD = 0.85; // Require 85% for auto-change
  const [workoutHistoryWindow, setWorkoutHistoryWindow] = useState([]); // Track history for ratio calculation
  
  // Add state for our custom notifications
  const [notifications, setNotifications] = useState([]);
  // Add notification queue
  const [notificationQueue, setNotificationQueue] = useState([]);
  const [isProcessingNotification, setIsProcessingNotification] = useState(false);
  
  // Function to add a notification to the queue
  const addNotification = (notificationData) => {
    const id = Date.now(); // Simple unique ID based on timestamp
    
    // Add to the queue instead of showing immediately
    setNotificationQueue(prev => [...prev, { ...notificationData, id }]);
    
    return id; // Return the ID for reference
  };
  
  // Process the notification queue
  useEffect(() => {
    if (notificationQueue.length > 0 && !isProcessingNotification) {
      // Get the first notification from the queue
      const nextNotification = notificationQueue[0];
      
      // Remove it from the queue
      setNotificationQueue(prev => prev.slice(1));
      
      // Set the active notification
      setNotifications([nextNotification]);
      setIsProcessingNotification(true);
      
      // Set timeout to clear this notification
      const timeout = nextNotification.autoClose || 3000;
      setTimeout(() => {
        setNotifications([]);
        setIsProcessingNotification(false);
      }, timeout);
    }
  }, [notificationQueue, isProcessingNotification]);
  
  // Fix the removeNotification function to be more robust
  const removeNotification = (id) => {
    setNotifications(prev => prev.filter(notification => notification.id !== id));
    setIsProcessingNotification(false);
  };
  
  // Update the clearAllNotifications function to clear everything
  const clearAllNotifications = () => {
    setNotifications([]);
    setNotificationQueue([]);
    setIsProcessingNotification(false);
  };
  
  // Simplify handleWorkoutChange to just set the workout without tracking previous
  const handleWorkoutChange = (newWorkoutId) => {
    const parsedId = Number(newWorkoutId);
    // Only call setCurrentWorkout once
    setCurrentWorkout(parsedId);
    // Update ref synchronously to always have the latest value
    currentWorkoutRef.current = parsedId;
    clearAllNotifications();
    
    // Update lastNotificationTime to prevent immediate new notifications
    setLastNotificationTime(Date.now());
    
    // Log the NEW workout that's being set rather than the current one
    console.log(`[Handle Workout Change] Changed from ${workoutMap[currentWorkout]} to ${workoutMap[parsedId]}`);
    
    // Also update the ref to ensure synchronous blocking of future changes
    nextAllowedChangeTimeRef.current = Date.now() + NOTIFICATION_COOLDOWN;
  };

  // Replace the updateStableWorkoutPrediction function with improved auto-change approach
  const updateStableWorkoutPrediction = (newPrediction) => {
    // Update the recent predictions array for confidence calculation
    setRecentWorkoutPredictions(prev => {
      // Add new prediction and keep window size limited
      const updated = [...prev, newPrediction].slice(-PREDICTION_WINDOW_SIZE);
      
      // Count occurrences of each workout type in our window
      const counts = updated.reduce((acc, type) => {
        acc[type] = (acc[type] || 0) + 1;
        return acc;
      }, {});
      
      // Find the most frequent workout type
      let mostFrequent = null;
      let highestCount = 0;
      
      Object.entries(counts).forEach(([type, count]) => {
        if (count > highestCount) {
          highestCount = count;
          mostFrequent = Number(type);
        }
      });
      
      // Calculate confidence (percentage of window with this prediction)
      const confidence = highestCount / updated.length;
      
      // Store the confidence value for UI display
      setPredictionConfidence(confidence);
      
      // Set the predicted workout type only if it's not the same as the current workout
      if (mostFrequent !== currentWorkoutRef.current) {
        setPredictedWorkout(mostFrequent);
      }
      
      return updated;
    });
    
    // Update longer history window for ratio-based suggestions
    setWorkoutHistoryWindow(prev => {
      // Keep the history window at RATIO_WINDOW_SIZE
      const updatedHistory = [...prev, newPrediction].slice(-RATIO_WINDOW_SIZE);
      
      // Calculate the ratio of each workout type in the history window
      const typeCount = {};
      updatedHistory.forEach(type => {
        typeCount[type] = (typeCount[type] || 0) + 1;
      });
      
      // If the history window is sufficiently full (at least half capacity)
      if (updatedHistory.length >= RATIO_WINDOW_SIZE / 2) {
        // Get the workout with the highest count
        let mostFrequentType = null;
        let highestCount = 0;
        
        Object.entries(typeCount).forEach(([type, count]) => {
          if (count > highestCount) {
            highestCount = count;
            mostFrequentType = Number(type);
          }
        });
        
        // Calculate the ratio of the most frequent workout
        const ratio = highestCount / updatedHistory.length;
        
        // If we have a different workout than selected with high ratio
        if (mostFrequentType !== currentWorkoutRef.current && ratio >= SUGGESTION_RATIO_THRESHOLD) {
          const now = Date.now();
          
          // Check if we're past the cooldown period using the ref for synchronous checking
          if (now >= nextAllowedChangeTimeRef.current) {
            // Check if this workout was recently declined
            const workoutKey = String(mostFrequentType);
            const declineTime = declinedWorkouts[workoutKey] || 0;
            const timeSinceDecline = now - declineTime;
            
            // Skip suggestion if recently declined
            if (timeSinceDecline < DECLINE_COOLDOWN) {
              return updatedHistory;
            }
            
            // Immediately update the ref to block subsequent calls
            nextAllowedChangeTimeRef.current = now + NOTIFICATION_COOLDOWN;
            
            // AUTO-CHANGE with notification when we detect a different workout with high confidence
            // Only change workout if it's different from the current one
            if (mostFrequentType !== currentWorkoutRef.current) {
              console.log(`[Auto-Change] Detected different workout: ${workoutMap[mostFrequentType]}`);
              // log current and predicted workout
              console.log(`[Auto-Change] Current workout: ${workoutMap[currentWorkout]}`);
              console.log(`[Auto-Change] Predicted workout: ${workoutMap[mostFrequentType]}`);
              // Then handle the workout change
              handleWorkoutChange(mostFrequentType);
              
              // Show notification about the change
              addNotification({
                type: 'info',
                message: `Changed to ${workoutMap[mostFrequentType]}`,
                autoClose: 3000
              });
              
              setLastSuggestedWorkout(mostFrequentType);
              setLastNotificationTime(now);
            }
          }
        }
      }
      
      return updatedHistory;
    });
  };
  
  // Add new function to stabilize muscle group predictions
  const updateStableMuscleGroupPrediction = (newPrediction) => {
    // Update the array of recent predictions
    setRecentMuscleGroupPredictions(prev => {
      // Add new prediction and keep window size limited
      const updated = [...prev, newPrediction].slice(-PREDICTION_WINDOW_SIZE);
      
      // Count occurrences of each muscle group in our window
      const counts = updated.reduce((acc, type) => {
        acc[type] = (acc[type] || 0) + 1;
        return acc;
      }, {});
      
      // Find the most frequent muscle group
      let mostFrequent = null;
      let highestCount = 0;
      
      Object.entries(counts).forEach(([type, count]) => {
        if (count > highestCount) {
          highestCount = count;
          mostFrequent = Number(type);
        }
      });
      
      // Calculate confidence (percentage of window with this prediction)
      const confidence = highestCount / updated.length;
      
      // Store the confidence value 
      setMuscleGroupConfidence(confidence);
      
      // Only update the displayed muscle group if confidence passes threshold
      // or if it's the same as our current stable muscle group
      if (confidence >= CONFIDENCE_THRESHOLD || mostFrequent === lastStableMuscleGroupRef.current) {
        setPredictedMuscleGroup(mostFrequent);
        lastStableMuscleGroupRef.current = mostFrequent;
      }
      
      return updated;
    });
  };

  // Add function to toggle fullscreen
  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      // Enter fullscreen
      document.documentElement.requestFullscreen().catch(err => {
        console.error(`Error attempting to enable fullscreen: ${err.message}`);
      });
    } else {
      // Exit fullscreen
      if (document.exitFullscreen) {
        document.exitFullscreen();
      }
    }
  };

  // Add event listener for fullscreen change
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
    };
  }, []);

  // --- WebSocket Connection Setup ---
  useEffect(() => {
    const connectSocket = () => {
      if (socketRef.current) { socketRef.current.disconnect(); }
      console.log('[Socket Setup] Attempting to connect WebSocket...');
      setConnectionStatus("connecting");
      socketRef.current = io('http://localhost:8001', {
        reconnection: true, reconnectionAttempts: Infinity, reconnectionDelay: 1000,
        reconnectionDelayMax: 5000, timeout: 10000, transports: ['websocket']
      });
      socketRef.current.on('connect', () => { 
        console.log('[Socket Status] WebSocket connected. ID:', socketRef.current?.id); 
        setConnectionStatus("connected"); 
      });
      socketRef.current.on('disconnect', (reason) => { 
        console.warn('[Socket Status] WebSocket disconnected. Reason:', reason); 
        setConnectionStatus("disconnected"); 
      });
      socketRef.current.on('connect_error', (error) => { 
        console.error('[Socket Status] WebSocket connection error:', error); 
        setConnectionStatus("disconnected"); 
      });
      socketRef.current.on('connected', (data) => { 
        console.log('[Socket Event] Received "connected" confirmation:', data); 
      });

      // --- Corrections Handler ---
      socketRef.current.on('pose_corrections', (data) => {
        // Update the ref with the latest data immediately
        latestCorrectionsRef.current = data;

        // Update state as well (might trigger other UI updates)
        setCorrections(data);

        // Check if there's a predicted workout type in the data
        if (data.predicted_workout_type !== undefined) {
          // Instead of directly setting the workout, update our stable prediction
          updateStableWorkoutPrediction(data.predicted_workout_type);
        }
        
        // Check if there's a predicted muscle group in the data
        if (data.predicted_muscle_group !== undefined) {
          // Update our stable muscle group prediction
          updateStableMuscleGroupPrediction(data.predicted_muscle_group);
        }

        // Update timing info
        const now = Date.now();
        const latency = lastLandmarkUpdateRef.current ? now - lastLandmarkUpdateRef.current : 0;
        setFeedbackLatency(latency);
        setReceivedCount(prevCount => prevCount + 1);
        lastCorrectionTimeRef.current = now;
      });
      // --- END Corrections Handler ---

      socketRef.current.on('error', (data) => { 
        console.error('[Socket Event] Received server error:', data.message || data); 
      });
    };
    
    connectSocket();
    
    return () => { 
      if (socketRef.current) { 
        console.log("[Socket Cleanup] Disconnecting WebSocket."); 
        socketRef.current.disconnect(); 
        socketRef.current = null; 
      } 
    };
  }, []);

  // --- Landmark Update Function ---
  const updateLandmarks = (landmarks) => {
    if (landmarks && landmarks.length > 0) {
        setUserLandmarksForDrawing(landmarks);
        latestLandmarksRef.current = landmarks;
        lastLandmarkUpdateRef.current = Date.now();
    } else {
        setUserLandmarksForDrawing(null);
        latestLandmarksRef.current = null;
    }
  };

  // Modify the send interval to include the selected workout type
  useEffect(() => {
    const sendIntervalDelay = 50;
    if (sendIntervalRef.current) { clearInterval(sendIntervalRef.current); }
    console.log(`[Data Send] Setting up interval to send data every ${sendIntervalDelay}ms`);
    sendIntervalRef.current = setInterval(() => {
      const landmarksToSend = latestLandmarksRef.current;
      const socketIsConnected = socketRef.current?.connected;
      if (landmarksToSend && socketIsConnected) {
        const sendTimestamp = Date.now();
        // Send both landmarks and the selected workout type
        socketRef.current.emit('pose_data', { 
          landmarks: landmarksToSend, 
          timestamp: sendTimestamp,
          selected_workout: currentWorkout
        });
      }
    }, sendIntervalDelay);
    
    return () => { 
      if (sendIntervalRef.current) { 
        console.log("[Data Send] Clearing send interval."); 
        clearInterval(sendIntervalRef.current); 
        sendIntervalRef.current = null; 
      } 
    };
  }, [currentWorkout]); // Add currentWorkout as a dependency

  // --- Correction Timeout Check ---
  useEffect(() => {
    lastCorrectionTimeRef.current = 0;
    const checkInterval = 5000;
    const checkCorrectionTimeout = setInterval(() => {
      const now = Date.now();
      if (connectionStatus === "connected" && latestLandmarksRef.current && lastLandmarkUpdateRef.current !== 0) {
          const timeSinceLastCorrection = now - lastCorrectionTimeRef.current;
          if (lastCorrectionTimeRef.current === 0 && (now - lastLandmarkUpdateRef.current > checkInterval)) {
              console.warn(`[Correction Check] Connected >${checkInterval/1000}s, sending data, but no corrections received yet. Verify backend.`);
          } else if (lastCorrectionTimeRef.current !== 0 && timeSinceLastCorrection > checkInterval) {
              console.warn(`[Correction Check] No corrections received for >${checkInterval/1000}s. Last at: ${new Date(lastCorrectionTimeRef.current).toLocaleTimeString()}. Forcing socket reconnect.`);
              if (socketRef.current) { socketRef.current.disconnect(); }
          }
      }
    }, checkInterval);
    
    return () => clearInterval(checkCorrectionTimeout);
  }, [connectionStatus]);

  // Toggle dark mode function
  const toggleDarkMode = () => {
    setIsDarkMode(!isDarkMode);
  };

  // Dark Mode Setup
  useEffect(() => {
    const prefersDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
    setIsDarkMode(prefersDarkMode);
  }, []);
  
  useEffect(() => {
    document.body.classList.toggle('dark', isDarkMode);
  }, [isDarkMode]);

  // --- MediaPipe Pose Setup ---
  useEffect(() => {
    console.log("[MediaPipe Setup] Effect triggered.");
    poseInstanceRef.current = null;
    cameraInstanceRef.current = null;
    let initTimeoutId = null;

    // --- MediaPipe Results Callback ---
    function onResults(results) {
        const canvas = canvasRef.current;
        if (!canvas || !webcamRef.current) { return; }
        const ctx = canvas.getContext('2d');
        if (!ctx) { return; }
        const videoWidth = webcamRef.current.videoWidth;
        const videoHeight = webcamRef.current.videoHeight;
        if (videoWidth === 0 || videoHeight === 0) { return; }
        if (canvas.width !== videoWidth) canvas.width = videoWidth;
        if (canvas.height !== videoHeight) canvas.height = videoHeight;

        ctx.save();
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (results.poseLandmarks) {
            const landmarks = results.poseLandmarks;
            updateLandmarks(landmarks); // Update ref and state

            // --- Drawing Logic ---
            drawUserPose(ctx, landmarks, canvas.width, canvas.height); // Draw skeleton

            // Use the latestCorrectionsRef for drawing
            const currentCorrections = latestCorrectionsRef.current;
            const jointsToCorrect = findJointsToCorrect(landmarks, currentCorrections);

            // Only attempt to draw if there are joints needing correction
            if (jointsToCorrect.length > 0) {
                 drawCorrectionArrows(ctx, landmarks, currentCorrections, jointsToCorrect, canvas.width, canvas.height);
            }

        } else {
            updateLandmarks(null);
        }
        ctx.restore();
    }

    const startMediaPipe = async () => {
        if (typeof window === 'undefined' || !webcamRef.current) {
            console.log("[MediaPipe Setup] Aborted: window or webcamRef not ready."); 
            return;
        }
        console.log("[MediaPipe Setup] Initializing Pose...");
        const pose = new poseDetection.Pose({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose@${poseDetection.VERSION}/${file}`
        });
        poseInstanceRef.current = pose;
        pose.setOptions({ 
          modelComplexity: 1, 
          smoothLandmarks: true, 
          enableSegmentation: false, 
          minDetectionConfidence: 0.5, 
          minTrackingConfidence: 0.5 
        });
        pose.onResults(onResults);

        initTimeoutId = setTimeout(async () => {
            try {
                console.log("[MediaPipe Setup] Attempting pose.initialize() after delay...");
                await pose.initialize();
                console.log("[MediaPipe Setup] Pose initialized successfully after delay.");
                if (!webcamRef.current) { 
                  console.warn("[MediaPipe Setup] Webcam ref null before camera setup."); 
                  return; 
                }
                console.log("[MediaPipe Setup] Setting up Camera...");
                const camera = new Camera(webcamRef.current, {
                    onFrame: async () => {
                        if (poseInstanceRef.current && webcamRef.current) {
                            try { 
                              await poseInstanceRef.current.send({ image: webcamRef.current }); 
                            }
                            catch (sendError) { 
                              console.error("[MediaPipe onFrame] Error sending frame:", sendError); 
                            }
                        }
                    },
                    width: 640, 
                    height: 480
                });
                cameraInstanceRef.current = camera;
                await camera.start();
                console.log("[MediaPipe Setup] Camera started successfully.");
            } catch (initError) {
                console.error("[MediaPipe Setup] Error initializing MediaPipe Pose (after delay):", initError);
                if (poseInstanceRef.current) { 
                  poseInstanceRef.current.close(); 
                  poseInstanceRef.current = null; 
                }
                if (cameraInstanceRef.current) { 
                  cameraInstanceRef.current = null; 
                }
            }
        }, 100);
    };

    startMediaPipe();

    return () => {
        console.log("[MediaPipe Cleanup] Cleaning up...");
        if (initTimeoutId) { clearTimeout(initTimeoutId); }
        if (cameraInstanceRef.current) { 
          cameraInstanceRef.current.stop(); 
          cameraInstanceRef.current = null; 
        }
        if (poseInstanceRef.current) { 
          poseInstanceRef.current.close(); 
          poseInstanceRef.current = null; 
        }
    };
  }, []); // Empty dependency array

  // Function to toggle muscle visualizer
  const toggleMuscleVisualizer = () => {
    setShowMuscleVisualizer(!showMuscleVisualizer);
  };

  // --- Render ---
  return (
    <motion.section 
      className={`min-h-screen overflow-hidden ${
        isDarkMode 
          ? 'bg-gradient-to-br from-gray-900 via-gray-800 to-indigo-900' 
          : 'bg-gradient-to-br from-gray-100 via-gray-100 to-indigo-100'
      }`}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      {!isFullscreen && <NavBar isDarkMode={isDarkMode} toggleDarkMode={toggleDarkMode} />}
      
      {/* Custom Notifications Container */}
      <AnimatePresence>
        {notifications.length > 0 && (
          <motion.div 
            className="fixed top-24 right-6 z-50 w-80 max-w-[90%]"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            {notifications.map(notification => (
              <CustomNotification
                key={notification.id}
                type={notification.type}
                message={notification.message}
                onClose={() => removeNotification(notification.id)}
              />
            ))}
          </motion.div>
        )}
      </AnimatePresence>
      
      <motion.main 
        className={`${isFullscreen ? 'h-screen' : 'container mx-auto px-4 py-4 sm:py-6 max-w-5xl'}`}
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.4, delay: 0.1 }}
      >
        <motion.div 
          className={`${
            isFullscreen 
              ? 'fixed inset-0 bg-black m-0 p-0 max-w-none rounded-none' 
              : 'mx-auto bg-black/20 dark:bg-gray-800/30 backdrop-blur-md rounded-2xl shadow-xl overflow-hidden border border-white/10'
          }`}
          layout
          transition={{ duration: 0.4, type: 'spring', bounce: 0.2 }}
        >
          <div className={`relative ${
            isFullscreen ? 'w-screen h-screen' : 'aspect-video'
          }`}>
            <video
              ref={webcamRef}
              className="absolute top-0 left-0 w-full h-full object-cover"
              style={{ 
                transform: 'scaleX(-1)',
                borderRadius: isFullscreen ? '0' : undefined 
              }}
              autoPlay 
              muted 
              playsInline
            />
            
            <canvas
              ref={canvasRef}
              className="absolute top-0 left-0 w-full h-full"
              style={{ 
                transform: 'scaleX(-1)',
                borderRadius: isFullscreen ? '0' : undefined 
              }}
            />
            
            {/* All UI controls positioned consistently */}
            <ConnectionStatus status={connectionStatus} />
            <FullscreenButton isFullscreen={isFullscreen} toggleFullscreen={toggleFullscreen} />
            
            {/* WorkoutSelector always in same position */}
            <div className="absolute top-4 left-16 z-40">
              <WorkoutSelector 
                selectedWorkout={currentWorkout} 
                onSelectWorkout={handleWorkoutChange} 
              />
            </div>
            
            <MuscleGroupVisualizer 
              isVisible={showMuscleVisualizer} 
              muscleGroup={predictedMuscleGroup} 
            />
            
            <MuscleVisualizerToggle 
              isVisible={showMuscleVisualizer} 
              toggleVisibility={toggleMuscleVisualizer} 
            />
            
            {/* InfoPanel moved inside video container with consistent positioning */}
            <div className="absolute bottom-4 left-0 right-0 mx-auto max-w-xl px-4">
              <InfoPanel 
                currentWorkout={currentWorkout}
                predictedWorkout={predictedWorkout}
                predictionConfidence={predictionConfidence}
                predictionThreshold={SUGGESTION_THRESHOLD}
                predictedMuscleGroup={predictedMuscleGroup}
                muscleGroupConfidence={muscleGroupConfidence}
                feedbackLatency={feedbackLatency}
                receivedCount={receivedCount}
                connectionStatus={connectionStatus}
              />
            </div>
          </div>
        </motion.div>
      </motion.main>
    </motion.section>
  );
};

export default TrainingPage;