import React, { useState, useEffect, useRef } from 'react';
import NavBar from '../components/Navbar';
import { useNavigate } from 'react-router-dom';
import * as poseDetection from '@mediapipe/pose';
import { POSE_CONNECTIONS } from '@mediapipe/pose';
import { Camera } from '@mediapipe/camera_utils';
import io from 'socket.io-client';

// Utility functions
const getJointName = (index) => {
  switch(index) {
    case 11: return "Left Shoulder";
    case 12: return "Right Shoulder";
    case 13: return "Left Elbow";
    case 14: return "Right Elbow";
    case 15: return "Left Wrist";
    case 16: return "Right Wrist";
    case 23: return "Left Hip";
    case 24: return "Right Hip";
    case 25: return "Left Knee";
    case 26: return "Right Knee";
    case 27: return "Left Ankle";
    case 28: return "Right Ankle";
    default: return "";
  }
};

const getArrowColor = (distance) => {
  if (distance < 15) {
    return '#eab308'; // yellow
  } else if (distance < 30) {
    return '#f97316'; // orange
  } else {
    return '#ef4444'; // red
  }
};

const drawArrow = (ctx, fromX, fromY, toX, toY, color, lineWidth) => {
  const headLength = 15;
  const dx = toX - fromX;
  const dy = toY - fromY;
  const angle = Math.atan2(dy, dx);
  
  // Draw line
  ctx.beginPath();
  ctx.moveTo(fromX, fromY);
  ctx.lineTo(toX, toY);
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.stroke();
  
  // Draw arrowhead
  ctx.beginPath();
  ctx.moveTo(toX, toY);
  ctx.lineTo(toX - headLength * Math.cos(angle - Math.PI/6), toY - headLength * Math.sin(angle - Math.PI/6));
  ctx.lineTo(toX - headLength * Math.cos(angle + Math.PI/6), toY - headLength * Math.sin(angle + Math.PI/6));
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
};

// Drawing functions
const drawUserPose = (ctx, landmarks, canvasWidth, canvasHeight) => {
  // Draw circles for body landmarks only
  for (let i = 11; i < landmarks.length; i++) {
    const x = landmarks[i].x * canvasWidth;
    const y = landmarks[i].y * canvasHeight;

    ctx.beginPath();
    ctx.arc(x, y, 5, 0, 2 * Math.PI);
    ctx.fillStyle = '#8b5cf6'; // purple
    ctx.fill();
  }

  // Draw lines between body-only connections
  POSE_CONNECTIONS.forEach(([i, j]) => {
    if (i >= 11 && j >= 11) {
      const p1 = landmarks[i];
      const p2 = landmarks[j];

      ctx.beginPath();
      ctx.moveTo(p1.x * canvasWidth, p1.y * canvasHeight);
      ctx.lineTo(p2.x * canvasWidth, p2.y * canvasHeight);
      ctx.strokeStyle = '#a78bfa'; // lighter purple
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  });
};

const findJointsToCorrect = (landmarks, corrections) => {
  const correctJoints = [];
  
  // Only check joints that have correction data
  Object.keys(corrections).forEach(indexStr => {
    const i = parseInt(indexStr);
    const correction = corrections[indexStr];
    
    // If correction is significant enough (threshold can be adjusted)
    if (Math.abs(correction.x) > 0.01 || Math.abs(correction.y) > 0.01) {
      const jointName = getJointName(i);
      if (jointName && i < landmarks.length) {
        correctJoints.push({
          name: jointName,
          index: i,
          correction: correction
        });
      }
    }
  });
  
  return correctJoints;
};

const drawCorrectionArrows = (ctx, landmarks, corrections, jointsToCorrect, canvasWidth, canvasHeight) => {
  jointsToCorrect.forEach(joint => {
    const i = joint.index;
    const originalX = landmarks[i].x * canvasWidth;
    const originalY = landmarks[i].y * canvasHeight;
    
    // Get correction from the model output
    const correction = joint.correction;
    
    // Calculate target point (current position + correction)
    const targetX = (landmarks[i].x + correction.x) * canvasWidth;
    const targetY = (landmarks[i].y + correction.y) * canvasHeight;
    
    // Calculate extended target point (50% longer for better visibility)
    const vectorX = targetX - originalX;
    const vectorY = targetY - originalY;
    const extendedTargetX = originalX + vectorX * 1.5;
    const extendedTargetY = originalY + vectorY * 1.5;
    
    // Calculate distance between current and target positions
    const distance = Math.sqrt(vectorX * vectorX + vectorY * vectorY);
    
    // Determine color based on distance
    const arrowColor = getArrowColor(distance);
    
    // Draw arrow
    drawArrow(ctx, originalX, originalY, extendedTargetX, extendedTargetY, arrowColor, 6);
  });
};

// OutOfFrameWarning component
const OutOfFrameWarning = () => (
  <div className="absolute inset-0 bg-black/60 flex items-center justify-center z-50">
    <div className="bg-white text-red-600 text-lg md:text-xl font-semibold px-6 py-3 rounded-lg shadow-lg animate-bounce">
      Get in Frame
    </div>
  </div>
);

const TrainingPage = () => {
  const navigate = useNavigate();
  const [isDarkMode, setIsDarkMode] = useState(false);
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [outOfFrame, setOutOfFrame] = useState(false);
  const [userLandmarks, setUserLandmarks] = useState(null);
  const [corrections, setCorrections] = useState({});
  const socketRef = useRef(null);
  
  // Socket connection setup
  useEffect(() => {
    // Create socket connection
    socketRef.current = io('http://localhost:8001');
    
    socketRef.current.on('connect', () => {
      console.log('Connected to WebSocket server');
    });
    
    socketRef.current.on('connected', (data) => {
      console.log('Server connection confirmed:', data);
    });
    
    socketRef.current.on('pose_corrections', (data) => {
      console.log('Received corrections:', data);
      setCorrections(data);
    });
    
    socketRef.current.on('error', (data) => {
      console.error('WebSocket error:', data.message);
    });
    
    // Clean up on unmount
    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
  }, []);
  
  // Send landmarks to server when they update
  useEffect(() => {
    if (userLandmarks && socketRef.current && socketRef.current.connected) {
      // Throttle updates to reduce network traffic (every 100ms)
      const throttleDelay = 100; // ms
      
      const timerId = setTimeout(() => {
        socketRef.current.emit('pose_data', {
          landmarks: userLandmarks
        });
      }, throttleDelay);
      
      return () => clearTimeout(timerId);
    }
  }, [userLandmarks]);

  // Dark mode setup
  useEffect(() => {
    const prefersDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
    setIsDarkMode(prefersDarkMode);
  }, [navigate]);

  useEffect(() => {
    document.body.classList.toggle('dark', isDarkMode);
  }, [isDarkMode]);

  // MediaPipe Pose setup
  useEffect(() => {
    if (typeof window === 'undefined' || !webcamRef.current) return;

    const pose = new poseDetection.Pose({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
    });

    pose.setOptions({
      modelComplexity: 2,
      smoothLandmarks: false,
      enableSegmentation: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    pose.onResults(onResults);

    const camera = new Camera(webcamRef.current, {
      onFrame: async () => {
        await pose.send({ image: webcamRef.current });
      },
      width: 640,
      height: 480
    });
    
    camera.start();

    function onResults(results) {
      const canvas = canvasRef.current;
      if (!canvas || !results.poseLandmarks) {
        setOutOfFrame(true);
        return;
      }
      
      const ctx = canvas.getContext('2d');
      const landmarks = results.poseLandmarks;
      
      // Check if user is in frame
      const visiblePoints = landmarks.filter(
        (landmark) => landmark.visibility > 0.6
      );

      if (visiblePoints.length < 12) {
        setOutOfFrame(true);
      } else {
        setOutOfFrame(false);
      }

      // Save user landmarks
      setUserLandmarks(landmarks);

      // Set canvas dimensions
      canvas.width = webcamRef.current.videoWidth;
      canvas.height = webcamRef.current.videoHeight;

      // Clear canvas
      ctx.save();
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw user pose
      drawUserPose(ctx, landmarks, canvas.width, canvas.height);
      
      // Find joints that need correction based on model output
      const jointsToCorrect = findJointsToCorrect(landmarks, corrections);
      
      // Draw correction arrows
      drawCorrectionArrows(ctx, landmarks, corrections, jointsToCorrect, canvas.width, canvas.height);

      ctx.restore();
    }

    return () => {
      camera.stop();
    };
  }, [corrections]); // Add corrections as dependency to redraw when new data arrives

  return (
    <section className={`overflow-scroll fixed inset-0 ${isDarkMode ? 'bg-gradient-to-br from-gray-800 to-indigo-500' : 'bg-gradient-to-br from-gray-100 to-indigo-500'}`}>
      <NavBar isDarkMode={isDarkMode} />
      <main>
        <div className="max-w-4xl mx-auto mt-2 bg-white dark:bg-gray-900 rounded-lg shadow-md overflow-hidden p-4">
          <div className="relative w-full aspect-video">
            {outOfFrame && <OutOfFrameWarning />}
            <video 
              ref={webcamRef} 
              className="absolute top-0 left-0 w-full h-full object-cover" 
              style={{ transform: 'scaleX(-1)' }}
              autoPlay 
              muted 
              playsInline 
            />
            <canvas 
              ref={canvasRef} 
              className="absolute top-0 left-0 w-full h-full" 
              style={{ transform: 'scaleX(-1)' }}
            />
          </div>
          <div className="mt-4 text-center text-sm text-gray-700 dark:text-gray-300">
            <p>Workout Type: Barbell Bicep Curl</p>
          </div>
        </div>
      </main>
    </section>
  );
};

export default TrainingPage;