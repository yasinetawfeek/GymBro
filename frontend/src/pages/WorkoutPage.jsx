import React, { useState, useEffect, useRef } from 'react';
import NavBar from '../components/Navbar';
import { useNavigate } from 'react-router-dom';
import * as poseDetection from '@mediapipe/pose';
import { POSE_CONNECTIONS } from '@mediapipe/pose';
import { Camera } from '@mediapipe/camera_utils';

// Utility functions
const createRandomSeed = () => {
  return Array(33).fill().map(() => ({
    x: Math.random() * 0.05 - 0.025,
    y: Math.random() * 0.05 - 0.025
  }));
};

const jointsToHide = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21, 22, 29, 30, 31, 32]; // Example: head and face joints

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
  for (let i = 0; i < landmarks.length; i++) {
    if (!jointsToHide.includes(i)) {
    const x = landmarks[i].x * canvasWidth;
    const y = landmarks[i].y * canvasHeight;

    ctx.beginPath();
    ctx.arc(x, y, 5, 0, 2 * Math.PI);
    ctx.fillStyle = '#8b5cf6'; // purple
    ctx.fill();
  }}

  // Draw lines between body-only connections
  POSE_CONNECTIONS.forEach(([i, j]) => {
    if (!jointsToHide.includes(i) && !jointsToHide.includes(j))  {
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

const findJointsToCorrect = (landmarks, randomSeed) => {
  const correctJoints = [];
  
  for (let i = 0; i < landmarks.length; i++) {
    if (!jointsToHide.includes(i)) {
    const random = randomSeed[i];
    // If the deviation is significant, mark it for correction
    if (Math.abs(random.x) > 0.02 || Math.abs(random.y) > 0.02) {
      const jointName = getJointName(i);
      if (jointName) {
        correctJoints.push({
          name: jointName,
          index: i
        });
      }
    }
  }}
  
  return correctJoints;
};

const drawCorrectionArrows = (ctx, landmarks, randomSeed, jointsToCorrect, canvasWidth, canvasHeight) => {
  jointsToCorrect.forEach(joint => {
    const i = joint.index;
    const originalX = landmarks[i].x * canvasWidth;
    const originalY = landmarks[i].y * canvasHeight;
    
    // Get correction offset from correction data
    const offsetX = randomSeed[i].x;
    const offsetY = randomSeed[i].y;
    
    const targetX = (landmarks[i].x + offsetX) * canvasWidth;
    const targetY = (landmarks[i].y + offsetY) * canvasHeight;
    
    // Calculate extended target point (50% longer)
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
  
  // Random seed for consistent random movements within a frame
  const randomSeedRef = useRef(createRandomSeed());
  
  // Update random seed every 3 seconds for more visible, longer-lasting movements
  useEffect(() => {
    const intervalId = setInterval(() => {
      // Choose several random joints to modify each time (not all)
      const newRandomValues = [...randomSeedRef.current];
      
      // Select 3-5 random joints to modify
      const numJointsToModify = Math.floor(Math.random() * 3) + 3; 
      for (let i = 0; i < numJointsToModify; i++) {
        // Select a joint index (focusing on body parts, not face)
        const jointIndex = Math.floor(Math.random() * 22) + 11; 
        // Give it a new random offset
        newRandomValues[jointIndex] = {
          x: Math.random() * 0.07 - 0.035, // Larger values: -0.035 to 0.035
          y: Math.random() * 0.07 - 0.035
        };
      }
      
      randomSeedRef.current = newRandomValues;
    }, 3000); // Update every 3 seconds
    
    return () => clearInterval(intervalId);
  }, []);

  // Dark mode setup
  useEffect(() => {
    const prefersDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
    setIsDarkMode(prefersDarkMode);
  }, [navigate]);

  useEffect(() => {
    document.body.classList.toggle('dark', isDarkMode);
  }, [isDarkMode]);

  const toggleDarkMode = () => setIsDarkMode(!isDarkMode);

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
      const visiblePoints = results.poseLandmarks
        .map((landmark, index) => ({ ...landmark, index }))
        .filter(
          (landmark) =>
            landmark.visibility > 0.6 && !jointsToHide.includes(landmark.index)
        );

      // You can tweak this threshold (e.g., at least 12 visible points)
      if (visiblePoints.length < 1) { // I Set it to 1 for now, just to test
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
      
      // Find joints that need correction
      const jointsToCorrect = findJointsToCorrect(landmarks, randomSeedRef.current);
      
      // Draw correction arrows
      drawCorrectionArrows(ctx, landmarks, randomSeedRef.current, jointsToCorrect, canvas.width, canvas.height);

      ctx.restore();
    }

    return () => {
      camera.stop();
    };
  }, []);

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
        </div>
      </main>
    </section>
  );
};

export default TrainingPage;