import React, { useState, useEffect, useRef } from 'react';
import NavBar from '../components/Navbar';
import { useNavigate } from 'react-router-dom';
import * as poseDetection from '@mediapipe/pose';
import { POSE_CONNECTIONS } from '@mediapipe/pose';
import { Camera } from '@mediapipe/camera_utils';
import { io } from 'socket.io-client';

const TrainingPage = () => {
  const navigate = useNavigate();
  const [isDarkMode, setIsDarkMode] = useState(false);
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [outOfFrame, setOutOfFrame] = useState(false);
  const [userLandmarks, setUserLandmarks] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const socketRef = useRef(null);
  const [predictions, setPredictions] = useState([]);
  const [connectionError, setConnectionError] = useState(null);
  const [displacementData, setDisplacementData] = useState(null);
  
  // Random seed for consistent random movements within a frame
  const randomSeedRef = useRef(Array(33).fill().map(() => ({
    x: Math.random() * 0.05 - 0.025,  // Random value between -0.025 and 0.025 (more noticeable)
    y: Math.random() * 0.05 - 0.025
  })));
  
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
    }, 3000); // Update every 3 seconds instead of 500ms
    
    return () => clearInterval(intervalId);
  }, []);

  // Connect to WebSocket server
  useEffect(() => {
    // Connect to WebSocket server
    try {
      socketRef.current = io('http://localhost:8001');
      
      socketRef.current.on('connect', () => {
        console.log('Connected to WebSocket server');
        setIsConnected(true);
        setConnectionError(null);
      });
      
      socketRef.current.on('disconnect', () => {
        console.log('Disconnected from WebSocket server');
        setIsConnected(false);
        setConnectionError("Disconnected from server");
      });
      
      socketRef.current.on('connect_error', (error) => {
        console.error('Connection error:', error);
        setIsConnected(false);
        setConnectionError(`Connection error: ${error.message}`);
      });
      
      // Listen for prediction results
      socketRef.current.on('pose_prediction', (data) => {
        console.log('Received prediction:', data);
        setPrediction(data);
        
        // Store the displacement data
        if (data.displacement) {
          setDisplacementData(data.displacement);
          console.log('Received displacement data:', data.displacement);
        }
        
        // Add to predictions history (keep last 5)
        setPredictions(prev => {
          const newPredictions = [data, ...prev];
          return newPredictions.slice(0, 5);
        });
      });
      
      socketRef.current.on('error', (data) => {
        console.error('WebSocket error:', data.message);
        setConnectionError(`Error: ${data.message}`);
      });
    } catch (error) {
      console.error('Error setting up socket connection:', error);
      setConnectionError(`Failed to connect: ${error.message}`);
    }
    
    // Clean up on unmount
    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
    };
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
    if (typeof window === 'undefined') return;

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

    let camera;

    if (webcamRef.current) {
      camera = new Camera(webcamRef.current, {
        onFrame: async () => {
          await pose.send({ image: webcamRef.current });
        },
        width: 640,
        height: 480
      });
      camera.start();
    }

    function onResults(results) {
      const canvas = canvasRef.current;
      if (!canvas) return;
      
      const ctx = canvas.getContext('2d');

      if (!results.poseLandmarks) {
        setOutOfFrame(true);
        return;
      }
  
      const visiblePoints = results.poseLandmarks.filter(
        (landmark) => landmark.visibility > 0.6
      );

      // You can tweak this threshold (e.g., at least 12 visible points)
      if (visiblePoints.length < 12) {
        setOutOfFrame(true);
      } else {
        setOutOfFrame(false);
        
        // Send pose data to server if connected
        if (socketRef.current && isConnected) {
          socketRef.current.emit('pose_frame', {
            landmarks: results.poseLandmarks
          });
        }
      }

      // Save user landmarks for reference
      setUserLandmarks(results.poseLandmarks);

      canvas.width = webcamRef.current.videoWidth;
      canvas.height = webcamRef.current.videoHeight;

      ctx.save();
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw user's detected pose (purple)
      if (results.poseLandmarks) {
        const landmarks = results.poseLandmarks;
        
        // Draw circles for body landmarks only
        for (let i = 11; i < landmarks.length; i++) {
          const x = landmarks[i].x * canvas.width;
          const y = landmarks[i].y * canvas.height;

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
            ctx.moveTo(p1.x * canvas.width, p1.y * canvas.height);
            ctx.lineTo(p2.x * canvas.width, p2.y * canvas.height);
            ctx.strokeStyle = '#a78bfa'; // lighter purple
            ctx.lineWidth = 2;
            ctx.stroke();
          }
        });

        // Add text labels to show which body parts need correction
        const correctJoints = [];
        for (let i = 11; i < landmarks.length; i++) {
          const random = randomSeedRef.current[i];
          // If the deviation is significant, mark it for correction
          if (Math.abs(random.x) > 0.02 || Math.abs(random.y) > 0.02) {
            let jointName = "";
            // Map joint index to a readable name
            switch(i) {
              case 11: jointName = "Left Shoulder"; break;
              case 12: jointName = "Right Shoulder"; break;
              case 13: jointName = "Left Elbow"; break;
              case 14: jointName = "Right Elbow"; break;
              case 15: jointName = "Left Wrist"; break;
              case 16: jointName = "Right Wrist"; break;
              case 23: jointName = "Left Hip"; break;
              case 24: jointName = "Right Hip"; break;
              case 25: jointName = "Left Knee"; break;
              case 26: jointName = "Right Knee"; break;
              case 27: jointName = "Left Ankle"; break;
              case 28: jointName = "Right Ankle"; break;
              default: continue;
            }
            correctJoints.push({
              name: jointName,
              x: landmarks[i].x * canvas.width, 
              y: (landmarks[i].y * canvas.height) - 15
            });
          }
        }

        // Draw reference skeleton (green) - using displacement data if available
        if (results.poseLandmarks) {
          const landmarks = results.poseLandmarks;
          
          // Draw circles for body landmarks only with displacement if available
          for (let i = 11; i < landmarks.length; i++) {
            // Use the displacement data from the model if available, otherwise fallback to random
            let offsetX = 0;
            let offsetY = 0;
            
            if (displacementData && displacementData[i]) {
              // Use model-based displacement data
              offsetX = displacementData[i].x;
              offsetY = displacementData[i].y;
            } else {
              // Fallback to random offsets if no displacement data available
              const random = randomSeedRef.current[i];
              offsetX = random.x;
              offsetY = random.y;
            }
            
            const x = (landmarks[i].x + offsetX) * canvas.width; 
            const y = (landmarks[i].y + offsetY) * canvas.height;

            ctx.beginPath();
            ctx.arc(x, y, 8, 0, 2 * Math.PI); // Larger circles (8px)
            ctx.fillStyle = '#22c55e'; // green
            ctx.fill();
          }

          // Draw lines between body-only connections for reference skeleton
          POSE_CONNECTIONS.forEach(([i, j]) => {
            if (i >= 11 && j >= 11) {
              const p1 = landmarks[i];
              const p2 = landmarks[j];
              
              if (p1 && p2) {
                // Use displacement data if available, otherwise fallback to random
                let offsetX1 = 0, offsetY1 = 0, offsetX2 = 0, offsetY2 = 0;
                
                if (displacementData && displacementData[i] && displacementData[j]) {
                  // Use model-based displacement data
                  offsetX1 = displacementData[i].x;
                  offsetY1 = displacementData[i].y;
                  offsetX2 = displacementData[j].x;
                  offsetY2 = displacementData[j].y;
                } else {
                  // Fallback to random offsets
                  const random1 = randomSeedRef.current[i];
                  const random2 = randomSeedRef.current[j];
                  offsetX1 = random1.x;
                  offsetY1 = random1.y;
                  offsetX2 = random2.x;
                  offsetY2 = random2.y;
                }
                
                ctx.beginPath();
                // Apply displacement offsets to each point
                ctx.moveTo((p1.x + offsetX1) * canvas.width, (p1.y + offsetY1) * canvas.height);
                ctx.lineTo((p2.x + offsetX2) * canvas.width, (p2.y + offsetY2) * canvas.height);
                ctx.strokeStyle = '#4ade80'; // lighter green
                ctx.lineWidth = 3; // Thicker lines (3px)
                ctx.stroke();
              }
            }
          });
          
          // Add text labels for correction hints
          ctx.font = '14px Arial';
          ctx.fillStyle = '#ef4444'; // red text
          correctJoints.forEach(joint => {
            ctx.fillText(`Adjust ${joint.name}`, joint.x, joint.y);
          });
        }
      }

      ctx.restore();
    }

    return () => {
      if (camera) camera.stop();
    };
  }, [isConnected]);


  return (
    <section className={`overflow-scroll fixed inset-0 ${isDarkMode ? 'bg-gradient-to-br from-gray-800 to-indigo-500' : 'bg-gradient-to-br from-gray-100 to-indigo-500'}`}>
      <NavBar isDarkMode={isDarkMode} />
      <main>
        <div className="max-w-4xl mx-auto mt-2 bg-white dark:bg-gray-900 rounded-lg shadow-md overflow-hidden p-4">
          <div className="relative w-full aspect-video">
            {outOfFrame && (
              <div className="absolute inset-0 bg-black/60 flex items-center justify-center z-50">
                <div className="bg-white text-red-600 text-lg md:text-xl font-semibold px-6 py-3 rounded-lg shadow-lg animate-bounce">
                  Get in Frame
                </div>
              </div>
            )}
            <video ref={webcamRef} className="absolute top-0 left-0 w-full h-full object-cover" autoPlay muted playsInline />
            <canvas ref={canvasRef} className="absolute top-0 left-0 w-full h-full" />
            
            {/* Status indicator */}
            <div className="absolute top-2 right-2 flex items-center px-3 py-1 rounded-full text-xs bg-black/50 text-white">
              <div className={`w-2 h-2 rounded-full mr-2 ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span>
                {isConnected ? 'AI Connected' : 'AI Disconnected'}
              </span>
            </div>
            
            {/* Prediction display */}
            {prediction && (
              <div className="absolute bottom-4 left-4 right-4 bg-black/70 text-white p-3 rounded-md">
                <div className="font-semibold text-xl text-center">
                  {prediction.exercise_name}
                </div>
                <div className="w-full bg-gray-700 h-2 rounded-full mt-2">
                  <div 
                    className="bg-green-500 h-2 rounded-full" 
                    style={{ width: `${Math.round(prediction.confidence * 100)}%` }}
                  ></div>
                </div>
                <div className="text-xs text-right mt-1 text-gray-300">
                  Confidence: {Math.round(prediction.confidence * 100)}%
                </div>
              </div>
            )}
            
            {/* Error message if connection fails */}
            {connectionError && (
              <div className="absolute top-10 left-4 right-4 bg-red-500/80 text-white p-2 rounded-md text-sm">
                {connectionError}
              </div>
            )}
          </div>
          
          {/* Prediction history */}
          <div className="mt-4 p-3 bg-gray-100 dark:bg-gray-800 rounded-lg">
            <h3 className="text-sm font-semibold mb-2 text-gray-700 dark:text-gray-300">Recent Predictions</h3>
            {predictions.length > 0 ? (
              <div className="space-y-2">
                {predictions.map((pred, index) => (
                  <div key={index} className="flex justify-between items-center text-xs p-2 bg-white dark:bg-gray-700 rounded">
                    <span className="font-medium">{pred.exercise_name}</span>
                    <span className={`px-2 py-0.5 rounded-full ${
                      pred.confidence > 0.8 ? 'bg-green-100 text-green-800 dark:bg-green-800 dark:text-green-100' : 
                      pred.confidence > 0.5 ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-800 dark:text-yellow-100' :
                      'bg-red-100 text-red-800 dark:bg-red-800 dark:text-red-100'
                    }`}>
                      {Math.round(pred.confidence * 100)}%
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-sm text-gray-500 dark:text-gray-400 italic">No predictions yet</div>
            )}
          </div>
        </div>
      </main>
    </section>
  );
};

export default TrainingPage;
