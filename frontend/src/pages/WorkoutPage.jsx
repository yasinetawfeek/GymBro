import React, { useState, useEffect, useRef } from 'react';
// Assuming NavBar is correctly located at '../components/Navbar'
// If not, adjust the import path.
// import NavBar from '../components/Navbar';
import { useNavigate } from 'react-router-dom';
import * as poseDetection from '@mediapipe/pose';
import { POSE_CONNECTIONS } from '@mediapipe/pose';
import { Camera } from '@mediapipe/camera_utils';
import io from 'socket.io-client';

// --- Utility functions ---
const getJointName = (index) => {
  // Returns the name of a specific joint index
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
  // Determines arrow color based on correction distance
  // Note: The distance calculation happens *inside* drawCorrectionArrows
  // This function just maps a distance value to a color.
  // Assuming distance is related to the magnitude of correction vector for now.
  if (distance < 0.05) return '#eab308'; // yellow for small corrections (Adjust threshold as needed)
  else if (distance < 0.1) return '#f97316'; // orange for medium corrections (Adjust threshold)
  else return '#ef4444'; // red for large corrections (Adjust threshold)
};


const drawArrow = (ctx, fromX, fromY, toX, toY, color, lineWidth) => {
  // Draws an arrow from (fromX, fromY) to (toX, toY)
  const headLength = 15; // Length of the arrowhead lines
  const dx = toX - fromX;
  const dy = toY - fromY;
  const angle = Math.atan2(dy, dx); // Angle of the arrow

  // Draw the main line
  ctx.beginPath();
  ctx.moveTo(fromX, fromY);
  ctx.lineTo(toX, toY);
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.stroke();

  // Draw the arrowhead
  ctx.beginPath();
  ctx.moveTo(toX, toY);
  // Calculate points for the arrowhead wings
  ctx.lineTo(toX - headLength * Math.cos(angle - Math.PI/6), toY - headLength * Math.sin(angle - Math.PI/6));
  ctx.lineTo(toX - headLength * Math.cos(angle + Math.PI/6), toY - headLength * Math.sin(angle + Math.PI/6));
  ctx.closePath(); // Close the path to form a triangle
  ctx.fillStyle = color; // Use the same color for fill
  ctx.fill();
};

// --- Drawing functions ---
const drawUserPose = (ctx, landmarks, canvasWidth, canvasHeight) => {
  // Draws the user's pose skeleton (purple dots and lines)
  if (!landmarks) return; // Don't draw if no landmarks

  // Draw circles for body landmarks (indices 11 and up)
  for (let i = 11; i < landmarks.length; i++) {
    const landmark = landmarks[i];
    if (!landmark || landmark.x == null || landmark.y == null) continue; // Skip if landmark is invalid
    const x = landmark.x * canvasWidth;
    const y = landmark.y * canvasHeight;
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, 2 * Math.PI); // Draw a circle
    ctx.fillStyle = '#8b5cf6'; // Purple color
    ctx.fill();
  }

  // Draw lines connecting the body joints
  POSE_CONNECTIONS.forEach(([i, j]) => {
    // Only draw connections between body parts (index >= 11)
    if (i >= 11 && j >= 11 && i < landmarks.length && j < landmarks.length) {
      const p1 = landmarks[i];
      const p2 = landmarks[j];
      // Ensure both points exist and have coordinates
      if (!p1 || !p2 || p1.x == null || p1.y == null || p2.x == null || p2.y == null) return;
      ctx.beginPath();
      ctx.moveTo(p1.x * canvasWidth, p1.y * canvasHeight);
      ctx.lineTo(p2.x * canvasWidth, p2.y * canvasHeight);
      ctx.strokeStyle = '#a78bfa'; // Lighter purple for lines
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  });
};

const findJointsToCorrect = (landmarks, corrections) => {
  // Identifies which joints have significant correction data
  if (!landmarks || !corrections || Object.keys(corrections).length === 0) {
      // console.log("[findJointsToCorrect] No landmarks or corrections data."); // Noisy
      return []; // Need both landmarks and corrections
  }
  const correctJoints = [];
  let foundSignificant = false; // Flag to log if any significant correction is found
  Object.keys(corrections).forEach(indexStr => {
    const i = parseInt(indexStr);
    const correction = corrections[indexStr];
    // Check if correction exists, is significant, and the corresponding landmark exists
    // **Important**: Backend sends x, y, z. Check x and y for 2D significance.
    if (correction && correction.x != null && correction.y != null &&
        (Math.abs(correction.x) > 0.01 || Math.abs(correction.y) > 0.01) &&
        i < landmarks.length && landmarks[i])
    {
      const jointName = getJointName(i);
      if (jointName) { // Ensure it's a joint we are tracking
        correctJoints.push({ name: jointName, index: i, correction: correction });
        foundSignificant = true;
      }
    }
  });
  // if (foundSignificant) {
  //   console.log("[findJointsToCorrect] Found joints needing correction:", correctJoints.map(j => j.name));
  // } else {
  //   console.log("[findJointsToCorrect] No significant corrections found in data:", corrections);
  // }
  return correctJoints;
};

const drawCorrectionArrows = (ctx, landmarks, corrections, jointsToCorrect, canvasWidth, canvasHeight) => {
  // Draws arrows indicating the direction and magnitude of corrections
  if (!landmarks || !corrections || jointsToCorrect.length === 0) {
      console.log("[drawCorrectionArrows] Skipping: No landmarks, corrections, or joints to correct."); // Noisy
      return; // Need landmarks, corrections, and joints identified for correction
  }

  console.log(`[drawCorrectionArrows] Attempting to draw ${jointsToCorrect.length} arrows.`); // Log how many arrows should be drawn

  jointsToCorrect.forEach(joint => {
    const i = joint.index;
    // Ensure the specific landmark for the arrow exists and has coordinates
    if (!landmarks[i] || landmarks[i].x == null || landmarks[i].y == null) {
        console.warn(`[drawCorrectionArrows] Skipping arrow for joint ${joint.name} (index ${i}): Landmark invalid.`);
        return;
    }

    const originalX = landmarks[i].x * canvasWidth;
    const originalY = landmarks[i].y * canvasHeight;
    const correction = joint.correction;

    // Ensure correction coordinates exist (using x and y for 2D arrows)
    if (correction.x == null || correction.y == null) {
         console.warn(`[drawCorrectionArrows] Skipping arrow for joint ${joint.name} (index ${i}): Correction data invalid.`);
         return;
    }

    // Calculate target point based on correction (using only x and y for 2D)
    // The correction values represent the *displacement* vector
    const vectorX = correction.x * canvasWidth; // Scale correction vector by canvas size
    const vectorY = correction.y * canvasHeight;

    // Target point is original + displacement vector
    const targetX = originalX + vectorX;
    const targetY = originalY + vectorY;

    // Calculate extended target point for better visibility
    const extendedTargetX = originalX + vectorX * 1.5; // Make arrow 50% longer
    const extendedTargetY = originalY + vectorY * 1.5;

    // Calculate magnitude of the *original* correction vector (before scaling) for color coding
    const correctionMagnitude = Math.sqrt(correction.x * correction.x + correction.y * correction.y);
    const arrowColor = getArrowColor(correctionMagnitude); // Get color based on magnitude

    console.log(`[drawCorrectionArrows] Drawing arrow for ${joint.name}: Mag=${correctionMagnitude.toFixed(3)}, Color=${arrowColor}`); // Log details

    // Draw the arrow on the canvas
    drawArrow(ctx, originalX, originalY, extendedTargetX, extendedTargetY, arrowColor, 6); // Line width 6
  });
};


// --- UI Components ---
const OutOfFrameWarning = () => (
  // Simple component to display when the user is out of frame
  <div className="absolute inset-0 bg-black/60 flex items-center justify-center z-50 pointer-events-none">
    <div className="bg-white text-red-600 text-lg md:text-xl font-semibold px-6 py-3 rounded-lg shadow-lg animate-bounce">
      Get in Frame
    </div>
  </div>
);

const ConnectionStatus = ({ status }) => {
  // Displays the WebSocket connection status indicator
  const statusColors = { connected: "bg-green-500", connecting: "bg-yellow-500", disconnected: "bg-red-500" };
  return (
    <div className="absolute top-4 right-4 flex items-center space-x-2 z-40">
      <div className={`w-4 h-4 rounded-full ${statusColors[status]}`}></div>
      <span className="text-xs font-medium text-white bg-black/30 px-2 py-1 rounded">
        {status === "connected" ? "Connected" : status === "connecting" ? "Connecting..." : "Disconnected"}
      </span>
    </div>
  );
};

// --- Main TrainingPage Component ---
const TrainingPage = () => {
  const navigate = useNavigate(); // For potential navigation later
  const [isDarkMode, setIsDarkMode] = useState(false); // Dark mode state
  const webcamRef = useRef(null); // Ref for the video element
  const canvasRef = useRef(null); // Ref for the canvas element
  const [outOfFrame, setOutOfFrame] = useState(false); // State for out-of-frame warning
  // State for landmarks - primarily used to trigger re-renders if other UI depends on it
  const [userLandmarksForDrawing, setUserLandmarksForDrawing] = useState(null);
  // Ref to store the *absolute latest* landmarks for the sending interval
  const latestLandmarksRef = useRef(null);
  const [corrections, setCorrections] = useState({}); // State for corrections data from backend
  const socketRef = useRef(null); // Ref for the WebSocket instance
  const [connectionStatus, setConnectionStatus] = useState("connecting"); // WebSocket connection status
  const lastLandmarkUpdateRef = useRef(0); // Ref to track time for latency calculation
  const lastCorrectionTimeRef = useRef(0); // Ref to track time for correction timeout check
  const [feedbackLatency, setFeedbackLatency] = useState(0); // State to display feedback latency
  const [receivedCount, setReceivedCount] = useState(0); // State to count received corrections
  const poseInstanceRef = useRef(null); // Ref for MediaPipe Pose instance
  const cameraInstanceRef = useRef(null); // Ref for MediaPipe Camera instance
  const sendIntervalRef = useRef(null); // Ref for the data sending interval ID

  // --- WebSocket Connection Setup (Unchanged) ---
  useEffect(() => {
    const connectSocket = () => {
      if (socketRef.current) {
        console.log('[Socket Cleanup] Disconnecting existing socket...');
        socketRef.current.disconnect();
      }
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
      socketRef.current.on('connected', (data) => { console.log('[Socket Event] Received "connected" confirmation:', data); });
      socketRef.current.on('pose_corrections', (data) => {
        const now = Date.now();
        const latency = lastLandmarkUpdateRef.current ? now - lastLandmarkUpdateRef.current : 0;
        setFeedbackLatency(latency);
        setReceivedCount(prevCount => prevCount + 1);
        // console.log(`[Socket Event] Received pose_corrections (#${receivedCount + 1}). Latency: ${latency}ms`, data); // Log received data
        setCorrections(data);
        lastCorrectionTimeRef.current = now;
      });
      socketRef.current.on('error', (data) => { console.error('[Socket Event] Received server error:', data.message || data); });
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

  // --- Landmark Update Function (Unchanged) ---
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

  // --- Send Landmarks Interval (Unchanged) ---
  useEffect(() => {
    const sendIntervalDelay = 50;
    if (sendIntervalRef.current) { clearInterval(sendIntervalRef.current); }
    console.log(`[Data Send] Setting up interval to send data every ${sendIntervalDelay}ms`);
    sendIntervalRef.current = setInterval(() => {
      const landmarksToSend = latestLandmarksRef.current;
      const socketIsConnected = socketRef.current?.connected;
      if (landmarksToSend && socketIsConnected) {
        const sendTimestamp = Date.now();
        // console.log(`[Data Send] Interval fired. Emitting pose_data at ${new Date(sendTimestamp).toLocaleTimeString()}`);
        socketRef.current.emit('pose_data', { landmarks: landmarksToSend, timestamp: sendTimestamp });
      }
    }, sendIntervalDelay);
    return () => {
      if (sendIntervalRef.current) {
        console.log("[Data Send] Clearing send interval.");
        clearInterval(sendIntervalRef.current);
        sendIntervalRef.current = null;
      }
    };
  }, []);

  // --- Correction Timeout Check (Unchanged) ---
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

  // --- Dark Mode Setup (Unchanged) ---
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
            const visiblePoints = landmarks.filter(lm => lm && lm.visibility && lm.visibility > 0.6);
            if (visiblePoints.length < 12) {
                if (!outOfFrame) setOutOfFrame(true);
            } else {
                if (outOfFrame) setOutOfFrame(false);
            }
            updateLandmarks(landmarks); // Update ref and state

            // --- Drawing Logic ---
            drawUserPose(ctx, landmarks, canvas.width, canvas.height); // Draw skeleton

            // --- ADDED LOGS FOR ARROW DEBUGGING ---
            console.log("[onResults] Current corrections state:", corrections); // Log corrections state
            const jointsToCorrect = findJointsToCorrect(landmarks, corrections);
            console.log("[onResults] Joints found to correct:", jointsToCorrect); // Log identified joints

            // Only attempt to draw if there are joints needing correction
            if (jointsToCorrect.length > 0) {
                 drawCorrectionArrows(ctx, landmarks, corrections, jointsToCorrect, canvas.width, canvas.height);
            }
            // --- END ADDED LOGS ---

        } else {
            if (!outOfFrame) setOutOfFrame(true);
            updateLandmarks(null);
        }
        ctx.restore();
    }


    const startMediaPipe = async () => {
        if (typeof window === 'undefined' || !webcamRef.current) {
            console.log("[MediaPipe Setup] Aborted: window or webcamRef not ready."); return;
        }
        console.log("[MediaPipe Setup] Initializing Pose...");
        const pose = new poseDetection.Pose({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose@${poseDetection.VERSION}/${file}`
        });
        poseInstanceRef.current = pose;
        pose.setOptions({ modelComplexity: 1, smoothLandmarks: true, enableSegmentation: false, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
        pose.onResults(onResults);

        initTimeoutId = setTimeout(async () => {
            try {
                console.log("[MediaPipe Setup] Attempting pose.initialize() after delay...");
                await pose.initialize();
                console.log("[MediaPipe Setup] Pose initialized successfully after delay.");
                if (!webcamRef.current) { console.warn("[MediaPipe Setup] Webcam ref null before camera setup."); return; }
                console.log("[MediaPipe Setup] Setting up Camera...");
                const camera = new Camera(webcamRef.current, {
                    onFrame: async () => {
                        if (poseInstanceRef.current && webcamRef.current) {
                            try { await poseInstanceRef.current.send({ image: webcamRef.current }); }
                            catch (sendError) { console.error("[MediaPipe onFrame] Error sending frame:", sendError); }
                        }
                    },
                    width: 640, height: 480
                });
                cameraInstanceRef.current = camera;
                await camera.start();
                console.log("[MediaPipe Setup] Camera started successfully.");
            } catch (initError) {
                console.error("[MediaPipe Setup] Error initializing MediaPipe Pose (after delay):", initError);
                if (poseInstanceRef.current) { poseInstanceRef.current.close(); poseInstanceRef.current = null; }
                if (cameraInstanceRef.current) { cameraInstanceRef.current = null; }
            }
        }, 100);
    };

    startMediaPipe();

    return () => {
        console.log("[MediaPipe Cleanup] Cleaning up...");
        if (initTimeoutId) { clearTimeout(initTimeoutId); }
        if (cameraInstanceRef.current) { cameraInstanceRef.current.stop(); cameraInstanceRef.current = null; }
        if (poseInstanceRef.current) { poseInstanceRef.current.close(); poseInstanceRef.current = null; }
    };
  }, []); // Empty dependency array

  // --- Render ---
  return (
    <section className={`overflow-hidden fixed inset-0 ${isDarkMode ? 'bg-gradient-to-br from-gray-800 to-indigo-500' : 'bg-gradient-to-br from-gray-100 to-indigo-500'}`}>
      {/* <NavBar isDarkMode={isDarkMode} /> */}
      <main>
        <div className="max-w-4xl mx-auto mt-2 bg-white dark:bg-gray-900 rounded-lg shadow-md overflow-hidden p-4">
          <div className="relative w-full aspect-video">
            {outOfFrame && <OutOfFrameWarning />}
            <ConnectionStatus status={connectionStatus} />
            <video
              ref={webcamRef}
              className="absolute top-0 left-0 w-full h-full object-cover rounded-lg"
              style={{ transform: 'scaleX(-1)' }}
              autoPlay muted playsInline
              onLoadedData={() => console.log("[Video Event] Video metadata loaded.")}
              onError={(e) => console.error("[Video Event] Video error:", e)}
            />
            <canvas
              ref={canvasRef}
              className="absolute top-0 left-0 w-full h-full rounded-lg"
              style={{ transform: 'scaleX(-1)' }}
            />
          </div>
          <div className="mt-4 text-center text-sm text-gray-700 dark:text-gray-300">
            <p>Workout Type: Barbell Bicep Curl</p>
            {(feedbackLatency > 0 || receivedCount > 0) && (
              <p className="mt-1 text-xs">
                Latency: {feedbackLatency > 0 ? `${feedbackLatency}ms` : 'N/A'} | Corrections: {receivedCount}
              </p>
            )}
             <p className="mt-1 text-xs">Socket: {connectionStatus}</p>
          </div>
        </div>
      </main>
    </section>
  );
};

export default TrainingPage;
