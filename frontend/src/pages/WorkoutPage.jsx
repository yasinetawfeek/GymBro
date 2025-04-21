import React, { useState, useEffect, useRef } from 'react';
import NavBar from '../components/Navbar';
import { useNavigate } from 'react-router-dom';
import * as poseDetection from '@mediapipe/pose';
import { POSE_CONNECTIONS } from '@mediapipe/pose';
import { Camera } from '@mediapipe/camera_utils';

const TrainingPage = () => {
  const navigate = useNavigate();
  const [isDarkMode, setIsDarkMode] = useState(false);
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  const [outOfFrame, setOutOfFrame] = useState(false);

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
      const ctx = canvas.getContext('2d');

      if (!results.poseLandmarks) {
        setOutOfFrame(true);
        return;
      }
  
      const visiblePoints = results.poseLandmarks.filter(
        (landmark) => landmark.visibility > 0.6
      );
      console.log(visiblePoints.length)

      // You can tweak this threshold (e.g., at least 12 visible points)
      if (visiblePoints.length < 12) {
        setOutOfFrame(true);
      } else {
        setOutOfFrame(false);
      }

      canvas.width = webcamRef.current.videoWidth;
      canvas.height = webcamRef.current.videoHeight;

      ctx.save();
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw pose keypoints (excluding face/head)
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
      }

      ctx.restore();
    }

    return () => {
      if (camera) camera.stop();
    };
  }, []);


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
          </div>
          <div className="mt-4 text-sm text-gray-700 dark:text-gray-300">
            <h3>Workout:</h3>
            <select>
              <option>Auto</option>
              <option>Manual</option>
            </select>
          </div>
        </div>
      </main>
    </section>
  );
};

export default TrainingPage;
