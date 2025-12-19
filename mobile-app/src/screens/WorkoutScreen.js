import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  Dimensions,
  Modal,
  ScrollView,
  ActivityIndicator,
} from 'react-native';
import { Camera } from 'expo-camera';
import { MaterialIcons, FontAwesome5, Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { useAuth } from '../context/AuthContext';
import { COLORS, WORKOUT_MAP, MUSCLE_GROUP_MAP, CAMERA_CONFIG } from '../config';
import io from 'socket.io-client';

const { width, height } = Dimensions.get('window');

const WorkoutScreen = () => {
  const navigation = useNavigation();
  const { user } = useAuth();
  const [hasPermission, setHasPermission] = useState(null);
  const [cameraType, setCameraType] = useState(Camera.Constants.Type.front);
  const [isRecording, setIsRecording] = useState(false);
  const [selectedWorkout, setSelectedWorkout] = useState(0);
  const [predictedWorkout, setPredictedWorkout] = useState(null);
  const [predictionConfidence, setPredictionConfidence] = useState(0);
  const [predictedMuscleGroup, setPredictedMuscleGroup] = useState(null);
  const [muscleGroupConfidence, setMuscleGroupConfidence] = useState(0);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [showWorkoutSelector, setShowWorkoutSelector] = useState(false);
  const [sessionDuration, setSessionDuration] = useState(0);
  const [isDarkMode, setIsDarkMode] = useState(false);
  
  const cameraRef = useRef(null);
  const socketRef = useRef(null);
  const sessionTimerRef = useRef(null);

  useEffect(() => {
    requestCameraPermission();
    setupSocket();
    startSessionTimer();
    
    return () => {
      if (socketRef.current) {
        socketRef.current.disconnect();
      }
      if (sessionTimerRef.current) {
        clearInterval(sessionTimerRef.current);
      }
    };
  }, []);

  const requestCameraPermission = async () => {
    try {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');
      
      if (status !== 'granted') {
        Alert.alert(
          'Camera Permission Required',
          'This app needs camera access to analyze your workout form and provide real-time feedback.',
          [
            { text: 'Cancel', style: 'cancel' },
            { text: 'Settings', onPress: () => navigation.navigate('Settings') }
          ]
        );
      }
    } catch (error) {
      console.error('Error requesting camera permission:', error);
      setHasPermission(false);
    }
  };

  const setupSocket = () => {
    try {
      socketRef.current = io('http://localhost:5000'); // Change to your AI service URL
      
      socketRef.current.on('connect', () => {
        console.log('Connected to AI service');
        setConnectionStatus('connected');
      });

      socketRef.current.on('disconnect', () => {
        console.log('Disconnected from AI service');
        setConnectionStatus('disconnected');
      });

      socketRef.current.on('workout_prediction', (data) => {
        setPredictedWorkout(data.workout_type);
        setPredictionConfidence(data.confidence);
      });

      socketRef.current.on('muscle_group_prediction', (data) => {
        setPredictedMuscleGroup(data.muscle_group);
        setMuscleGroupConfidence(data.confidence);
      });

      socketRef.current.on('form_correction', (data) => {
        showFormCorrection(data.message);
      });

    } catch (error) {
      console.error('Error setting up socket:', error);
      setConnectionStatus('error');
    }
  };

  const startSessionTimer = () => {
    sessionTimerRef.current = setInterval(() => {
      setSessionDuration(prev => prev + 1);
    }, 1000);
  };

  const toggleCameraType = () => {
    setCameraType(
      cameraType === Camera.Constants.Type.front
        ? Camera.Constants.Type.back
        : Camera.Constants.Type.front
    );
  };

  const startRecording = async () => {
    if (!cameraRef.current) return;

    try {
      setIsRecording(true);
      const video = await cameraRef.current.recordAsync({
        quality: CAMERA_CONFIG.quality,
        maxDuration: 300, // 5 minutes max
        mute: false,
      });

      // Send video data to AI service
      if (socketRef.current && socketRef.current.connected) {
        socketRef.current.emit('video_data', {
          video: video.uri,
          workout_type: selectedWorkout,
          user_id: user?.id,
        });
      }

    } catch (error) {
      console.error('Error recording video:', error);
      Alert.alert('Error', 'Failed to start recording. Please try again.');
    } finally {
      setIsRecording(false);
    }
  };

  const stopRecording = async () => {
    if (!cameraRef.current) return;

    try {
      await cameraRef.current.stopRecording();
    } catch (error) {
      console.error('Error stopping recording:', error);
    }
  };

  const showFormCorrection = (message) => {
    Alert.alert(
      'Form Correction',
      message,
      [{ text: 'OK' }]
    );
  };

  const handleWorkoutChange = (workoutId) => {
    setSelectedWorkout(workoutId);
    setShowWorkoutSelector(false);
  };

  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const ConnectionStatus = () => (
    <View style={styles.connectionStatus}>
      <View style={[styles.connectionDot, { backgroundColor: connectionStatus === 'connected' ? COLORS.success : COLORS.error }]} />
      <Text style={styles.connectionText}>
        {connectionStatus === 'connected' ? 'Connected' : 'Disconnected'}
      </Text>
    </View>
  );

  const WorkoutSelector = () => (
    <Modal
      visible={showWorkoutSelector}
      animationType="slide"
      transparent={true}
      onRequestClose={() => setShowWorkoutSelector(false)}
    >
      <View style={styles.modalOverlay}>
        <View style={styles.modalContent}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Select Workout</Text>
            <TouchableOpacity onPress={() => setShowWorkoutSelector(false)}>
              <MaterialIcons name="close" size={24} color={COLORS.light.text} />
            </TouchableOpacity>
          </View>
          <ScrollView style={styles.workoutList}>
            {Object.entries(WORKOUT_MAP).map(([id, name]) => (
              <TouchableOpacity
                key={id}
                style={[
                  styles.workoutItem,
                  selectedWorkout === parseInt(id) && styles.selectedWorkoutItem
                ]}
                onPress={() => handleWorkoutChange(parseInt(id))}
              >
                <Text style={[
                  styles.workoutItemText,
                  selectedWorkout === parseInt(id) && styles.selectedWorkoutItemText
                ]}>
                  {name}
                </Text>
              </TouchableOpacity>
            ))}
          </ScrollView>
        </View>
      </View>
    </Modal>
  );

  if (hasPermission === null) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color={COLORS.primary} />
        <Text style={styles.loadingText}>Requesting camera permission...</Text>
      </View>
    );
  }

  if (hasPermission === false) {
    return (
      <View style={styles.errorContainer}>
        <MaterialIcons name="camera-alt" size={64} color={COLORS.error} />
        <Text style={styles.errorTitle}>Camera Access Required</Text>
        <Text style={styles.errorMessage}>
          This app needs camera access to analyze your workout form and provide real-time feedback.
        </Text>
        <TouchableOpacity style={styles.retryButton} onPress={requestCameraPermission}>
          <Text style={styles.retryButtonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Camera View */}
      <Camera
        ref={cameraRef}
        style={styles.camera}
        type={cameraType}
        ratio="16:9"
        autoFocus={CAMERA_CONFIG.autoFocus}
        whiteBalance={CAMERA_CONFIG.whiteBalance}
        zoom={CAMERA_CONFIG.zoom}
      >
        {/* Overlay UI */}
        <View style={styles.overlay}>
          {/* Top Controls */}
          <View style={styles.topControls}>
            <TouchableOpacity style={styles.backButton} onPress={() => navigation.goBack()}>
              <MaterialIcons name="arrow-back" size={24} color="white" />
            </TouchableOpacity>
            
            <ConnectionStatus />
            
            <TouchableOpacity style={styles.switchButton} onPress={toggleCameraType}>
              <MaterialIcons name="flip-camera-ios" size={24} color="white" />
            </TouchableOpacity>
          </View>

          {/* Workout Info */}
          <View style={styles.workoutInfo}>
            <TouchableOpacity 
              style={styles.workoutSelector}
              onPress={() => setShowWorkoutSelector(true)}
            >
              <Text style={styles.workoutName}>
                {WORKOUT_MAP[selectedWorkout]}
              </Text>
              <MaterialIcons name="expand-more" size={20} color="white" />
            </TouchableOpacity>
            
            <Text style={styles.sessionDuration}>
              {formatDuration(sessionDuration)}
            </Text>
          </View>

          {/* Prediction Display */}
          {predictedWorkout !== null && (
            <View style={styles.predictionContainer}>
              <Text style={styles.predictionLabel}>Detected:</Text>
              <Text style={styles.predictionText}>
                {WORKOUT_MAP[predictedWorkout]}
              </Text>
              <Text style={styles.confidenceText}>
                Confidence: {(predictionConfidence * 100).toFixed(1)}%
              </Text>
            </View>
          )}

          {/* Muscle Group Display */}
          {predictedMuscleGroup !== null && (
            <View style={styles.muscleGroupContainer}>
              <Text style={styles.muscleGroupLabel}>Muscle Group:</Text>
              <Text style={styles.muscleGroupText}>
                {MUSCLE_GROUP_MAP[predictedMuscleGroup]}
              </Text>
              <Text style={styles.confidenceText}>
                Confidence: {(muscleGroupConfidence * 100).toFixed(1)}%
              </Text>
            </View>
          )}

          {/* Bottom Controls */}
          <View style={styles.bottomControls}>
            <TouchableOpacity 
              style={[styles.recordButton, isRecording && styles.recordingButton]}
              onPress={isRecording ? stopRecording : startRecording}
            >
              <FontAwesome5 
                name={isRecording ? "stop" : "play"} 
                size={24} 
                color="white" 
              />
            </TouchableOpacity>
          </View>
        </View>
      </Camera>

      {/* Workout Selector Modal */}
      <WorkoutSelector />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'black',
  },
  camera: {
    flex: 1,
  },
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
  },
  topControls: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingTop: 50,
  },
  backButton: {
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    borderRadius: 25,
    padding: 10,
  },
  switchButton: {
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    borderRadius: 25,
    padding: 10,
  },
  connectionStatus: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 20,
  },
  connectionDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 8,
  },
  connectionText: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
  },
  workoutInfo: {
    position: 'absolute',
    top: 120,
    left: 20,
    right: 20,
    alignItems: 'center',
  },
  workoutSelector: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 25,
    marginBottom: 10,
  },
  workoutName: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
    marginRight: 10,
  },
  sessionDuration: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  predictionContainer: {
    position: 'absolute',
    top: 200,
    left: 20,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    padding: 15,
    borderRadius: 10,
  },
  predictionLabel: {
    color: 'white',
    fontSize: 12,
    marginBottom: 5,
  },
  predictionText: {
    color: COLORS.primary,
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 5,
  },
  confidenceText: {
    color: 'rgba(255, 255, 255, 0.8)',
    fontSize: 12,
  },
  muscleGroupContainer: {
    position: 'absolute',
    top: 200,
    right: 20,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    padding: 15,
    borderRadius: 10,
  },
  muscleGroupLabel: {
    color: 'white',
    fontSize: 12,
    marginBottom: 5,
  },
  muscleGroupText: {
    color: COLORS.secondary,
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 5,
  },
  bottomControls: {
    position: 'absolute',
    bottom: 50,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  recordButton: {
    backgroundColor: COLORS.primary,
    width: 80,
    height: 80,
    borderRadius: 40,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 4,
    borderColor: 'white',
  },
  recordingButton: {
    backgroundColor: COLORS.error,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: COLORS.light.background,
  },
  loadingText: {
    marginTop: 20,
    fontSize: 16,
    color: COLORS.light.text,
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: COLORS.light.background,
    padding: 20,
  },
  errorTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: COLORS.light.text,
    marginTop: 20,
    marginBottom: 10,
  },
  errorMessage: {
    fontSize: 16,
    color: COLORS.light.textSecondary,
    textAlign: 'center',
    marginBottom: 30,
  },
  retryButton: {
    backgroundColor: COLORS.primary,
    paddingHorizontal: 30,
    paddingVertical: 15,
    borderRadius: 25,
  },
  retryButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: COLORS.light.background,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    maxHeight: height * 0.7,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.light.border,
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: COLORS.light.text,
  },
  workoutList: {
    padding: 20,
  },
  workoutItem: {
    paddingVertical: 15,
    paddingHorizontal: 20,
    borderRadius: 10,
    marginBottom: 10,
    backgroundColor: COLORS.light.surface,
  },
  selectedWorkoutItem: {
    backgroundColor: COLORS.primary,
  },
  workoutItemText: {
    fontSize: 16,
    color: COLORS.light.text,
  },
  selectedWorkoutItemText: {
    color: 'white',
    fontWeight: 'bold',
  },
});

export default WorkoutScreen; 