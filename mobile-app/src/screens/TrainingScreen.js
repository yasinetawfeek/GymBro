import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { MaterialIcons, FontAwesome5 } from '@expo/vector-icons';
import { COLORS } from '../config';

const TrainingScreen = () => {
  const [selectedModel, setSelectedModel] = useState('workout-classification');
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingResults, setTrainingResults] = useState(null);

  const modelOptions = [
    {
      id: 'workout-classification',
      name: 'Workout Classification',
      description: 'Classify different types of exercises',
      icon: 'fitness-center',
      color: COLORS.primary,
    },
    {
      id: 'muscle-activation',
      name: 'Muscle Group Detection',
      description: 'Detect which muscle groups are being activated',
      icon: 'psychology',
      color: COLORS.secondary,
    },
    {
      id: 'form-correction',
      name: 'Form Correction',
      description: 'Provide real-time form feedback',
      icon: 'check-circle',
      color: COLORS.success,
    },
  ];

  const startTraining = () => {
    if (!selectedModel) {
      Alert.alert('Error', 'Please select a model to train.');
      return;
    }

    setIsTraining(true);
    setTrainingProgress(0);
    setTrainingResults(null);

    // Simulate training progress
    const interval = setInterval(() => {
      setTrainingProgress((prev) => {
        const newProgress = prev + Math.random() * 10;
        if (newProgress >= 100) {
          clearInterval(interval);
          setIsTraining(false);
          setTrainingResults({
            accuracy: (80 + Math.random() * 15).toFixed(2),
            loss: (0.1 + Math.random() * 0.2).toFixed(4),
            trainingTime: (10 + Math.random() * 50).toFixed(1),
          });
          return 100;
        }
        return newProgress;
      });
    }, 500);
  };

  const ModelCard = ({ model, isSelected, onSelect }) => (
    <TouchableOpacity
      style={[styles.modelCard, isSelected && styles.modelCardSelected]}
      onPress={() => onSelect(model.id)}
    >
      <View style={styles.modelCardHeader}>
        <View style={[styles.modelIcon, { backgroundColor: model.color }]}>
          <MaterialIcons name={model.icon} size={24} color="white" />
        </View>
        <View style={styles.modelInfo}>
          <Text style={styles.modelName}>{model.name}</Text>
          <Text style={styles.modelDescription}>{model.description}</Text>
        </View>
        {isSelected && (
          <MaterialIcons name="check-circle" size={24} color={COLORS.primary} />
        )}
      </View>
    </TouchableOpacity>
  );

  const TrainingProgress = () => (
    <View style={styles.trainingProgress}>
      <Text style={styles.progressTitle}>Training in Progress...</Text>
      <View style={styles.progressBar}>
        <View 
          style={[
            styles.progressFill, 
            { width: `${trainingProgress}%` }
          ]} 
        />
      </View>
      <Text style={styles.progressText}>{trainingProgress.toFixed(1)}%</Text>
    </View>
  );

  const TrainingResults = () => (
    <View style={styles.trainingResults}>
      <Text style={styles.resultsTitle}>Training Complete!</Text>
      <View style={styles.resultsGrid}>
        <View style={styles.resultCard}>
          <Text style={styles.resultValue}>{trainingResults.accuracy}%</Text>
          <Text style={styles.resultLabel}>Accuracy</Text>
        </View>
        <View style={styles.resultCard}>
          <Text style={styles.resultValue}>{trainingResults.loss}</Text>
          <Text style={styles.resultLabel}>Loss</Text>
        </View>
        <View style={styles.resultCard}>
          <Text style={styles.resultValue}>{trainingResults.trainingTime}s</Text>
          <Text style={styles.resultLabel}>Time</Text>
        </View>
      </View>
    </View>
  );

  return (
    <ScrollView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>AI Training</Text>
        <Text style={styles.headerSubtitle}>
          Train and configure AI models for better workout analysis
        </Text>
      </View>

      {/* Model Selection */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Select Model</Text>
        <View style={styles.modelList}>
          {modelOptions.map((model) => (
            <ModelCard
              key={model.id}
              model={model}
              isSelected={selectedModel === model.id}
              onSelect={setSelectedModel}
            />
          ))}
        </View>
      </View>

      {/* Training Controls */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Training Controls</Text>
        <View style={styles.controlsContainer}>
          <TouchableOpacity
            style={[styles.trainButton, isTraining && styles.trainButtonDisabled]}
            onPress={startTraining}
            disabled={isTraining}
          >
            {isTraining ? (
              <ActivityIndicator color="white" size="small" />
            ) : (
              <MaterialIcons name="play-arrow" size={24} color="white" />
            )}
            <Text style={styles.trainButtonText}>
              {isTraining ? 'Training...' : 'Start Training'}
            </Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Training Progress */}
      {isTraining && <TrainingProgress />}

      {/* Training Results */}
      {trainingResults && <TrainingResults />}

      {/* Model Information */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Model Information</Text>
        <View style={styles.infoCard}>
          <View style={styles.infoRow}>
            <MaterialIcons name="info" size={20} color={COLORS.primary} />
            <Text style={styles.infoText}>
              Models are trained on your device to ensure privacy and faster inference.
            </Text>
          </View>
          <View style={styles.infoRow}>
            <MaterialIcons name="schedule" size={20} color={COLORS.secondary} />
            <Text style={styles.infoText}>
              Training typically takes 5-15 minutes depending on model complexity.
            </Text>
          </View>
          <View style={styles.infoRow}>
            <MaterialIcons name="battery-charging-full" size={20} color={COLORS.success} />
            <Text style={styles.infoText}>
              Keep your device plugged in during training for best results.
            </Text>
          </View>
        </View>
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingHorizontal: 20,
    paddingTop: 20,
    backgroundColor: COLORS.light.background,
  },
  header: {
    marginBottom: 30,
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: COLORS.light.text,
    marginBottom: 5,
  },
  headerSubtitle: {
    fontSize: 16,
    color: COLORS.light.textSecondary,
  },
  section: {
    marginBottom: 30,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: COLORS.light.text,
    marginBottom: 15,
  },
  modelList: {
    gap: 15,
  },
  modelCard: {
    backgroundColor: COLORS.light.surface,
    borderRadius: 15,
    padding: 20,
    borderWidth: 2,
    borderColor: 'transparent',
  },
  modelCardSelected: {
    borderColor: COLORS.primary,
    backgroundColor: COLORS.light.background,
  },
  modelCardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  modelIcon: {
    width: 50,
    height: 50,
    borderRadius: 25,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 15,
  },
  modelInfo: {
    flex: 1,
  },
  modelName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: COLORS.light.text,
    marginBottom: 5,
  },
  modelDescription: {
    fontSize: 14,
    color: COLORS.light.textSecondary,
  },
  controlsContainer: {
    backgroundColor: COLORS.light.surface,
    borderRadius: 15,
    padding: 20,
  },
  trainButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: COLORS.primary,
    paddingVertical: 15,
    borderRadius: 12,
    gap: 10,
  },
  trainButtonDisabled: {
    opacity: 0.6,
  },
  trainButtonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
  trainingProgress: {
    backgroundColor: COLORS.light.surface,
    borderRadius: 15,
    padding: 20,
    marginBottom: 30,
  },
  progressTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: COLORS.light.text,
    marginBottom: 15,
    textAlign: 'center',
  },
  progressBar: {
    height: 8,
    backgroundColor: COLORS.light.border,
    borderRadius: 4,
    overflow: 'hidden',
    marginBottom: 10,
  },
  progressFill: {
    height: '100%',
    backgroundColor: COLORS.primary,
    borderRadius: 4,
  },
  progressText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: COLORS.primary,
    textAlign: 'center',
  },
  trainingResults: {
    backgroundColor: COLORS.light.surface,
    borderRadius: 15,
    padding: 20,
    marginBottom: 30,
  },
  resultsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: COLORS.light.text,
    marginBottom: 15,
    textAlign: 'center',
  },
  resultsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  resultCard: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: 15,
  },
  resultValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: COLORS.primary,
    marginBottom: 5,
  },
  resultLabel: {
    fontSize: 14,
    color: COLORS.light.textSecondary,
  },
  infoCard: {
    backgroundColor: COLORS.light.surface,
    borderRadius: 15,
    padding: 20,
  },
  infoRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 15,
  },
  infoText: {
    flex: 1,
    fontSize: 14,
    color: COLORS.light.text,
    marginLeft: 10,
    lineHeight: 20,
  },
});

export default TrainingScreen; 