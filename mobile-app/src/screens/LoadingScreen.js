import React from 'react';
import { View, Text, StyleSheet, ActivityIndicator } from 'react-native';
import { FontAwesome5 } from '@expo/vector-icons';
import { COLORS } from '../config';

const LoadingScreen = () => {
  return (
    <View style={styles.container}>
      <FontAwesome5 name="dumbbell" size={64} color={COLORS.primary} />
      <Text style={styles.title}>GymBro</Text>
      <Text style={styles.subtitle}>Loading...</Text>
      <ActivityIndicator size="large" color={COLORS.primary} style={styles.spinner} />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: COLORS.light.background,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    color: COLORS.light.text,
    marginTop: 20,
    marginBottom: 10,
  },
  subtitle: {
    fontSize: 16,
    color: COLORS.light.textSecondary,
    marginBottom: 30,
  },
  spinner: {
    marginTop: 20,
  },
});

export default LoadingScreen; 