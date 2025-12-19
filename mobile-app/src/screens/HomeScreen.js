import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Image,
  Dimensions,
  Alert,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useNavigation } from '@react-navigation/native';
import { useAuth } from '../context/AuthContext';
import { COLORS, WORKOUT_MAP, API_URL } from '../config';
import { MaterialIcons, Ionicons, FontAwesome5 } from '@expo/vector-icons';
import * as Notifications from 'expo-notifications';
import Constants from 'expo-constants';
import axios from 'axios';

const { width, height } = Dimensions.get('window');

const HomeScreen = () => {
  const navigation = useNavigation();
  const { user, logout, token } = useAuth();
  const [lastViewedExercise, setLastViewedExercise] = useState(null);
  const [isDarkMode, setIsDarkMode] = useState(false);

  useEffect(() => {
    loadLastViewedExercise();
    requestNotificationPermissions();
  }, []);

  const loadLastViewedExercise = async () => {
    setLastViewedExercise({
      workout_type: 0,
      workout_name: "Barbell Bicep Curl",
      timestamp: new Date().toISOString(),
    });
  };

  const requestNotificationPermissions = async () => {
    try {
      if (Constants.executionEnvironment === 'storeClient') {
        return;
      }
      const { status } = await Notifications.requestPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert(
          'Permission Required',
          'Please enable notifications to receive workout reminders and achievements.'
        );
      }
    } catch (error) {
      console.log('Error requesting notification permissions:', error);
    }
  };

  const handleStartWorkout = () => {
    navigation.navigate('Workout');
  };

  const handleViewStats = () => {
    navigation.navigate('Dashboard');
  };

  const handleAccountSettings = () => {
    navigation.navigate('Account');
  };

  const handleLogout = async () => {
    Alert.alert(
      'Logout',
      'Are you sure you want to logout?',
      [
        {
          text: 'Cancel',
          style: 'cancel',
        },
        {
          text: 'Logout',
          style: 'destructive',
          onPress: async () => {
            await logout();
          },
        },
      ]
    );
  };

  const testBackend = async () => {
    const base = API_URL.replace(/\/$/, '');
    const url = `${base}/api/stream-info/`;
    try {
      const res = await axios.get(url, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
        timeout: 8000,
      });
      Alert.alert('Backend OK', `Status: ${res.status}\n${typeof res.data === 'string' ? res.data : JSON.stringify(res.data).slice(0, 400)}`);
    } catch (err) {
      const status = err.response?.status;
      const data = err.response?.data;
      Alert.alert('Backend Error', `URL: ${url}\nStatus: ${status ?? 'N/A'}\n${data ? JSON.stringify(data).slice(0, 400) : err.message}`);
    }
  };

  const QuickActionCard = ({ title, subtitle, icon, onPress, gradientColors }) => (
    <TouchableOpacity style={styles.quickActionCard} onPress={onPress}>
      <LinearGradient
        colors={gradientColors}
        style={styles.quickActionGradient}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
      >
        <View style={styles.quickActionContent}>
          <View style={styles.quickActionIcon}>
            {icon}
          </View>
          <View style={styles.quickActionText}>
            <Text style={styles.quickActionTitle}>{title}</Text>
            <Text style={styles.quickActionSubtitle}>{subtitle}</Text>
          </View>
        </View>
      </LinearGradient>
    </TouchableOpacity>
  );

  const LastViewedCard = () => {
    if (!lastViewedExercise) return null;

    return (
      <TouchableOpacity 
        style={styles.lastViewedCard}
        onPress={() => navigation.navigate('Workout')}
      >
        <View style={styles.lastViewedHeader}>
          <Text style={styles.lastViewedTitle}>Continue Your Workout</Text>
          <MaterialIcons name="fitness-center" size={24} color={COLORS.primary} />
        </View>
        <Text style={styles.lastViewedExercise}>
          {lastViewedExercise.workout_name}
        </Text>
        <Text style={styles.lastViewedTime}>
          Last viewed: {new Date(lastViewedExercise.timestamp).toLocaleDateString()}
        </Text>
      </TouchableOpacity>
    );
  };

  const StatsCard = ({ title, value, icon, color }) => (
    <View style={styles.statsCard}>
      <View style={[styles.statsIcon, { backgroundColor: color }]}>
        {icon}
      </View>
      <View style={styles.statsContent}>
        <Text style={styles.statsValue}>{value}</Text>
        <Text style={styles.statsTitle}>{title}</Text>
      </View>
    </View>
  );

  return (
    <ScrollView style={[styles.container, { backgroundColor: isDarkMode ? COLORS.dark.background : COLORS.light.background }]}>
      <View style={styles.header}>
        <View style={styles.headerContent}>
          <Text style={[styles.greeting, { color: isDarkMode ? COLORS.dark.text : COLORS.light.text }]}>
            Welcome back, {user?.forename || user?.username || 'Fitness Enthusiast'}! 💪
          </Text>
          <Text style={[styles.subtitle, { color: isDarkMode ? COLORS.dark.textSecondary : COLORS.light.textSecondary }]}>
            Ready for your next workout?
          </Text>
        </View>
        <TouchableOpacity style={styles.profileButton} onPress={handleAccountSettings}>
          <FontAwesome5 name="user-circle" size={24} color={COLORS.primary} />
        </TouchableOpacity>
      </View>

      <View style={styles.section}>
        <Text style={[styles.sectionTitle, { color: isDarkMode ? COLORS.dark.text : COLORS.light.text }]}>
          Quick Actions
        </Text>
        <View style={styles.quickActionsGrid}>
          <QuickActionCard
            title="Start Workout"
            subtitle="Begin your fitness journey"
            icon={<FontAwesome5 name="dumbbell" size={24} color="white" />}
            onPress={handleStartWorkout}
            gradientColors={[COLORS.primary, '#1d4ed8']}
          />
          <QuickActionCard
            title="View Stats"
            subtitle="Track your progress"
            icon={<FontAwesome5 name="chart-line" size={24} color="white" />}
            onPress={handleViewStats}
            gradientColors={[COLORS.secondary, '#d97706']}
          />
        </View>
        <TouchableOpacity style={styles.testButton} onPress={testBackend}>
          <Text style={styles.testButtonText}>Test Backend</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.section}>
        <LastViewedCard />
      </View>

      <View style={styles.section}>
        <Text style={[styles.sectionTitle, { color: isDarkMode ? COLORS.dark.text : COLORS.light.text }]}>
          This Week
        </Text>
        <View style={styles.statsGrid}>
          <StatsCard
            title="Workouts"
            value="5"
            icon={<FontAwesome5 name="dumbbell" size={16} color="white" />}
            color={COLORS.primary}
          />
          <StatsCard
            title="Minutes"
            value="120"
            icon={<FontAwesome5 name="clock" size={16} color="white" />}
            color={COLORS.secondary}
          />
          <StatsCard
            title="Calories"
            value="850"
            icon={<FontAwesome5 name="fire" size={16} color="white" />}
            color={COLORS.error}
          />
        </View>
      </View>

      <View style={styles.section}>
        <Text style={[styles.sectionTitle, { color: isDarkMode ? COLORS.dark.text : COLORS.light.text }]}>
          Features
        </Text>
        <View style={styles.featuresList}>
          <View style={styles.featureItem}>
            <MaterialIcons name="camera-alt" size={20} color={COLORS.primary} />
            <Text style={[styles.featureText, { color: isDarkMode ? COLORS.dark.text : COLORS.light.text }]}>
              AI-Powered Form Analysis
            </Text>
          </View>
          <View style={styles.featureItem}>
            <MaterialIcons name="psychology" size={20} color={COLORS.primary} />
            <Text style={[styles.featureText, { color: isDarkMode ? COLORS.dark.text : COLORS.light.text }]}>
              Real-time Feedback
            </Text>
          </View>
          <View style={styles.featureItem}>
            <MaterialIcons name="analytics" size={20} color={COLORS.primary} />
            <Text style={[styles.featureText, { color: isDarkMode ? COLORS.dark.text : COLORS.light.text }]}>
              Progress Tracking
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
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 30,
  },
  headerContent: {
    flex: 1,
  },
  greeting: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 5,
  },
  subtitle: {
    fontSize: 16,
  },
  profileButton: {
    padding: 10,
  },
  section: {
    marginBottom: 30,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 15,
  },
  quickActionsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  quickActionCard: {
    width: (width - 60) / 2,
    borderRadius: 15,
    overflow: 'hidden',
  },
  quickActionGradient: {
    padding: 20,
    borderRadius: 15,
  },
  quickActionContent: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  quickActionIcon: {
    marginRight: 15,
  },
  quickActionText: {
    flex: 1,
  },
  quickActionTitle: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 5,
  },
  quickActionSubtitle: {
    color: 'rgba(255, 255, 255, 0.8)',
    fontSize: 12,
  },
  testButton: {
    marginTop: 12,
    backgroundColor: COLORS.primary,
    paddingVertical: 12,
    borderRadius: 12,
    alignItems: 'center',
  },
  testButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
  lastViewedCard: {
    backgroundColor: COLORS.light.surface,
    padding: 20,
    borderRadius: 15,
    borderLeftWidth: 4,
    borderLeftColor: COLORS.primary,
  },
  lastViewedHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  lastViewedTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: COLORS.light.text,
  },
  lastViewedExercise: {
    fontSize: 18,
    fontWeight: 'bold',
    color: COLORS.primary,
    marginBottom: 5,
  },
  lastViewedTime: {
    fontSize: 12,
    color: COLORS.light.textSecondary,
  },
  statsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  statsCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: COLORS.light.surface,
    padding: 15,
    borderRadius: 10,
    width: (width - 80) / 3,
  },
  statsIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 10,
  },
  statsContent: {
    flex: 1,
  },
  statsValue: {
    fontSize: 18,
    fontWeight: 'bold',
    color: COLORS.light.text,
  },
  statsTitle: {
    fontSize: 12,
    color: COLORS.light.textSecondary,
  },
  featuresList: {
    backgroundColor: COLORS.light.surface,
    borderRadius: 15,
    padding: 20,
  },
  featureItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 15,
  },
  featureText: {
    marginLeft: 15,
    fontSize: 16,
  },
});

export default HomeScreen; 