import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Dimensions,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { MaterialIcons, FontAwesome5 } from '@expo/vector-icons';
import { useAuth } from '../context/AuthContext';
import { COLORS } from '../config';

const { width } = Dimensions.get('window');

const DashboardScreen = () => {
  const { user } = useAuth();
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [selectedPeriod, setSelectedPeriod] = useState('week');

  // Mock data - replace with actual API calls
  const [stats, setStats] = useState({
    totalWorkouts: 45,
    totalMinutes: 1800,
    totalCalories: 12500,
    currentStreak: 7,
    averageAccuracy: 85,
    favoriteWorkout: 'Bench Press',
  });

  const [weeklyData, setWeeklyData] = useState([
    { day: 'Mon', workouts: 2, minutes: 45 },
    { day: 'Tue', workouts: 1, minutes: 30 },
    { day: 'Wed', workouts: 3, minutes: 60 },
    { day: 'Thu', workouts: 0, minutes: 0 },
    { day: 'Fri', workouts: 2, minutes: 50 },
    { day: 'Sat', workouts: 1, minutes: 25 },
    { day: 'Sun', workouts: 2, minutes: 40 },
  ]);

  const StatCard = ({ title, value, subtitle, icon, color, gradientColors }) => (
    <TouchableOpacity style={styles.statCard}>
      <LinearGradient
        colors={gradientColors || [color, color]}
        style={styles.statCardGradient}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
      >
        <View style={styles.statCardContent}>
          <View style={styles.statCardIcon}>
            {icon}
          </View>
          <View style={styles.statCardText}>
            <Text style={styles.statCardValue}>{value}</Text>
            <Text style={styles.statCardTitle}>{title}</Text>
            {subtitle && <Text style={styles.statCardSubtitle}>{subtitle}</Text>}
          </View>
        </View>
      </LinearGradient>
    </TouchableOpacity>
  );

  const ProgressBar = ({ value, maxValue, color, label }) => (
    <View style={styles.progressContainer}>
      <View style={styles.progressHeader}>
        <Text style={styles.progressLabel}>{label}</Text>
        <Text style={styles.progressValue}>{value}/{maxValue}</Text>
      </View>
      <View style={styles.progressBar}>
        <View 
          style={[
            styles.progressFill, 
            { 
              width: `${(value / maxValue) * 100}%`,
              backgroundColor: color 
            }
          ]} 
        />
      </View>
    </View>
  );

  const WeeklyChart = () => (
    <View style={styles.weeklyChart}>
      <Text style={styles.chartTitle}>This Week's Activity</Text>
      <View style={styles.chartContainer}>
        {weeklyData.map((day, index) => (
          <View key={index} style={styles.chartBar}>
            <View style={styles.chartBarContainer}>
              <View 
                style={[
                  styles.chartBarFill,
                  { 
                    height: day.minutes > 0 ? `${(day.minutes / 60) * 100}%` : 0,
                    backgroundColor: day.minutes > 0 ? COLORS.primary : COLORS.light.border
                  }
                ]} 
              />
            </View>
            <Text style={styles.chartBarLabel}>{day.day}</Text>
            <Text style={styles.chartBarValue}>{day.workouts}</Text>
          </View>
        ))}
      </View>
    </View>
  );

  const PeriodSelector = () => (
    <View style={styles.periodSelector}>
      {['week', 'month', 'year'].map((period) => (
        <TouchableOpacity
          key={period}
          style={[
            styles.periodButton,
            selectedPeriod === period && styles.periodButtonActive
          ]}
          onPress={() => setSelectedPeriod(period)}
        >
          <Text style={[
            styles.periodButtonText,
            selectedPeriod === period && styles.periodButtonTextActive
          ]}>
            {period.charAt(0).toUpperCase() + period.slice(1)}
          </Text>
        </TouchableOpacity>
      ))}
    </View>
  );

  return (
    <ScrollView style={[styles.container, { backgroundColor: isDarkMode ? COLORS.dark.background : COLORS.light.background }]}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={[styles.headerTitle, { color: isDarkMode ? COLORS.dark.text : COLORS.light.text }]}>
          Dashboard
        </Text>
        <Text style={[styles.headerSubtitle, { color: isDarkMode ? COLORS.dark.textSecondary : COLORS.light.textSecondary }]}>
          Track your fitness progress
        </Text>
      </View>

      {/* Period Selector */}
      <PeriodSelector />

      {/* Stats Grid */}
      <View style={styles.statsGrid}>
        <StatCard
          title="Total Workouts"
          value={stats.totalWorkouts}
          icon={<FontAwesome5 name="dumbbell" size={24} color="white" />}
          color={COLORS.primary}
          gradientColors={[COLORS.primary, '#1d4ed8']}
        />
        <StatCard
          title="Total Minutes"
          value={stats.totalMinutes}
          subtitle="This month"
          icon={<MaterialIcons name="timer" size={24} color="white" />}
          color={COLORS.secondary}
          gradientColors={[COLORS.secondary, '#d97706']}
        />
        <StatCard
          title="Calories Burned"
          value={stats.totalCalories.toLocaleString()}
          icon={<FontAwesome5 name="fire" size={24} color="white" />}
          color={COLORS.error}
          gradientColors={[COLORS.error, '#dc2626']}
        />
        <StatCard
          title="Current Streak"
          value={`${stats.currentStreak} days`}
          icon={<MaterialIcons name="local-fire-department" size={24} color="white" />}
          color={COLORS.success}
          gradientColors={[COLORS.success, '#059669']}
        />
      </View>

      {/* Progress Section */}
      <View style={styles.progressSection}>
        <Text style={[styles.sectionTitle, { color: isDarkMode ? COLORS.dark.text : COLORS.light.text }]}>
          Progress
        </Text>
        <View style={styles.progressCards}>
          <ProgressBar
            value={stats.averageAccuracy}
            maxValue={100}
            color={COLORS.primary}
            label="Form Accuracy"
          />
          <ProgressBar
            value={stats.currentStreak}
            maxValue={30}
            color={COLORS.success}
            label="Streak Goal"
          />
        </View>
      </View>

      {/* Weekly Chart */}
      <WeeklyChart />

      {/* Recent Activity */}
      <View style={styles.recentActivity}>
        <Text style={[styles.sectionTitle, { color: isDarkMode ? COLORS.dark.text : COLORS.light.text }]}>
          Recent Activity
        </Text>
        <View style={styles.activityList}>
          {[
            { workout: 'Bench Press', date: '2 hours ago', accuracy: 92 },
            { workout: 'Squats', date: 'Yesterday', accuracy: 88 },
            { workout: 'Deadlift', date: '2 days ago', accuracy: 85 },
          ].map((activity, index) => (
            <View key={index} style={styles.activityItem}>
              <View style={styles.activityIcon}>
                <FontAwesome5 name="dumbbell" size={16} color={COLORS.primary} />
              </View>
              <View style={styles.activityContent}>
                <Text style={[styles.activityWorkout, { color: isDarkMode ? COLORS.dark.text : COLORS.light.text }]}>
                  {activity.workout}
                </Text>
                <Text style={[styles.activityDate, { color: isDarkMode ? COLORS.dark.textSecondary : COLORS.light.textSecondary }]}>
                  {activity.date}
                </Text>
              </View>
              <View style={styles.activityAccuracy}>
                <Text style={styles.accuracyText}>{activity.accuracy}%</Text>
                <Text style={styles.accuracyLabel}>Accuracy</Text>
              </View>
            </View>
          ))}
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
    marginBottom: 30,
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    marginBottom: 5,
  },
  headerSubtitle: {
    fontSize: 16,
  },
  periodSelector: {
    flexDirection: 'row',
    backgroundColor: COLORS.light.surface,
    borderRadius: 12,
    padding: 4,
    marginBottom: 20,
  },
  periodButton: {
    flex: 1,
    paddingVertical: 10,
    alignItems: 'center',
    borderRadius: 8,
  },
  periodButtonActive: {
    backgroundColor: COLORS.primary,
  },
  periodButtonText: {
    fontSize: 14,
    fontWeight: '500',
    color: COLORS.light.textSecondary,
  },
  periodButtonTextActive: {
    color: 'white',
    fontWeight: 'bold',
  },
  statsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    marginBottom: 30,
  },
  statCard: {
    width: (width - 60) / 2,
    marginBottom: 15,
    borderRadius: 15,
    overflow: 'hidden',
  },
  statCardGradient: {
    padding: 20,
    borderRadius: 15,
  },
  statCardContent: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  statCardIcon: {
    marginRight: 15,
  },
  statCardText: {
    flex: 1,
  },
  statCardValue: {
    color: 'white',
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 5,
  },
  statCardTitle: {
    color: 'rgba(255, 255, 255, 0.9)',
    fontSize: 14,
    fontWeight: '500',
  },
  statCardSubtitle: {
    color: 'rgba(255, 255, 255, 0.7)',
    fontSize: 12,
    marginTop: 2,
  },
  progressSection: {
    marginBottom: 30,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 15,
  },
  progressCards: {
    backgroundColor: COLORS.light.surface,
    borderRadius: 15,
    padding: 20,
  },
  progressContainer: {
    marginBottom: 15,
  },
  progressHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  progressLabel: {
    fontSize: 14,
    fontWeight: '500',
    color: COLORS.light.text,
  },
  progressValue: {
    fontSize: 14,
    fontWeight: 'bold',
    color: COLORS.primary,
  },
  progressBar: {
    height: 8,
    backgroundColor: COLORS.light.border,
    borderRadius: 4,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: 4,
  },
  weeklyChart: {
    marginBottom: 30,
  },
  chartTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: COLORS.light.text,
    marginBottom: 15,
  },
  chartContainer: {
    backgroundColor: COLORS.light.surface,
    borderRadius: 15,
    padding: 20,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-end',
    height: 150,
  },
  chartBar: {
    alignItems: 'center',
    flex: 1,
  },
  chartBarContainer: {
    width: 20,
    height: 80,
    backgroundColor: COLORS.light.border,
    borderRadius: 10,
    overflow: 'hidden',
    marginBottom: 8,
  },
  chartBarFill: {
    position: 'absolute',
    bottom: 0,
    width: '100%',
    borderRadius: 10,
  },
  chartBarLabel: {
    fontSize: 12,
    color: COLORS.light.textSecondary,
    marginBottom: 4,
  },
  chartBarValue: {
    fontSize: 12,
    fontWeight: 'bold',
    color: COLORS.primary,
  },
  recentActivity: {
    marginBottom: 30,
  },
  activityList: {
    backgroundColor: COLORS.light.surface,
    borderRadius: 15,
    padding: 20,
  },
  activityItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 15,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.light.border,
  },
  activityIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: COLORS.light.background,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 15,
  },
  activityContent: {
    flex: 1,
  },
  activityWorkout: {
    fontSize: 16,
    fontWeight: '500',
    marginBottom: 2,
  },
  activityDate: {
    fontSize: 14,
  },
  activityAccuracy: {
    alignItems: 'flex-end',
  },
  accuracyText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: COLORS.primary,
  },
  accuracyLabel: {
    fontSize: 12,
    color: COLORS.light.textSecondary,
  },
});

export default DashboardScreen; 