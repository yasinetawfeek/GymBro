import React, { useState, useEffect } from 'react';
import { StatusBar } from 'expo-status-bar';
import { NavigationContainer, DefaultTheme as NavDefaultTheme, DarkTheme as NavDarkTheme } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createDrawerNavigator } from '@react-navigation/drawer';
import { Provider as PaperProvider, MD3LightTheme, MD3DarkTheme } from 'react-native-paper';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Constants from 'expo-constants';

// Import screens
import HomeScreen from './src/screens/HomeScreen';
import AuthScreen from './src/screens/AuthScreen';
import WorkoutScreen from './src/screens/WorkoutScreen';
import TrainingScreen from './src/screens/TrainingScreen';
import DashboardScreen from './src/screens/DashboardScreen';
import AccountScreen from './src/screens/AccountScreen';
import LoadingScreen from './src/screens/LoadingScreen';

// Import components
import CustomDrawerContent from './src/components/CustomDrawerContent';
import CustomTabBar from './src/components/CustomTabBar';

// Import context
import { AuthProvider, useAuth } from './src/context/AuthContext';

const Stack = createStackNavigator();
const Tab = createBottomTabNavigator();
const Drawer = createDrawerNavigator();

// Custom Paper themes (MD3)
const paperLightTheme = {
  ...MD3LightTheme,
  colors: { ...MD3LightTheme.colors, primary: '#3b82f6', surface: '#f8fafc', text: '#1e293b' },
};
const paperDarkTheme = {
  ...MD3DarkTheme,
  colors: { ...MD3DarkTheme.colors, primary: '#3b82f6', surface: '#1e293b', text: '#f1f5f9' },
};

// Navigation themes
const navLightTheme = {
  ...NavDefaultTheme,
  colors: {
    ...NavDefaultTheme.colors,
    primary: '#3b82f6',
    background: '#ffffff',
    card: '#f8fafc',
    text: '#1e293b',
    border: '#e2e8f0',
    notification: '#3b82f6',
  },
};
const navDarkTheme = {
  ...NavDarkTheme,
  colors: {
    ...NavDarkTheme.colors,
    primary: '#3b82f6',
    background: '#0f172a',
    card: '#1e293b',
    text: '#f1f5f9',
    border: '#334155',
    notification: '#3b82f6',
  },
};

// Tab Navigator
const TabNavigator = () => {
  return (
    <Tab.Navigator
      tabBar={props => <CustomTabBar {...props} />}
      screenOptions={{
        headerShown: false,
      }}
    >
      <Tab.Screen 
        name="Home" 
        component={HomeScreen}
        options={{
          tabBarIcon: ({ focused, color }) => (
            <TabIcon name="home" focused={focused} color={color} />
          ),
        }}
      />
      <Tab.Screen 
        name="Workout" 
        component={WorkoutScreen}
        options={{
          tabBarIcon: ({ focused, color }) => (
            <TabIcon name="dumbbell" focused={focused} color={color} />
          ),
        }}
      />
      <Tab.Screen 
        name="Training" 
        component={TrainingScreen}
        options={{
          tabBarIcon: ({ focused, color }) => (
            <TabIcon name="brain" focused={focused} color={color} />
          ),
        }}
      />
      <Tab.Screen 
        name="Dashboard" 
        component={DashboardScreen}
        options={{
          tabBarIcon: ({ focused, color }) => (
            <TabIcon name="bar-chart-2" focused={focused} color={color} />
          ),
        }}
      />
    </Tab.Navigator>
  );
};

// Drawer Navigator
const DrawerNavigator = () => {
  return (
    <Drawer.Navigator
      drawerContent={props => <CustomDrawerContent {...props} />}
      screenOptions={{
        headerShown: false,
        drawerStyle: {
          backgroundColor: '#1e293b',
          width: 280,
        },
        drawerLabelStyle: {
          color: '#f1f5f9',
        },
        drawerActiveBackgroundColor: '#3b82f6',
        drawerActiveTintColor: '#ffffff',
        drawerInactiveTintColor: '#94a3b8',
      }}
    >
      <Drawer.Screen name="MainTabs" component={TabNavigator} />
      <Drawer.Screen name="Account" component={AccountScreen} />
    </Drawer.Navigator>
  );
};

// Main App Component
const AppContent = () => {
  const { user, isLoading } = useAuth();
  const [isDarkMode, setIsDarkMode] = useState(false);

  useEffect(() => {
    loadThemePreference();
    requestNotificationPermissions();
  }, []);

  const loadThemePreference = async () => {
    try {
      const theme = await AsyncStorage.getItem('theme');
      if (theme) {
        setIsDarkMode(theme === 'dark');
      }
    } catch (error) {
      console.log('Error loading theme preference:', error);
    }
  };

  const requestNotificationPermissions = async () => {
    try {
      // Skip in Expo Go (store client) where remote push is unsupported
      if (Constants.executionEnvironment === 'storeClient') {
        return;
      }
      const Notifications = await import('expo-notifications');
      const { status } = await Notifications.requestPermissionsAsync();
      if (status !== 'granted') {
        console.log('Notification permissions not granted');
      }
    } catch (error) {
      console.log('Error requesting notification permissions:', error);
    }
  };

  if (isLoading) {
    return <LoadingScreen />;
  }

  return (
    <PaperProvider theme={isDarkMode ? paperDarkTheme : paperLightTheme}>
      <NavigationContainer theme={isDarkMode ? navDarkTheme : navLightTheme}>
        <StatusBar style={isDarkMode ? 'light' : 'dark'} />
        <Stack.Navigator screenOptions={{ headerShown: false }}>
          {user ? (
            <Stack.Screen name="Main" component={DrawerNavigator} />
          ) : (
            <>
              <Stack.Screen name="Auth" component={AuthScreen} />
              <Stack.Screen name="Home" component={HomeScreen} />
            </>
          )}
        </Stack.Navigator>
      </NavigationContainer>
    </PaperProvider>
  );
};

// Simple Tab Icon Component
const TabIcon = ({ name, focused, color }) => {
  return null;
};

// Main App
export default function App() {
  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <AuthProvider>
        <AppContent />
      </AuthProvider>
    </GestureHandlerRootView>
  );
} 