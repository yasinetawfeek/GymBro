import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Image,
  ScrollView,
} from 'react-native';
import { MaterialIcons, FontAwesome5 } from '@expo/vector-icons';
import { useAuth } from '../context/AuthContext';
import { COLORS } from '../config';

const CustomDrawerContent = (props) => {
  const { user, logout } = useAuth();

  const handleLogout = async () => {
    await logout();
    props.navigation.closeDrawer();
  };

  const menuItems = [
    {
      title: 'Home',
      icon: 'home',
      onPress: () => {
        props.navigation.navigate('MainTabs', { screen: 'Home' });
        props.navigation.closeDrawer();
      },
    },
    {
      title: 'Workout',
      icon: 'dumbbell',
      onPress: () => {
        props.navigation.navigate('MainTabs', { screen: 'Workout' });
        props.navigation.closeDrawer();
      },
    },
    {
      title: 'Training',
      icon: 'psychology',
      onPress: () => {
        props.navigation.navigate('MainTabs', { screen: 'Training' });
        props.navigation.closeDrawer();
      },
    },
    {
      title: 'Dashboard',
      icon: 'bar-chart',
      onPress: () => {
        props.navigation.navigate('MainTabs', { screen: 'Dashboard' });
        props.navigation.closeDrawer();
      },
    },
    {
      title: 'Account',
      icon: 'person',
      onPress: () => {
        props.navigation.navigate('Account');
        props.navigation.closeDrawer();
      },
    },
  ];

  const MenuItem = ({ item }) => (
    <TouchableOpacity style={styles.menuItem} onPress={item.onPress}>
      <View style={styles.menuItemContent}>
        {item.icon === 'dumbbell' ? (
          <FontAwesome5 name={item.icon} size={20} color={COLORS.light.text} />
        ) : (
          <MaterialIcons name={item.icon} size={24} color={COLORS.light.text} />
        )}
        <Text style={styles.menuItemText}>{item.title}</Text>
      </View>
      <MaterialIcons name="chevron-right" size={20} color={COLORS.light.textSecondary} />
    </TouchableOpacity>
  );

  return (
    <View style={styles.container}>
      {/* User Profile Section */}
      <View style={styles.profileSection}>
        <View style={styles.profileImageContainer}>
          <FontAwesome5 name="user-circle" size={60} color={COLORS.primary} />
        </View>
        <View style={styles.profileInfo}>
          <Text style={styles.userName}>
            {user?.forename && user?.surname 
              ? `${user.forename} ${user.surname}`
              : user?.username || 'User'
            }
          </Text>
          <Text style={styles.userEmail}>
            {user?.email || 'user@example.com'}
          </Text>
        </View>
      </View>

      {/* Menu Items */}
      <ScrollView style={styles.menuContainer}>
        {menuItems.map((item, index) => (
          <MenuItem key={index} item={item} />
        ))}
      </ScrollView>

      {/* Logout Section */}
      <View style={styles.logoutSection}>
        <TouchableOpacity style={styles.logoutButton} onPress={handleLogout}>
          <MaterialIcons name="logout" size={20} color={COLORS.error} />
          <Text style={styles.logoutText}>Logout</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.light.background,
  },
  profileSection: {
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.light.border,
    backgroundColor: COLORS.light.surface,
  },
  profileImageContainer: {
    alignItems: 'center',
    marginBottom: 15,
  },
  profileInfo: {
    alignItems: 'center',
  },
  userName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: COLORS.light.text,
    marginBottom: 5,
  },
  userEmail: {
    fontSize: 14,
    color: COLORS.light.textSecondary,
  },
  menuContainer: {
    flex: 1,
    paddingTop: 10,
  },
  menuItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 15,
    paddingHorizontal: 20,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.light.border,
  },
  menuItemContent: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  menuItemText: {
    fontSize: 16,
    color: COLORS.light.text,
    marginLeft: 15,
    fontWeight: '500',
  },
  logoutSection: {
    padding: 20,
    borderTopWidth: 1,
    borderTopColor: COLORS.light.border,
  },
  logoutButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
  },
  logoutText: {
    fontSize: 16,
    color: COLORS.error,
    marginLeft: 15,
    fontWeight: '500',
  },
});

export default CustomDrawerContent; 