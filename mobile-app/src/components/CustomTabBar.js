import React from 'react';
import { View, TouchableOpacity, Text, StyleSheet, Dimensions } from 'react-native';
import { MaterialIcons, FontAwesome5 } from '@expo/vector-icons';
import { COLORS } from '../config';

const { width } = Dimensions.get('window');

const CustomTabBar = ({ state, descriptors, navigation }) => {
  const getIcon = (routeName, focused) => {
    const iconColor = focused ? COLORS.primary : COLORS.light.textSecondary;
    const iconSize = 24;

    switch (routeName) {
      case 'Home':
        return <MaterialIcons name="home" size={iconSize} color={iconColor} />;
      case 'Workout':
        return <FontAwesome5 name="dumbbell" size={iconSize} color={iconColor} />;
      case 'Training':
        return <MaterialIcons name="psychology" size={iconSize} color={iconColor} />;
      case 'Dashboard':
        return <MaterialIcons name="bar-chart" size={iconSize} color={iconColor} />;
      default:
        return <MaterialIcons name="home" size={iconSize} color={iconColor} />;
    }
  };

  const getLabel = (routeName) => {
    switch (routeName) {
      case 'Home':
        return 'Home';
      case 'Workout':
        return 'Workout';
      case 'Training':
        return 'Training';
      case 'Dashboard':
        return 'Stats';
      default:
        return 'Home';
    }
  };

  return (
    <View style={styles.container}>
      {state.routes.map((route, index) => {
        const { options } = descriptors[route.key];
        const label = getLabel(route.name);
        const isFocused = state.index === index;

        const onPress = () => {
          const event = navigation.emit({
            type: 'tabPress',
            target: route.key,
            canPreventDefault: true,
          });

          if (!isFocused && !event.defaultPrevented) {
            navigation.navigate(route.name);
          }
        };

        return (
          <TouchableOpacity
            key={route.key}
            accessibilityRole="button"
            accessibilityState={isFocused ? { selected: true } : {}}
            accessibilityLabel={options.tabBarAccessibilityLabel}
            testID={options.tabBarTestID}
            onPress={onPress}
            style={styles.tabButton}
          >
            <View style={styles.tabContent}>
              {getIcon(route.name, isFocused)}
              <Text style={[
                styles.tabLabel,
                isFocused && styles.tabLabelFocused
              ]}>
                {label}
              </Text>
              {isFocused && <View style={styles.activeIndicator} />}
            </View>
          </TouchableOpacity>
        );
      })}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    backgroundColor: COLORS.light.background,
    borderTopWidth: 1,
    borderTopColor: COLORS.light.border,
    paddingBottom: 20,
    paddingTop: 10,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: -2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 3,
    elevation: 8,
  },
  tabButton: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 8,
  },
  tabContent: {
    alignItems: 'center',
    position: 'relative',
  },
  tabLabel: {
    fontSize: 12,
    marginTop: 4,
    color: COLORS.light.textSecondary,
    fontWeight: '500',
  },
  tabLabelFocused: {
    color: COLORS.primary,
    fontWeight: 'bold',
  },
  activeIndicator: {
    position: 'absolute',
    bottom: -8,
    width: 4,
    height: 4,
    borderRadius: 2,
    backgroundColor: COLORS.primary,
  },
});

export default CustomTabBar; 