# GymBro Mobile App

A React Native mobile application that provides AI-powered fitness tracking and real-time workout form analysis.

## Features

### 🏋️ **AI-Powered Workout Tracking**
- Real-time pose detection and analysis
- Workout classification for 22+ exercise types
- Muscle group activation detection
- Form correction feedback

### 📊 **Progress Tracking**
- Comprehensive workout statistics
- Weekly activity charts
- Progress visualization
- Achievement tracking

### 🎯 **Personalized Experience**
- User authentication and profiles
- Customizable workout preferences
- Dark/Light theme support
- Push notifications

### 🤖 **AI Training Interface**
- On-device model training
- Real-time model performance metrics
- Custom model configuration
- Privacy-focused local processing

## Tech Stack

- **Framework**: React Native with Expo
- **Navigation**: React Navigation v6
- **UI Components**: React Native Paper
- **State Management**: React Context API
- **Camera**: Expo Camera
- **Real-time Communication**: Socket.io
- **Storage**: AsyncStorage
- **Notifications**: Expo Notifications

## Prerequisites

- Node.js (v16 or higher)
- npm or yarn
- Expo CLI
- iOS Simulator (for iOS development)
- Android Studio (for Android development)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mobile-app
   ```

2. **Install dependencies**
   ```bash
   npm install
   # or
   yarn install
   ```

3. **Install Expo CLI globally**
   ```bash
   npm install -g @expo/cli
   ```

4. **Start the development server**
   ```bash
   npm start
   # or
   yarn start
   ```

## Configuration

### API Configuration

Update the API URLs in `src/config.js`:

```javascript
export const API_URL = 'http://your-backend-url:8000/api';
export const AI_URL = 'http://your-ai-service-url:5000';
```

### Environment Setup

1. **iOS Development**
   - Install Xcode
   - Install iOS Simulator
   - Run `npx expo run:ios`

2. **Android Development**
   - Install Android Studio
   - Set up Android SDK
   - Run `npx expo run:android`

3. **Web Development**
   - Run `npx expo start --web`

## Project Structure

```
mobile-app/
├── src/
│   ├── components/          # Reusable UI components
│   ├── context/            # React Context providers
│   ├── screens/            # Screen components
│   ├── services/           # API services
│   └── config.js           # App configuration
├── assets/                 # Images, fonts, etc.
├── App.js                  # Main app component
├── app.json               # Expo configuration
└── package.json           # Dependencies
```

## Key Components

### Screens
- **HomeScreen**: Dashboard with quick actions and stats
- **WorkoutScreen**: Camera-based workout tracking
- **TrainingScreen**: AI model training interface
- **DashboardScreen**: Progress tracking and analytics
- **AccountScreen**: User profile and settings
- **AuthScreen**: Login and registration

### Services
- **authService**: Authentication API calls
- **workoutService**: Workout data management
- **notificationService**: Push notification handling

## Features in Detail

### 1. Real-time Pose Detection
- Uses device camera for pose analysis
- Connects to AI service for real-time predictions
- Provides form correction feedback
- Supports 22+ workout types

### 2. Workout Classification
- Barbell Bicep Curl
- Bench Press
- Deadlift
- Squats
- And 18+ more exercises

### 3. Muscle Group Detection
- Shoulders
- Chest
- Biceps
- Core
- Triceps
- Legs
- Back

### 4. Progress Tracking
- Weekly activity charts
- Workout frequency tracking
- Calorie burn estimation
- Form accuracy metrics

## Development

### Running the App

```bash
# Start development server
npm start

# Run on iOS
npm run ios

# Run on Android
npm run android

# Run on web
npm run web
```

### Building for Production

```bash
# Build for iOS
eas build --platform ios

# Build for Android
eas build --platform android
```

### Testing

```bash
# Run tests
npm test

# Run linting
npm run lint
```

## Backend Integration

The mobile app integrates with your existing backend services:

1. **Authentication**: Uses your Django backend for user management
2. **AI Service**: Connects to your Python AI service for pose analysis
3. **Data Sync**: Syncs workout data with your backend

## Permissions

The app requires the following permissions:

- **Camera**: For pose detection and workout analysis
- **Microphone**: For video recording during workouts
- **Notifications**: For workout reminders and achievements
- **Storage**: For saving workout videos and data

## Troubleshooting

### Common Issues

1. **Camera Permission Denied**
   - Go to device settings
   - Enable camera permission for GymBro

2. **AI Service Connection Failed**
   - Check if AI service is running
   - Verify API URL in config.js
   - Check network connectivity

3. **Build Errors**
   - Clear cache: `npx expo start --clear`
   - Reinstall dependencies: `npm install`
   - Update Expo CLI: `npm install -g @expo/cli`

### Performance Optimization

- Use release builds for performance testing
- Optimize camera resolution for better performance
- Implement proper error handling for network requests

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the documentation

## Roadmap

- [ ] Offline mode support
- [ ] Social features (workout sharing)
- [ ] Advanced analytics
- [ ] Custom workout plans
- [ ] Integration with fitness trackers
- [ ] Voice commands
- [ ] AR workout guidance

---

**Note**: This mobile app is designed to work with your existing GymBro backend and AI services. Make sure your backend services are running and properly configured before testing the mobile app. 