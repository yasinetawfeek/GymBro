#!/bin/bash

# GymBro Mobile App Setup Script
echo "🏋️ Setting up GymBro Mobile App..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js v16 or higher."
    exit 1
fi

# Check Node.js version
NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 16 ]; then
    echo "❌ Node.js version 16 or higher is required. Current version: $(node -v)"
    exit 1
fi

echo "✅ Node.js version: $(node -v)"

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm."
    exit 1
fi

echo "✅ npm version: $(npm -v)"

# Install Expo CLI globally
echo "📱 Installing Expo CLI..."
npm install -g @expo/cli

# Install project dependencies
echo "📦 Installing project dependencies..."
npm install

# Create assets directory if it doesn't exist
if [ ! -d "assets" ]; then
    echo "📁 Creating assets directory..."
    mkdir assets
fi

# Create placeholder assets
echo "🎨 Creating placeholder assets..."
touch assets/icon.png
touch assets/splash.png
touch assets/adaptive-icon.png
touch assets/favicon.png

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️ Creating .env file..."
    cat > .env << EOF
# API Configuration
API_URL=http://localhost:8000/api
AI_URL=http://localhost:5000

# App Configuration
APP_NAME=GymBro
APP_VERSION=1.0.0

# Development
EXPO_DEVTOOLS_LISTEN_ADDRESS=0.0.0.0
EOF
fi

# Update config.js with environment variables
echo "🔧 Updating configuration..."
if [ -f "src/config.js" ]; then
    # Backup original config
    cp src/config.js src/config.js.backup
    
    # Update API URLs to use environment variables
    sed -i 's|http://localhost:8000/api|process.env.API_URL|g' src/config.js
    sed -i 's|http://localhost:5000|process.env.AI_URL|g' src/config.js
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Update API URLs in src/config.js"
echo "2. Add your app icons to the assets/ directory"
echo "3. Run 'npm start' to start the development server"
echo ""
echo "For iOS development:"
echo "- Install Xcode and iOS Simulator"
echo "- Run 'npx expo run:ios'"
echo ""
echo "For Android development:"
echo "- Install Android Studio and Android SDK"
echo "- Run 'npx expo run:android'"
echo ""
echo "For web development:"
echo "- Run 'npx expo start --web'"
echo ""
echo "📚 Check README.md for detailed instructions" 