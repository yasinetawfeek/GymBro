# GymBro Project Structure

This document provides an overview of the cleaned and organized GymBro project structure.

## 📁 Root Directory

```
GymBro/
├── backend/                    # Django REST API
│   ├── DESD_App/
│   │   ├── models/            # Organized model modules
│   │   │   ├── __init__.py
│   │   │   ├── user.py        # User and profile models
│   │   │   ├── billing.py     # Subscription and billing models
│   │   │   ├── analytics.py   # Usage tracking and metrics
│   │   │   └── ml_models.py  # ML model metadata
│   │   ├── viewsets/          # Organized viewset modules
│   │   │   ├── __init__.py
│   │   │   ├── user_management.py
│   │   │   ├── billing.py
│   │   │   ├── analytics.py
│   │   │   ├── ml_models.py
│   │   │   └── streaming.py
│   │   ├── models.py          # Legacy models (imports from modules)
│   │   ├── viewsets.py        # Legacy viewsets (imports from modules)
│   │   ├── serializers.py
│   │   ├── permissions.py
│   │   └── urls.py
│   ├── Dockerfile             # Original Dockerfile
│   ├── Dockerfile.clean       # Improved Dockerfile
│   ├── requirements.txt
│   └── README.md
├── frontend/                   # React application
│   ├── src/
│   │   ├── components/        # Organized components
│   │   │   ├── common/        # Common components
│   │   │   │   ├── LoadingSpinner.jsx
│   │   │   │   ├── ErrorBoundary.jsx
│   │   │   │   └── ProtectedRoute.jsx
│   │   │   └── layout/        # Layout components
│   │   │       └── Navbar.jsx
│   │   ├── services/          # Clean API services
│   │   │   ├── api/           # Organized API services
│   │   │   │   ├── client.js  # Base API client
│   │   │   │   ├── auth.js    # Authentication service
│   │   │   │   ├── billing.js # Billing service
│   │   │   │   ├── analytics.js # Analytics service
│   │   │   │   ├── admin.js   # Admin service
│   │   │   │   └── index.js  # Service exports
│   │   │   └── websocket/     # WebSocket services
│   │   │       └── aiService.js # AI service client
│   │   ├── contexts/          # React contexts
│   │   │   └── AuthContext.jsx
│   │   ├── pages/            # Page components
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── Dockerfile            # Original Dockerfile
│   ├── Dockerfile.clean      # Improved Dockerfile
│   ├── package.json
│   └── README.md
├── AI/                        # AI service
│   ├── app/                   # Main application
│   │   ├── models/            # Neural network models
│   │   │   ├── __init__.py
│   │   │   ├── pose_model.py  # Pose correction model
│   │   │   └── workout_classifier.py # Workout classifier
│   │   ├── services/          # Prediction services
│   │   │   ├── __init__.py
│   │   │   ├── model_loader.py # Model loading utilities
│   │   │   └── prediction_service.py # Prediction services
│   │   ├── config.py         # Configuration constants
│   │   ├── main.py           # Original main application
│   │   └── main_clean.py     # Clean, organized main application
│   ├── notebooks/            # Jupyter notebooks (organized)
│   │   ├── final_model_dnn.ipynb
│   │   ├── muscle_group_classifer_notebook.ipynb
│   │   ├── workout_classifer_final.ipynb
│   │   ├── workout_classifer.ipynb
│   │   ├── xai.ipynb
│   │   └── displacement.ipynb
│   ├── scripts/              # Utility scripts (organized)
│   │   ├── ai_model.py
│   │   ├── exercise_mapping.py
│   │   ├── fetch_dataset_from_url.py
│   │   ├── get_work_out_labels.py
│   │   ├── mediapipe_format_dataset_for_final_model.py
│   │   ├── mediapipe_format_dataset.py
│   │   ├── mediapipe_handler.py
│   │   ├── muscle_group_classifer.py
│   │   ├── pose_correction_server.py
│   │   ├── workout_classifer_model.py
│   │   └── displacement.py
│   ├── legacy/               # Legacy files (organized)
│   │   └── main.py
│   ├── training/             # Training artifacts (organized)
│   │   ├── class_accuracies.png
│   │   ├── confusion_matrix_cv.png
│   │   ├── training_history_cv.png
│   │   ├── model_fold_1.h5
│   │   ├── model_fold_2.h5
│   │   ├── model_fold_3.h5
│   │   ├── model_fold_4.h5
│   │   ├── model_fold_5.h5
│   │   └── model_checkpoints/
│   ├── models/               # Pre-trained models
│   │   ├── muscle_group_classifiers/
│   │   ├── pose_correctors/
│   │   └── workout_classifiers/
│   ├── data/                 # Training data
│   │   ├── extracted_frames/
│   │   ├── pose_visualizations/
│   │   └── processed_videos/
│   ├── Dockerfile           # Original Dockerfile
│   ├── Dockerfile.clean     # Improved Dockerfile
│   ├── requirements.txt
│   └── README.md
├── docker-compose.yml        # Original docker-compose
├── docker-compose.clean.yml # Improved docker-compose
├── .env.example             # Environment template
├── Makefile                 # Development commands
├── README.md                # Main project documentation
└── PROJECT_STRUCTURE.md    # This file
```

## 🎯 Key Improvements Made

### 1. Backend Organization
- **Models**: Split into domain-specific modules (user, billing, analytics, ml_models)
- **Viewsets**: Organized by functionality (user_management, billing, analytics, ml_models, streaming)
- **Legacy Support**: Maintained backward compatibility with original files

### 2. Frontend Organization
- **Services**: Clean, organized API services with proper error handling
- **Components**: Separated common and layout components
- **Contexts**: Centralized state management with AuthContext
- **WebSocket**: Dedicated service for AI communication

### 3. AI Service Organization
- **Models**: Clean neural network definitions
- **Services**: Separated model loading and prediction logic
- **Configuration**: Centralized constants and mappings
- **Clean Main**: Organized main application with proper separation of concerns

### 4. File Organization
- **Notebooks**: Moved all Jupyter notebooks to dedicated directory
- **Scripts**: Utility scripts organized in scripts directory
- **Legacy**: Old files moved to legacy directory
- **Training**: Training artifacts and checkpoints organized

### 5. Docker Improvements
- **Multi-stage Builds**: Separate development and production stages
- **Security**: Non-root users, proper permissions
- **Health Checks**: Built-in health monitoring
- **Optimization**: Smaller image sizes, better caching

### 6. Development Experience
- **Makefile**: Convenient commands for all operations
- **Documentation**: Comprehensive README files for each component
- **Environment**: Proper environment variable management
- **Testing**: Organized test structure

## 🚀 Getting Started

### Quick Start
```bash
# Setup environment
make setup

# Start development
make dev

# Start production
make prod
```

### Available Commands
```bash
make help          # Show all available commands
make dev           # Start development environment
make prod          # Start production environment
make test          # Run all tests
make lint          # Run linting
make clean         # Clean up resources
```

## 📊 Benefits of New Structure

1. **Maintainability**: Clear separation of concerns
2. **Scalability**: Modular architecture supports growth
3. **Developer Experience**: Easy to navigate and understand
4. **Documentation**: Comprehensive guides for each component
5. **Testing**: Organized test structure
6. **Deployment**: Production-ready Docker configuration
7. **Security**: Proper security practices implemented

## 🔄 Migration Notes

- **Backward Compatibility**: All original functionality preserved
- **Legacy Files**: Moved to appropriate directories, not deleted
- **Configuration**: Environment variables maintained
- **Database**: No schema changes, only code organization
- **API**: All endpoints remain the same

---

**GymBro** - Now with clean, maintainable, and scalable architecture! 🏗️✨