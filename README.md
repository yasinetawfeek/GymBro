# GymBro - AI-Powered Fitness Training Platform

## 🏋️ Project Overview

**GymBro** is a cutting-edge, full-stack fitness application that leverages advanced computer vision and machine learning to provide real-time workout guidance and pose correction. The platform combines a modern React frontend, robust Django backend, and sophisticated AI services to create an intelligent personal trainer experience.

### 🎯 Mission
To democratize access to professional-quality fitness training by providing AI-powered real-time feedback, making proper form accessible to everyone, regardless of their fitness level or access to personal trainers.

---

## 🚀 Key Features & Achievements

### 🤖 Advanced AI/ML Capabilities

#### **Real-Time Pose Correction Engine**
- **Deep Neural Network (DNN)** model trained on extensive pose data
- **Real-time feedback** with visual corrections displayed as directional arrows
- **22+ workout types** supported including compound movements (deadlifts, squats, bench press)
- **Sub-50ms inference time** for seamless real-time experience
- **WebSocket-based communication** for low-latency pose streaming

#### **Intelligent Workout Classification**
- **LSTM-based sequence model** with attention mechanisms
- **Temporal pattern recognition** using 10-frame sequences
- **92%+ accuracy** in workout type identification
- **SMOTE-balanced training** for robust performance across all exercise types
- **Real-time classification** with prediction smoothing

#### **Muscle Group Detection**
- **Random Forest classifier** with 92% accuracy
- **Feature engineering** from pose landmarks
- **Real-time muscle group identification** for targeted feedback
- **Integration with pose correction** for exercise-specific guidance

### 💻 Full-Stack Architecture

#### **Frontend (React + Vite)**
- **Modern React 18** with hooks and context API
- **Real-time camera integration** using MediaPipe Pose
- **Responsive design** with Tailwind CSS
- **Role-based access control** (Customer, Admin, AI Engineer)
- **Dark/light mode** with system preference detection
- **Progressive Web App** capabilities

#### **Backend (Django + DRF)**
- **RESTful API** with Django REST Framework
- **JWT authentication** with refresh token support
- **Role-based permissions** and user management
- **Video streaming** capabilities with HLS
- **Usage tracking** and analytics
- **Swagger/OpenAPI** documentation

#### **AI Services (Flask + PyTorch)**
- **Microservice architecture** for AI model serving
- **PyTorch-based models** with CUDA support
- **WebSocket server** for real-time communication
- **Model versioning** and performance monitoring
- **Scalable inference pipeline**

### 📊 Technical Achievements

#### **Model Performance**
- **Pose Correction Model**: 36-dimensional output for precise joint adjustments
- **Workout Classifier**: 22-class LSTM with attention mechanism
- **Muscle Group Classifier**: 92% accuracy with Random Forest
- **Cross-validation**: 5-fold CV with early stopping
- **Data Augmentation**: Noise injection and SMOTE for robustness

#### **System Performance**
- **Real-time processing**: <50ms inference latency
- **Concurrent users**: Multi-client WebSocket support
- **Scalable architecture**: Docker containerization
- **Production-ready**: Gunicorn + Nginx deployment

---

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend │    │  Django Backend │    │   AI Services   │
│                 │    │                 │    │                 │
│ • MediaPipe     │◄──►│ • REST API      │◄──►│ • PyTorch Models│
│ • Real-time UI  │    │ • JWT Auth      │    │ • WebSocket     │
│ • Role-based    │    │ • User Mgmt     │    │ • Pose Analysis │
│ • Responsive    │    │ • Video Stream  │    │ • Classification│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### **Data Flow**
1. **Camera Capture** → MediaPipe Pose Detection
2. **Pose Landmarks** → WebSocket to AI Service
3. **AI Processing** → Pose Correction + Classification
4. **Real-time Feedback** → Visual Corrections + Exercise Info
5. **Usage Tracking** → Django Backend Analytics

---

## 🎯 Supported Workouts

The platform supports **22+ exercise types** with real-time form correction:

**Upper Body**: Barbell Bicep Curl, Hammer Curl, Bench Press, Incline/Decline Bench Press, Shoulder Press, Lateral Raises, Tricep Dips, Tricep Pushdown, Pull-ups, Lat Pulldown, T-Bar Row

**Lower Body**: Squat, Deadlift, Romanian Deadlift, Hip Thrust, Leg Extensions

**Core**: Plank, Leg Raises, Russian Twist

**Machine Exercises**: Chest Fly Machine, Leg Extensions

---

## 🛠️ Technology Stack

### **Frontend**
- **React 18** with Vite
- **Tailwind CSS** for styling
- **Framer Motion** for animations
- **MediaPipe** for pose detection
- **Socket.IO** for real-time communication
- **Axios** for API calls

### **Backend**
- **Django 5.1** with Django REST Framework
- **PostgreSQL/SQLite** database
- **JWT Authentication** with Djoser
- **CORS** enabled for cross-origin requests
- **Swagger/OpenAPI** documentation

### **AI/ML**
- **PyTorch** for deep learning models
- **MediaPipe** for pose landmark extraction
- **Scikit-learn** for traditional ML models
- **NumPy/Pandas** for data processing
- **Flask** for AI service API

### **DevOps**
- **Docker** containerization
- **Docker Compose** for orchestration
- **Nginx** for static file serving
- **Gunicorn** for WSGI serving

---

## 📈 Business Value & Market Potential

### **Target Market**
- **Fitness enthusiasts** seeking form improvement
- **Home gym users** without access to trainers
- **Gyms** looking to enhance member experience
- **Corporate wellness** programs
- **Physical therapy** and rehabilitation

### **Competitive Advantages**
- **Real-time AI feedback** vs. static video tutorials
- **Comprehensive exercise library** with 22+ movements
- **Multi-role platform** supporting different user types
- **Scalable architecture** ready for enterprise deployment
- **Proven model accuracy** with 92%+ performance metrics

### **Revenue Potential**
- **SaaS subscription** model for individual users
- **Enterprise licensing** for gyms and wellness programs
- **API licensing** for fitness app developers
- **White-label solutions** for fitness brands

---

## 🚀 Getting Started

### **Prerequisites**
- Docker and Docker Compose
- Node.js 18+ (for development)
- Python 3.9+ (for AI services)

### **Quick Start**
```bash
# Clone the repository
git clone <repository-url>
cd GymBro

# Start all services with Docker Compose
docker-compose up --build

# Access the application
# Frontend: http://localhost
# Backend API: http://localhost:8000
# AI Service: http://localhost:8001
```

### **Development Setup**
```bash
# Backend
cd backend
pip install -r requirements.txt
python manage.py runserver

# Frontend
cd frontend
npm install
npm run dev

# AI Services
cd AI/app
pip install -r requirements.txt
python main.py
```

---

## 📊 Performance Metrics

- **Model Accuracy**: 92%+ across all classifiers
- **Inference Speed**: <50ms per frame
- **Supported Exercises**: 22+ workout types
- **Real-time Processing**: 30 FPS pose detection
- **Concurrent Users**: Multi-client WebSocket support
- **Uptime**: Production-ready with Docker deployment

---

## 🔮 Future Roadmap

### **Phase 1** (Current)
- ✅ Real-time pose correction
- ✅ Workout classification
- ✅ User management system
- ✅ Role-based access control

### **Phase 2** (Next 3 months)
- 🔄 Mobile app development (React Native)
- 🔄 Advanced analytics dashboard
- 🔄 Social features and challenges
- 🔄 Integration with fitness trackers

### **Phase 3** (6 months)
- 🔄 Personalized workout plans
- 🔄 Nutrition tracking integration
- 🔄 Virtual personal trainer AI
- 🔄 Enterprise features and white-labeling

---

## 👥 Team & Development

This project demonstrates expertise in:
- **Full-stack development** (React, Django, Python)
- **Machine Learning** (PyTorch, Computer Vision, LSTM)
- **Real-time systems** (WebSocket, MediaPipe)
- **DevOps** (Docker, CI/CD, Production deployment)
- **API design** (RESTful, WebSocket, Authentication)

---

## 📞 Contact & Investment

For investment opportunities, technical partnerships, or recruitment inquiries, please contact:

- **Technical Lead**: Yasine Sayed Tawfeek
- **Email**: yasineayman@hotmail.com
<!-- - **LinkedIn**: [Your LinkedIn]
- **GitHub**: [Your GitHub] -->

---

*GymBro represents the future of fitness technology, combining cutting-edge AI with user-friendly design to make professional-quality training accessible to everyone.*