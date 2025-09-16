# GymBro - AI-Powered Fitness Training Platform

GymBro is a comprehensive fitness training platform that uses AI to provide real-time pose correction and workout classification. The platform consists of three main components: a Django backend, a React frontend, and a Flask AI service.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend │    │  Django Backend  │    │   AI Service    │
│                 │    │                 │    │                 │
│  - User Interface│◄──►│  - REST API     │◄──►│  - Pose Analysis│
│  - Real-time UI │    │  - Authentication│    │  - Workout Class│
│  - WebSocket     │    │  - User Management│   │  - Corrections  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   PostgreSQL    │
                    │                 │
                    │  - User Data    │
                    │  - Analytics    │
                    │  - Billing      │
                    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Make (optional, for convenience commands)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd GymBro
   ```

2. **Setup environment**
   ```bash
   make setup
   # Edit .env file with your configuration
   ```

3. **Start the application**
   ```bash
   # For development
   make dev
   
   # For production
   make prod
   ```

4. **Access the application**
   - Frontend: http://localhost (production) or http://localhost:3000 (development)
   - Backend API: http://localhost:8000
   - AI Service: http://localhost:8001

## 📁 Project Structure

```
GymBro/
├── backend/                 # Django REST API
│   ├── DESD_App/
│   │   ├── models/          # Organized model modules
│   │   ├── viewsets/        # Organized viewset modules
│   │   ├── serializers.py   # API serializers
│   │   └── ...
│   ├── Dockerfile.clean     # Improved Dockerfile
│   └── requirements.txt
├── frontend/                # React application
│   ├── src/
│   │   ├── components/      # Organized components
│   │   ├── services/        # Clean API services
│   │   ├── contexts/        # React contexts
│   │   └── ...
│   ├── Dockerfile.clean     # Improved Dockerfile
│   └── package.json
├── AI/                      # AI service
│   ├── app/
│   │   ├── models/          # ML model definitions
│   │   ├── services/        # Prediction services
│   │   ├── config.py        # Configuration
│   │   └── main_clean.py    # Clean main application
│   ├── Dockerfile.clean     # Improved Dockerfile
│   └── requirements.txt
├── docker-compose.clean.yml # Improved Docker Compose
├── Makefile                 # Development commands
├── .env.example            # Environment template
└── README.md               # This file
```

## 🛠️ Development

### Available Commands

```bash
# Setup and start
make setup          # Setup development environment
make dev            # Start development environment
make prod           # Start production environment

# Docker operations
make build          # Build all images
make up             # Start services
make down           # Stop services
make logs           # View logs

# Database operations
make db-migrate      # Run migrations
make db-shell       # Open database shell
make db-reset        # Reset database (WARNING: deletes data)

# Testing and quality
make test           # Run all tests
make lint           # Run linting
make format         # Format code
make health         # Check service health

# Cleanup
make clean          # Clean Docker resources
make clean-volumes  # Clean Docker volumes
```

### Development Workflow

1. **Start development environment**
   ```bash
   make dev
   ```

2. **Make changes to your code**
   - Backend changes are automatically reloaded
   - Frontend changes are hot-reloaded
   - AI service changes require restart

3. **Run tests**
   ```bash
   make test
   ```

4. **Check code quality**
   ```bash
   make lint
   make format
   ```

## 🔧 Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Database
DB_USER=gymbro
DB_PASSWORD=your-password
DB_NAME=gymbro_db

# Django
SECRET_KEY=your-secret-key
DEBUG=False

# Services
VITE_API_URL=http://localhost:8000
VITE_AI_URL=http://localhost:8001
```

### Service Configuration

- **Backend**: Django REST API with PostgreSQL
- **Frontend**: React with Vite, served by Nginx
- **AI Service**: Flask with PyTorch models
- **Database**: PostgreSQL with Redis for caching

## 📊 Features

### Core Features
- **User Authentication**: JWT-based authentication with role management
- **Real-time Pose Correction**: AI-powered pose analysis and corrections
- **Workout Classification**: Automatic workout type detection
- **Muscle Group Detection**: Identify activated muscle groups
- **Session Tracking**: Usage analytics and billing

### User Roles
- **Customer**: Basic workout features
- **AI Engineer**: Model management and analytics
- **Admin**: User management and billing

### AI Capabilities
- **Pose Correction**: Real-time feedback on exercise form
- **Workout Classification**: 22 different workout types
- **Muscle Group Detection**: 7 muscle group categories
- **Performance Analytics**: Model performance tracking

## 🧪 Testing

### Backend Testing
```bash
make test-backend
```

### Frontend Testing
```bash
make test-frontend
```

### Integration Testing
```bash
make test
```

## 🚀 Deployment

### Production Deployment

1. **Configure production environment**
   ```bash
   cp .env.example .env
   # Edit .env with production values
   ```

2. **Build and start production services**
   ```bash
   make build
   make prod
   ```

3. **Run initial setup**
   ```bash
   make db-migrate
   make django-createsuperuser
   make django-collectstatic
   ```

### Docker Swarm Deployment

```bash
docker stack deploy -c docker-compose.clean.yml gymbro
```

## 📈 Monitoring

### Health Checks
```bash
make health
```

### Resource Monitoring
```bash
make monitor
```

### Logs
```bash
make logs          # All services
make logs-backend  # Backend only
make logs-frontend # Frontend only
make logs-ai      # AI service only
```

## 🔒 Security

### Security Features
- JWT authentication with refresh tokens
- Role-based access control
- CORS configuration
- Input validation and sanitization
- SQL injection prevention
- XSS protection

### Security Scanning
```bash
make security-scan
```

## 📚 API Documentation

### Backend API
- **Base URL**: http://localhost:8000/api/
- **Authentication**: JWT Bearer tokens
- **Documentation**: Available at `/api/docs/` (Swagger UI)

### AI Service API
- **WebSocket**: ws://localhost:8001
- **Health Check**: http://localhost:8001/health

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

### Code Standards
- Follow PEP 8 for Python code
- Use ESLint for JavaScript/React code
- Write tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

### Troubleshooting

**Services won't start**
```bash
make logs
make health
```

**Database connection issues**
```bash
make db-shell
```

**Permission issues**
```bash
make clean
make build
```

### Getting Help

- Check the logs: `make logs`
- Verify health: `make health`
- Review configuration: Check `.env` file
- Check Docker resources: `make monitor`

## 🔄 Updates

### Updating the Application

1. **Pull latest changes**
   ```bash
   git pull origin main
   ```

2. **Rebuild services**
   ```bash
   make build
   make up
   ```

3. **Run migrations**
   ```bash
   make db-migrate
   ```

---

**GymBro** - Making fitness training smarter with AI 🏋️‍♂️🤖