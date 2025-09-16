# GymBro Backend - Django REST API

The backend service provides a comprehensive REST API for the GymBro fitness platform, handling user management, authentication, billing, analytics, and AI service integration.

## 🏗️ Architecture

The backend is built with Django REST Framework and follows a clean, modular architecture:

```
backend/
├── DESD_App/
│   ├── models/              # Data models organized by domain
│   │   ├── user.py         # User and profile models
│   │   ├── billing.py      # Subscription and billing models
│   │   ├── analytics.py    # Usage tracking and metrics
│   │   └── ml_models.py   # ML model metadata
│   ├── viewsets/           # API viewsets organized by domain
│   │   ├── user_management.py
│   │   ├── billing.py
│   │   ├── analytics.py
│   │   ├── ml_models.py
│   │   └── streaming.py
│   ├── serializers.py      # API serializers
│   ├── permissions.py      # Custom permissions
│   └── urls.py            # URL routing
├── Dockerfile.clean        # Production-ready Dockerfile
└── requirements.txt       # Python dependencies
```

## 🚀 Quick Start

### Development Setup

```bash
# Start development environment
make dev

# Or manually with Docker
docker-compose -f docker-compose.clean.yml --profile development up backend-dev
```

### Production Setup

```bash
# Start production environment
make prod

# Or manually with Docker
docker-compose -f docker-compose.clean.yml up backend
```

## 📊 Models

### User Management
- **UserProfile**: Extended user information with fitness data
- **User**: Django's built-in user model

### Billing & Subscriptions
- **Subscription**: User subscription plans and limits
- **Invoice**: Billing records and payment tracking
- **BillingRecord**: Legacy billing data (for compatibility)

### Analytics & Tracking
- **UsageRecord**: Session tracking and usage metrics
- **ModelPerformanceMetric**: AI model performance analytics
- **UserLastViewedExercise**: User activity tracking

### ML Models
- **MLModel**: Machine learning model metadata and configuration

## 🔌 API Endpoints

### Authentication
- `POST /auth/jwt/create/` - Login
- `POST /auth/users/` - Register
- `POST /auth/jwt/refresh/` - Refresh token

### User Management
- `GET /api/my_account/` - Get current user
- `PATCH /api/my_account/` - Update current user
- `GET /api/manage_accounts/` - List users (admin)
- `POST /api/manage_accounts/` - Create user (admin)
- `PATCH /api/manage_accounts/{id}/` - Update user (admin)
- `DELETE /api/manage_accounts/{id}/` - Delete user (admin)

### Role Management
- `GET /api/role_info/` - Get user role information
- `PATCH /api/approvals/{id}/approve/` - Approve user (admin)
- `PATCH /api/approvals/{id}/reject/` - Reject user (admin)

### Billing & Subscriptions
- `GET /api/subscriptions/` - Get user subscriptions
- `POST /api/subscriptions/` - Create subscription
- `PATCH /api/subscriptions/{id}/cancel/` - Cancel subscription
- `GET /api/invoices/` - Get user invoices
- `PATCH /api/invoices/{id}/pay/` - Pay invoice
- `GET /api/billing/overview/` - Get billing overview

### Analytics & Usage
- `POST /api/usage/start_session/` - Start usage session
- `PATCH /api/usage/{id}/end_session/` - End usage session
- `PATCH /api/usage/{id}/update_metrics/` - Update session metrics
- `GET /api/usage/` - Get usage records

### ML Models (AI Engineers)
- `GET /api/ml-models/` - List ML models
- `POST /api/ml-models/` - Create ML model
- `PATCH /api/ml-models/{id}/deploy/` - Deploy model
- `PATCH /api/ml-models/{id}/undeploy/` - Undeploy model
- `GET /api/ml-models/deployed/` - Get deployed models

### Performance Analytics (AI Engineers)
- `POST /api/model-performance/record_metrics/` - Record performance metrics
- `GET /api/model-performance/analytics/` - Get performance analytics

### Streaming
- `GET /api/stream-info/` - Get stream information
- `GET /api/get-token/` - Get VideoSDK token

## 🔐 Authentication & Permissions

### Authentication
- JWT-based authentication with access and refresh tokens
- Automatic token refresh on API calls
- Token verification for WebSocket connections

### Permission Classes
- **IsAuthenticated**: Requires valid authentication
- **IsAdminUser**: Requires admin privileges
- **IsOwner**: User can only access their own data
- **IsApprovedUser**: Requires user approval
- **IsAIEngineerRole**: Requires AI Engineer role
- **IsApprovedAIEngineer**: Requires approved AI Engineer status

### Role-Based Access Control
- **Customer**: Basic workout features
- **AI Engineer**: Model management and analytics
- **Admin**: Full system access

## 🗄️ Database

### PostgreSQL Configuration
```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.getenv('DB_NAME', 'gymbro_db'),
        'USER': os.getenv('DB_USER', 'gymbro'),
        'PASSWORD': os.getenv('DB_PASSWORD', 'gymbro123'),
        'HOST': os.getenv('DB_HOST', 'db'),
        'PORT': os.getenv('DB_PORT', '5432'),
    }
}
```

### Migrations
```bash
# Create migrations
make db-makemigrations

# Apply migrations
make db-migrate

# Reset database (WARNING: deletes all data)
make db-reset
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
make test-backend

# Run specific test
docker-compose exec backend python manage.py test DESD_App.tests.test_subscription
```

### Test Structure
```
DESD_App/
├── tests/
│   ├── test_subscription.py
│   ├── test_user_management.py
│   ├── test_analytics.py
│   └── test_permissions.py
```

## 📈 Monitoring & Logging

### Health Checks
- Endpoint: `/health/`
- Checks database connectivity
- Returns service status

### Logging Configuration
```python
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': 'logs/gymbro.log',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['file'],
            'level': 'INFO',
            'propagate': True,
        },
    },
}
```

## 🔧 Configuration

### Environment Variables
```bash
# Database
DB_USER=gymbro
DB_PASSWORD=your-password
DB_NAME=gymbro_db

# Django
SECRET_KEY=your-secret-key
DEBUG=False
ALLOWED_HOSTS=localhost,127.0.0.1

# Redis
REDIS_URL=redis://redis:6379/0

# AI Service
AI_SERVICE_URL=http://ai:8001
```

### Settings Structure
- `settings.py`: Main settings file
- Environment-based configuration
- Separate settings for development/production

## 🚀 Deployment

### Docker Deployment
```bash
# Build image
docker build -f Dockerfile.clean -t gymbro-backend .

# Run container
docker run -p 8000:8000 gymbro-backend
```

### Production Considerations
- Use environment variables for secrets
- Enable HTTPS in production
- Configure proper CORS settings
- Set up database backups
- Monitor resource usage

## 🔒 Security

### Security Features
- JWT authentication with refresh tokens
- Role-based access control
- Input validation and sanitization
- SQL injection prevention
- XSS protection
- CORS configuration

### Security Best Practices
- Use strong secret keys
- Enable HTTPS in production
- Regular security updates
- Input validation
- Rate limiting (consider adding)

## 📚 API Documentation

### Swagger UI
- Available at `/api/docs/` when DEBUG=True
- Interactive API documentation
- Request/response examples

### API Versioning
- Current version: v1
- URL-based versioning: `/api/v1/`
- Backward compatibility maintained

## 🤝 Contributing

### Development Workflow
1. Create feature branch
2. Make changes
3. Write tests
4. Run linting and tests
5. Submit pull request

### Code Standards
- Follow PEP 8
- Write docstrings for functions/classes
- Add type hints where possible
- Write tests for new features

## 🆘 Troubleshooting

### Common Issues

**Database Connection Errors**
```bash
# Check database status
make logs-db

# Test connection
make db-shell
```

**Permission Errors**
```bash
# Check user permissions
docker-compose exec backend python manage.py shell
>>> from django.contrib.auth.models import User
>>> user = User.objects.get(username='your-username')
>>> user.groups.all()
```

**Migration Issues**
```bash
# Reset migrations
make db-reset
make db-migrate
```

### Debug Mode
```bash
# Enable debug mode
export DEBUG=True
make restart
```

---

**Backend Service** - The heart of the GymBro platform 🚀