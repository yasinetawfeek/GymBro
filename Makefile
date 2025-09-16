# GymBro Development Makefile
# This file provides convenient commands for development and deployment

.PHONY: help build up down restart logs clean dev prod test lint format

# Default target
help: ## Show this help message
	@echo "GymBro Development Commands"
	@echo "=========================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Environment setup
setup: ## Setup development environment
	@echo "Setting up development environment..."
	@cp .env.example .env
	@echo "Please edit .env file with your configuration"
	@echo "Installing dependencies..."

# Docker commands
build: ## Build all Docker images
	@echo "Building Docker images..."
	docker-compose -f docker-compose.clean.yml build

build-dev: ## Build development Docker images
	@echo "Building development Docker images..."
	docker-compose -f docker-compose.clean.yml --profile development build

up: ## Start all services in production mode
	@echo "Starting services in production mode..."
	docker-compose -f docker-compose.clean.yml up -d

up-dev: ## Start all services in development mode
	@echo "Starting services in development mode..."
	docker-compose -f docker-compose.clean.yml --profile development up -d

down: ## Stop all services
	@echo "Stopping all services..."
	docker-compose -f docker-compose.clean.yml down

restart: ## Restart all services
	@echo "Restarting all services..."
	docker-compose -f docker-compose.clean.yml restart

logs: ## Show logs for all services
	docker-compose -f docker-compose.clean.yml logs -f

logs-backend: ## Show backend logs
	docker-compose -f docker-compose.clean.yml logs -f backend

logs-frontend: ## Show frontend logs
	docker-compose -f docker-compose.clean.yml logs -f frontend

logs-ai: ## Show AI service logs
	docker-compose -f docker-compose.clean.yml logs -f ai

logs-db: ## Show database logs
	docker-compose -f docker-compose.clean.yml logs -f db

# Development commands
dev: ## Start development environment
	@echo "Starting development environment..."
	docker-compose -f docker-compose.clean.yml --profile development up -d
	@echo "Development environment started!"
	@echo "Frontend: http://localhost:3000"
	@echo "Backend: http://localhost:8000"
	@echo "AI Service: http://localhost:8001"

prod: ## Start production environment
	@echo "Starting production environment..."
	docker-compose -f docker-compose.clean.yml up -d
	@echo "Production environment started!"
	@echo "Application: http://localhost"

# Database commands
db-migrate: ## Run database migrations
	@echo "Running database migrations..."
	docker-compose -f docker-compose.clean.yml exec backend python manage.py migrate

db-makemigrations: ## Create database migrations
	@echo "Creating database migrations..."
	docker-compose -f docker-compose.clean.yml exec backend python manage.py makemigrations

db-reset: ## Reset database (WARNING: This will delete all data)
	@echo "Resetting database..."
	docker-compose -f docker-compose.clean.yml exec backend python manage.py flush --noinput

db-shell: ## Open database shell
	docker-compose -f docker-compose.clean.yml exec db psql -U gymbro -d gymbro_db

# Django commands
django-shell: ## Open Django shell
	docker-compose -f docker-compose.clean.yml exec backend python manage.py shell

django-createsuperuser: ## Create Django superuser
	docker-compose -f docker-compose.clean.yml exec backend python manage.py createsuperuser

django-collectstatic: ## Collect static files
	docker-compose -f docker-compose.clean.yml exec backend python manage.py collectstatic --noinput

# Testing commands
test: ## Run all tests
	@echo "Running tests..."
	docker-compose -f docker-compose.clean.yml exec backend python manage.py test
	docker-compose -f docker-compose.clean.yml exec frontend npm test

test-backend: ## Run backend tests
	docker-compose -f docker-compose.clean.yml exec backend python manage.py test

test-frontend: ## Run frontend tests
	docker-compose -f docker-compose.clean.yml exec frontend npm test

# Code quality commands
lint: ## Run linting for all services
	@echo "Running linting..."
	docker-compose -f docker-compose.clean.yml exec backend flake8 .
	docker-compose -f docker-compose.clean.yml exec frontend npm run lint

format: ## Format code for all services
	@echo "Formatting code..."
	docker-compose -f docker-compose.clean.yml exec backend black .
	docker-compose -f docker-compose.clean.yml exec frontend npm run format

# Cleanup commands
clean: ## Clean up Docker resources
	@echo "Cleaning up Docker resources..."
	docker-compose -f docker-compose.clean.yml down -v
	docker system prune -f

clean-volumes: ## Clean up Docker volumes
	@echo "Cleaning up Docker volumes..."
	docker-compose -f docker-compose.clean.yml down -v

clean-images: ## Clean up Docker images
	@echo "Cleaning up Docker images..."
	docker-compose -f docker-compose.clean.yml down --rmi all

# Health checks
health: ## Check health of all services
	@echo "Checking service health..."
	@curl -f http://localhost:8000/health/ && echo "Backend: OK" || echo "Backend: FAIL"
	@curl -f http://localhost:8001/health && echo "AI Service: OK" || echo "AI Service: FAIL"
	@curl -f http://localhost/ && echo "Frontend: OK" || echo "Frontend: FAIL"

# Backup and restore
backup-db: ## Backup database
	@echo "Creating database backup..."
	docker-compose -f docker-compose.clean.yml exec db pg_dump -U gymbro gymbro_db > backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "Database backup created!"

restore-db: ## Restore database from backup (usage: make restore-db FILE=backup.sql)
	@echo "Restoring database from $(FILE)..."
	docker-compose -f docker-compose.clean.yml exec -T db psql -U gymbro -d gymbro_db < $(FILE)
	@echo "Database restored!"

# Monitoring
monitor: ## Monitor resource usage
	@echo "Monitoring resource usage..."
	docker stats

# Security
security-scan: ## Run security scan
	@echo "Running security scan..."
	docker-compose -f docker-compose.clean.yml exec backend safety check
	docker-compose -f docker-compose.clean.yml exec frontend npm audit

# Documentation
docs: ## Generate documentation
	@echo "Generating documentation..."
	docker-compose -f docker-compose.clean.yml exec backend python manage.py generate_docs
	@echo "Documentation generated in docs/ directory"

# Quick start
quick-start: setup build-dev up-dev ## Quick start for new developers
	@echo "Quick start completed!"
	@echo "Please wait for services to start up..."
	@sleep 30
	@make health