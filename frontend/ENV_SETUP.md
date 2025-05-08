# Environment Variable Setup

This project uses environment variables to configure the API URL. This allows for different configurations in development and production environments.

## Setting Up Environment Variables

### Development

1. Create a `.env` file in the root of the frontend directory with the following content:

```
VITE_API_URL=http://localhost:8000
VITE_AI_URL=http://localhost:8001
```

2. Restart your development server for the changes to take effect.

### Production

In production environments, you can set the `VITE_API_URL` and `VITE_AI_URL` environment variables to point to your production APIs.

## How It Works

The application uses a configuration file (`src/config.js`) that reads the environment variables:

```javascript
export const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
export const AI_URL = import.meta.env.VITE_AI_URL || 'http://localhost:8001';
```

This means:
- If the environment variables are set, they will be used
- If not, they fall back to the default localhost URLs

## Docker Environment

When running the application in Docker, you can set the environment variables in your docker-compose.yml or Dockerfile. 