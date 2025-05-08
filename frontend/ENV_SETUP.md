# Environment Variable Setup

This project uses environment variables to configure the API URL. This allows for different configurations in development and production environments.

## Setting Up Environment Variables

### Development

1. Create a `.env` file in the root of the frontend directory with the following content:

```
VITE_API_URL=http://localhost:8000
```

2. Restart your development server for the changes to take effect.

### Production

In production environments, you can set the `VITE_API_URL` environment variable to point to your production API.

## How It Works

The application uses a configuration file (`src/config.js`) that reads the environment variable:

```javascript
export const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
```

This means:
- If `VITE_API_URL` is set, it will be used
- If not, it falls back to 'http://localhost:8000'

## Docker Environment

When running the application in Docker, you can set the environment variable in your docker-compose.yml or Dockerfile. 