# Stage 1: Build the application using Node with npm
FROM node:22-alpine as builder

# Set working directory
WORKDIR /app

# Update npm to version 11.2.0 (as mentioned in the GitHub release)
RUN npm install -g npm@11.2.0

# Copy package files and install dependencies using npm with legacy-peer-deps flag
COPY package.json package-lock.json ./
RUN npm install --legacy-peer-deps

# Copy the rest of the application code
COPY . .

# Build the app (make sure your package.json has a build script, e.g., "vite build")
RUN npm run build

# Stage 2: Serve the app using Nginx
FROM nginx:stable-alpine

# Remove default Nginx website
RUN rm -rf /usr/share/nginx/html/*

# Install curl and wait-for-it script
RUN apk add --no-cache curl bash
ADD https://raw.githubusercontent.com/vishnubob/wait-for-it/master/wait-for-it.sh /usr/local/bin/wait-for-it
RUN chmod +x /usr/local/bin/wait-for-it

# Copy build output from builder stage to Nginx's html directory
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Create startup script
RUN echo '#!/bin/sh\n\
wait-for-it backend:8000 -t 60\n\
wait-for-it ai:8001 -t 60\n\
exec nginx -g "daemon off;"\n\
' > /docker-entrypoint.sh && chmod +x /docker-entrypoint.sh

# Expose port 80
EXPOSE 80

ENTRYPOINT ["/docker-entrypoint.sh"]