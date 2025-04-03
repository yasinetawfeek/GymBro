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

# Copy build output from builder stage to Nginx's html directory
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

# Start Nginx in the foreground
CMD ["nginx", "-g", "daemon off;"]