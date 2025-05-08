# Use a smaller CPU-only PyTorch image 
FROM python:3.10-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV BACKEND_URL=http://backend:8000

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and other requirements
RUN apt-get update && \
    apt-get install -y python3-opencv libgl1 && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean
    
# Install PyTorch CPU version (much smaller than CUDA version)
RUN pip install --no-cache-dir torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu

# Copy and install Python dependencies excluding PyTorch (already installed)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the AI service files
COPY . .

# Expose port for the Flask app
EXPOSE 8001

# Start running the Flask application
CMD ["python", "app/main.py"]