# Use an official Python image as the base
FROM python:3.10-slim 
#according to stackoverflow do not use 3.9 because 
#ERROR: Could not find a version that satisfies the requirement contourpy==1.3.1
#make sure you have 3.10 at least in order to pypass this error

# Environment variables to prevent Python from writing pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Install system dependencies (e.g., for compiling any Python dependencies)
RUN apt-get update && \
    apt-get install -y build-essential libpq-dev && \
    apt-get clean


    
# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Copy the AI service files
COPY . .


# Expose port 8000 for the Flask app
EXPOSE 8001


#start running the flask application which is solely responsible for AI API requests [training,predicting]
CMD ["python", "app/main.py"]