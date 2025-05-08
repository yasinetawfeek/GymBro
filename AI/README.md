# Real-time Workout Pose Correction

This application provides real-time pose correction for workouts using a deep neural network model.

## Setup

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Make sure the model file exists at `data/best_model.pth`.

3. In the frontend directory, install the required npm packages:

```bash
cd ../frontend
npm install
```

## Running the Application

1. Start the Flask backend:

```bash
cd app
python main.py
```

This will start the server at http://localhost:8001.

2. Start the React frontend:

```bash
cd ../frontend
npm run dev
```

This will start the frontend development server.

3. Open your browser and navigate to the frontend URL (typically http://localhost:3000).

## Endpoints

- WebSocket: Connect to `ws://localhost:8001` for real-time pose corrections.
  - Send: `pose_data` event with landmarks data
  - Receive: `pose_corrections` event with correction data

## How It Works

1. The frontend captures pose data using MediaPipe Pose detection.
2. Pose landmarks are sent to the server via WebSocket.
3. The server processes the landmarks through the DNN model, using the barbell bicep curl workout type.
4. The model returns pose corrections, which are sent back to the client.
5. The frontend visualizes the corrections as arrows, guiding the user to the correct pose. 

## Notebooks

The project is supported by three key Jupyter notebooks that document the machine learning development process:

1. **app/final_model_dnn.ipynb**: This notebook contains the development of the Deep Neural Network (DNN) model used for pose correction. It includes data preprocessing, model architecture design, training, and evaluation of the regression model that predicts optimal joint positions.

2. **app/muscle_group_classifer_notebook.ipynb**: This notebook demonstrates the development of a classifier that identifies which muscle groups are being targeted during a workout. The model analyzes pose landmarks to determine primary muscle engagement, which helps provide appropriate feedback.

3. **app/workout_classifer_final.ipynb**: This notebook implements a Long Short-Term Memory (LSTM) neural network that classifies entire workout sequences. It processes time-series pose data to identify which exercise is being performed, achieving high accuracy even with limited training data.

These notebooks provide detailed documentation of the AI development process and can be referenced to understand the technical implementation behind the application.

# AI Module

## Environment Variables

The AI service uses the following environment variables:

- `BACKEND_URL`: The URL of the backend API (default: http://localhost:8000)

## Running Locally

To run the AI service locally:

1. Create a `.env` file in the AI directory with the required environment variables:
   ```
   BACKEND_URL=http://localhost:8000
   ```

2. Install the requirements:
   ```
   pip install -r requirements.txt
   ```

3. Run the server:
   ```
   python app/main.py
   ```

## Running with Docker

When running with Docker, the environment variables are set in the Dockerfile, but can be overridden when starting the container:

```
docker run -p 8001:8001 -e BACKEND_URL=http://your-backend-url:8000 desd-ai-service
``` 