import React, { useState, useEffect} from 'react';
import { AlertCircle, CheckCircle, Upload, Play, Download, RotateCcw, TrendingUp, BarChart2 } from 'lucide-react';
import NavBar from '../components/Navbar';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const ModelManagement = () => {
const navigate = useNavigate();

const [isDarkMode, setIsDarkMode] = useState(false);
  
// Check system preference on initial load
useEffect(() => {
  const prefersDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
  setIsDarkMode(prefersDarkMode);
  
}, [navigate]);

// Update dark mode class on body
useEffect(() => {
  if (isDarkMode) {
    document.body.classList.add('dark');
  } else {
    document.body.classList.remove('dark');
  }
}, [isDarkMode]);

const toggleDarkMode = () => {
  setIsDarkMode(!isDarkMode);
};

  const [dataset, setDataset] = useState('');
  const [model, setModel] = useState('workout-classification');
  const [hyperparameters, setHyperparameters] = useState({
    learningRate: 0.01,
    epochs: 100,
    batchSize: 32,
    optimizer: 'adam',
  });
  const [trainingStatus, setTrainingStatus] = useState('idle'); // idle, training, complete, error
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingResults, setTrainingResults] = useState(null);
  const [modelVersions, setModelVersions] = useState([]);
  const [modelPerformance, setModelPerformance] = useState(null);
  const [activeTab, setActiveTab] = useState('training');
  const [error, setError] = useState(null);

  // Model options
  const modelOptions = [
    { value: 'workout-classification', label: 'Workout Classification' },
    { value: 'muscle-activation', label: 'Muscle Group Activation' },
    { value: 'displacement-values', label: 'Displacement Estimation Values' },
  ];

  // Optimizer options
  const optimizerOptions = [
    { value: 'adam', label: 'Adam' },
    { value: 'sgd', label: 'Stochastic Gradient Descent' },
    { value: 'rmsprop', label: 'RMSProp' },
    { value: 'adagrad', label: 'AdaGrad' },
  ];
  
  // Fetch model versions and performance data on component mount
  useEffect(() => {
    const fetchModelData = async () => {
      try {
        const token = localStorage.getItem('access_token');
        
        // Fetch model versions
        const versionsResponse = await axios.get('http://localhost:8000/api/models/versions/', {
          headers: { Authorization: `Bearer ${token}` }
        });
        
        // Fetch performance metrics for the latest model
        if (versionsResponse.data.length > 0) {
          const latestModelId = versionsResponse.data[0].id;
          const performanceResponse = await axios.get(`http://localhost:8000/api/models/${latestModelId}/performance/`, {
            headers: { Authorization: `Bearer ${token}` }
          });
          
          setModelVersions(versionsResponse.data);
          setModelPerformance(performanceResponse.data);
        }
      } catch (err) {
        console.error("Error fetching model data:", err);
        setError("Failed to fetch model data. Please try again.");
      }
    };
    
    fetchModelData();
  }, []);

  // Handle dataset URL change
  const handleDatasetChange = (e) => {
    setDataset(e.target.value);
  };

  // Handle hyperparameter changes
  const handleHyperparameterChange = (param, value) => {
    setHyperparameters({
      ...hyperparameters,
      [param]: param === 'optimizer' ? value : parseFloat(value),
    });
  };

  // Start training
  const startTraining = async () => {
    if (!dataset) {
      alert('Please enter a dataset URL');
      return;
    }
    
    setError(null);
    setTrainingStatus('training');
    setTrainingProgress(0);
    
    try {
      const token = localStorage.getItem('access_token');
      
      // Make API call to start training
      const trainingRequest = {
        dataset_url: dataset,
        model_type: model,
        hyperparameters: hyperparameters
      };
      
      // Set up progress updates
      const progressInterval = setInterval(() => {
        setTrainingProgress(prev => {
          if (prev >= 99) {
            clearInterval(progressInterval);
            return 99;
          }
          return prev + Math.random() * 5;
        });
      }, 500);
      
      // Call training API
      const response = await axios.post('http://localhost:8000/api/models/train/', trainingRequest, {
        headers: { Authorization: `Bearer ${token}` }
      });
      
      clearInterval(progressInterval);
      setTrainingProgress(100);
      setTrainingStatus('complete');
      
      // Format results
      setTrainingResults({
        accuracy: response.data.accuracy + '%',
        loss: response.data.loss.toFixed(4),
        trainTime: response.data.training_time + ' seconds'
      });
      
      // Refresh model versions and performance
      const versionsResponse = await axios.get('http://localhost:8000/api/models/versions/', {
        headers: { Authorization: `Bearer ${token}` }
      });
      
      if (versionsResponse.data.length > 0) {
        const latestModelId = versionsResponse.data[0].id;
        const performanceResponse = await axios.get(`http://localhost:8000/api/models/${latestModelId}/performance/`, {
          headers: { Authorization: `Bearer ${token}` }
        });
        
        setModelVersions(versionsResponse.data);
        setModelPerformance(performanceResponse.data);
      }
      
    } catch (err) {
      console.error("Error during model training:", err);
      setTrainingStatus('error');
      setError(err.response?.data?.message || "An error occurred during training");
      clearInterval(progressInterval);
    }
  };

  // Reset training
  const resetTraining = () => {
    setTrainingStatus('idle');
    setTrainingProgress(0);
    setTrainingResults(null);
    setError(null);
  };
  
  // Update model parameters
  const updateModelParameters = async (modelId, parameters) => {
    try {
      const token = localStorage.getItem('access_token');
      
      await axios.patch(`http://localhost:8000/api/models/${modelId}/update/`, parameters, {
        headers: { Authorization: `Bearer ${token}` }
      });
      
      // Refresh model versions
      const versionsResponse = await axios.get('http://localhost:8000/api/models/versions/', {
        headers: { Authorization: `Bearer ${token}` }
      });
      
      setModelVersions(versionsResponse.data);
      
      return true;
    } catch (err) {
      console.error("Error updating model parameters:", err);
      setError(err.response?.data?.message || "Failed to update model parameters");
      return false;
    }
  };

  return (
    <section className={`overflow-scroll fixed inset-0 ${isDarkMode ? 'bg-gradient-to-br from-gray-800 to-indigo-500':'bg-gradient-to-br from-gray-100 to-indigo-500'}`}>
    <NavBar isDarkMode={isDarkMode} />
    <main>
      <div className="max-w-6xl mx-auto mt-2 mb-20">
        {/* Tabs */}
        <div className="flex space-x-2 mb-4">
          <button
            onClick={() => setActiveTab('training')}
            className={`px-4 py-2 rounded-lg font-medium transition ${
              activeTab === 'training'
                ? isDarkMode 
                  ? 'bg-white text-indigo-900' 
                  : 'bg-indigo-600 text-white'
                : isDarkMode
                  ? 'text-white hover:bg-white/20'
                  : 'text-gray-800 hover:bg-gray-100'
            }`}
          >
            Model Training
          </button>
          
          <button
            onClick={() => setActiveTab('performance')}
            className={`px-4 py-2 rounded-lg font-medium transition ${
              activeTab === 'performance'
                ? isDarkMode 
                  ? 'bg-white text-indigo-900' 
                  : 'bg-indigo-600 text-white'
                : isDarkMode
                  ? 'text-white hover:bg-white/20'
                  : 'text-gray-800 hover:bg-gray-100'
            }`}
          >
            Performance Metrics
          </button>
        </div>
      
        {activeTab === 'training' && (
          <div className="bg-white rounded-lg shadow-md overflow-hidden">
            <div className="bg-indigo-600 p-6">
              <h1 className="text-2xl font-bold text-white">Model Training</h1>
              <p className="text-blue-100 mt-2">Train new ML models with custom parameters</p>
            </div>

            <div className="p-6">
              {error && (
                <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-6">
                  <div className="flex">
                    <AlertCircle className="w-5 h-5 text-red-500 mr-2" />
                    <p className="text-red-700">{error}</p>
                  </div>
                </div>
              )}
            
              <div className="mb-8">
                <h2 className="text-xl font-semibold mb-4">1. Choose Dataset</h2>
                <div className="flex flex-col">
                  <input 
                    placeholder="Enter URL to dataset" 
                    value={dataset} 
                    onChange={handleDatasetChange}
                    className="border border-gray-300 rounded p-2 w-full"
                  />
                  <p className="mt-2 text-sm text-gray-500">
                    Enter a valid dataset URL (e.g., https://huggingface.co/datasets/averrous/workout)
                  </p>
                </div>
              </div>

              <div className="mb-8">
                <h2 className="text-xl font-semibold mb-4">2. Model</h2>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {modelOptions.map((modelOption) => (
                    <div
                      key={modelOption.value}
                      className={`border rounded-lg p-4 cursor-pointer ${
                        model === modelOption.value ? 'border-blue-500 bg-blue-50' : 'border-gray-200'
                      }`}
                      onClick={() => setModel(modelOption.value)}
                    >
                      <h3 className="font-medium">{modelOption.label}</h3>
                    </div>
                  ))}
                </div>
              </div>

              <div className="mb-8">
                <h2 className="text-xl font-semibold mb-4">3. Configure Hyperparameters</h2>
                <div className="bg-gray-50 rounded-lg p-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Learning Rate</label>
                      <input
                        type="number"
                        className="border border-gray-300 rounded p-2 w-full"
                        value={hyperparameters.learningRate}
                        min="0.0001"
                        max="1"
                        step="0.001"
                        onChange={(e) => handleHyperparameterChange('learningRate', e.target.value)}
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Epochs</label>
                      <input
                        type="number"
                        className="border border-gray-300 rounded p-2 w-full"
                        value={hyperparameters.epochs}
                        min="1"
                        max="1000"
                        onChange={(e) => handleHyperparameterChange('epochs', e.target.value)}
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Batch Size</label>
                      <input
                        type="number"
                        className="border border-gray-300 rounded p-2 w-full"
                        value={hyperparameters.batchSize}
                        min="1"
                        max="256"
                        onChange={(e) => handleHyperparameterChange('batchSize', e.target.value)}
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">Optimizer</label>
                      <select
                        className="border border-gray-300 rounded p-2 w-full"
                        value={hyperparameters.optimizer}
                        onChange={(e) => handleHyperparameterChange('optimizer', e.target.value)}
                      >
                        {optimizerOptions.map(option => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                </div>
              </div>

              {/* Training Section */}
              <div className="mb-8">
                <h2 className="text-xl font-semibold mb-4">4. Train Your Model</h2>
                {trainingStatus === 'idle' ? (
                  <button
                    className="bg-green-500 hover:bg-green-600 text-white py-3 px-6 rounded-lg flex items-center"
                    onClick={startTraining}
                    disabled={!dataset}
                  >
                    <Play size={20} className="mr-2" />
                    Start Training
                  </button>
                ) : (
                  <div>
                    <div className="mb-4">
                      <div className="flex justify-between text-sm mb-1">
                        <span>{trainingStatus === 'complete' ? 'Training complete!' : trainingStatus === 'error' ? 'Training failed' : 'Training in progress...'}</span>
                        <span>{Math.round(trainingProgress)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2.5">
                        <div
                          className={`h-2.5 rounded-full ${trainingStatus === 'error' ? 'bg-red-500' : 'bg-green-500'}`}
                          style={{ width: `${trainingProgress}%` }}
                        ></div>
                      </div>
                    </div>

                    {trainingStatus === 'complete' && trainingResults && (
                      <div className="bg-green-50 border border-green-200 rounded-lg p-4 mb-4">
                        <h3 className="font-medium text-green-800 mb-2">Training Results</h3>
                        <div className="grid grid-cols-3 gap-2">
                          <div className="bg-white rounded p-3 shadow-sm">
                            <p className="text-xs text-gray-500">Accuracy</p>
                            <p className="font-bold text-lg">{trainingResults.accuracy}</p>
                          </div>
                          <div className="bg-white rounded p-3 shadow-sm">
                            <p className="text-xs text-gray-500">Loss</p>
                            <p className="font-bold text-lg">{trainingResults.loss}</p>
                          </div>
                          <div className="bg-white rounded p-3 shadow-sm">
                            <p className="text-xs text-gray-500">Training Time</p>
                            <p className="font-bold text-lg">{trainingResults.trainTime}</p>
                          </div>
                        </div>
                        <div className="mt-4 flex space-x-2">
                          <button className="bg-blue-500 hover:bg-blue-600 text-white py-2 px-4 rounded flex items-center">
                            <Download size={16} className="mr-1" />
                            Download Model
                          </button>
                          <button
                            className="bg-gray-200 hover:bg-gray-300 text-gray-800 py-2 px-4 rounded flex items-center"
                            onClick={resetTraining}
                          >
                            <RotateCcw size={16} className="mr-1" />
                            Train Again
                          </button>
                        </div>
                      </div>
                    )}

                    {trainingStatus === 'error' && (
                      <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-4">
                        <div className="flex items-center">
                          <AlertCircle className="text-red-500 w-5 h-5 mr-2" />
                          <h3 className="font-medium text-red-800">Training Failed</h3>
                        </div>
                        <p className="mt-2 text-sm text-red-700">{error}</p>
                        <button
                          className="mt-4 bg-gray-200 hover:bg-gray-300 text-gray-800 py-2 px-4 rounded flex items-center"
                          onClick={resetTraining}
                        >
                          <RotateCcw size={16} className="mr-1" />
                          Try Again
                        </button>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
        
        {activeTab === 'performance' && (
          <div className="bg-white rounded-lg shadow-md overflow-hidden">
            <div className="bg-indigo-600 p-6">
              <h1 className="text-2xl font-bold text-white">Model Performance</h1>
              <p className="text-blue-100 mt-2">Track and analyze model performance metrics</p>
            </div>
            
            <div className="p-6">
              {/* Version Selection */}
              <div className="mb-8">
                <h2 className="text-xl font-semibold mb-4">Model Versions</h2>
                {modelVersions.length === 0 ? (
                  <p className="text-gray-500">No model versions available</p>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {modelVersions.map((version) => (
                      <div
                        key={version.id}
                        className="border rounded-lg p-4 hover:shadow-md transition"
                      >
                        <div className="flex justify-between items-start">
                          <h3 className="font-medium text-lg">{version.model_type}</h3>
                          <span className={`px-2 py-1 text-xs rounded-full ${
                            version.is_active ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
                          }`}>
                            {version.is_active ? 'Active' : 'Archived'}
                          </span>
                        </div>
                        <p className="text-sm text-gray-500 mt-1">v{version.version_number}</p>
                        <div className="mt-3 text-sm text-gray-600">
                          <p>Created: {new Date(version.created_at).toLocaleDateString()}</p>
                          <p className="mt-1">By: {version.created_by}</p>
                        </div>
                        <div className="mt-4 flex space-x-2">
                          <button 
                            className="text-indigo-600 hover:text-indigo-800 text-sm font-medium"
                            onClick={async () => {
                              const performanceResponse = await axios.get(`http://localhost:8000/api/models/${version.id}/performance/`, {
                                headers: { Authorization: `Bearer ${localStorage.getItem('access_token')}` }
                              });
                              setModelPerformance(performanceResponse.data);
                            }}
                          >
                            View Performance
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              
              {/* Performance Metrics */}
              {modelPerformance && (
                <div className="mb-8">
                  <h2 className="text-xl font-semibold mb-4">Performance Metrics</h2>
                  
                  <div className="bg-gray-50 rounded-lg p-6">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                      <div className="bg-white p-4 rounded-lg shadow-sm">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm text-gray-500">Accuracy</span>
                          <span className="p-1.5 rounded-full bg-green-100">
                            <CheckCircle className="w-3 h-3 text-green-600" />
                          </span>
                        </div>
                        <div className="text-2xl font-bold">{(modelPerformance.accuracy * 100).toFixed(2)}%</div>
                        <div className="mt-2 text-xs flex items-center">
                          <TrendingUp className="w-3 h-3 text-green-600 mr-1" />
                          <span className="text-green-600">+2.4% from previous model</span>
                        </div>
                      </div>
                      
                      <div className="bg-white p-4 rounded-lg shadow-sm">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm text-gray-500">Precision</span>
                          <span className="p-1.5 rounded-full bg-green-100">
                            <BarChart2 className="w-3 h-3 text-green-600" />
                          </span>
                        </div>
                        <div className="text-2xl font-bold">{(modelPerformance.precision * 100).toFixed(2)}%</div>
                        <div className="mt-2 text-xs flex items-center">
                          <TrendingUp className="w-3 h-3 text-green-600 mr-1" />
                          <span className="text-green-600">+1.8% from previous model</span>
                        </div>
                      </div>
                      
                      <div className="bg-white p-4 rounded-lg shadow-sm">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm text-gray-500">Recall</span>
                          <span className="p-1.5 rounded-full bg-blue-100">
                            <BarChart2 className="w-3 h-3 text-blue-600" />
                          </span>
                        </div>
                        <div className="text-2xl font-bold">{(modelPerformance.recall * 100).toFixed(2)}%</div>
                        <div className="mt-2 text-xs flex items-center">
                          <TrendingUp className="w-3 h-3 text-green-600 mr-1" />
                          <span className="text-green-600">+3.2% from previous model</span>
                        </div>
                      </div>
                      
                      <div className="bg-white p-4 rounded-lg shadow-sm">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm text-gray-500">F1 Score</span>
                          <span className="p-1.5 rounded-full bg-indigo-100">
                            <BarChart2 className="w-3 h-3 text-indigo-600" />
                          </span>
                        </div>
                        <div className="text-2xl font-bold">{(modelPerformance.f1_score * 100).toFixed(2)}%</div>
                        <div className="mt-2 text-xs flex items-center">
                          <TrendingUp className="w-3 h-3 text-green-600 mr-1" />
                          <span className="text-green-600">+2.0% from previous model</span>
                        </div>
                      </div>
                    </div>
                    
                    {/* Confusion Matrix (Placeholder) */}
                    <div className="bg-white p-4 rounded-lg shadow-sm">
                      <h3 className="font-medium mb-4">Confusion Matrix</h3>
                      <div className="h-64 bg-gray-100 rounded flex items-center justify-center">
                        <p className="text-gray-500">Confusion Matrix Visualization</p>
                        {/* In a real implementation, you would render the confusion matrix here */}
                      </div>
                    </div>
                  </div>
                </div>
              )}
              
              {/* Model Parameters */}
              {modelPerformance && (
                <div className="mb-8">
                  <h2 className="text-xl font-semibold mb-4">Model Parameters</h2>
                  
                  <div className="bg-gray-50 rounded-lg p-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                          Threshold
                        </label>
                        <input
                          type="number"
                          className="border border-gray-300 rounded p-2 w-full"
                          defaultValue="0.5"
                          min="0.1"
                          max="0.9"
                          step="0.05"
                        />
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                          Batch Size
                        </label>
                        <input
                          type="number"
                          className="border border-gray-300 rounded p-2 w-full"
                          defaultValue="32"
                          min="1"
                          max="256"
                        />
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                          Inference Optimization
                        </label>
                        <select className="border border-gray-300 rounded p-2 w-full">
                          <option value="speed">Optimize for Speed</option>
                          <option value="accuracy">Optimize for Accuracy</option>
                          <option value="balanced" selected>Balanced</option>
                        </select>
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                          Model Description
                        </label>
                        <textarea
                          className="border border-gray-300 rounded p-2 w-full"
                          rows="2"
                          defaultValue="Workout classification model trained on MediaPipe pose data."
                        ></textarea>
                      </div>
                    </div>
                    
                    <div className="mt-6">
                      <button
                        className="bg-indigo-600 hover:bg-indigo-700 text-white py-2 px-4 rounded"
                        onClick={async () => {
                          // Get form values and update model parameters
                          // This is a simplified implementation
                          const success = await updateModelParameters(modelPerformance.model_id, {
                            threshold: 0.5,
                            batch_size: 32,
                            optimization: 'balanced',
                            description: 'Updated model parameters'
                          });
                          
                          if (success) {
                            alert('Model parameters updated successfully');
                          }
                        }}
                      >
                        Update Parameters
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </main>
    </section>
  );
};

export default ModelManagement;