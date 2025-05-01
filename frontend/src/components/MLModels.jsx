import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ChevronDown, ChevronUp, Edit, Check, X, Plus, Trash2,
  Save, RefreshCw, AlertCircle, Heart, Dumbbell, Activity
} from 'lucide-react';
import axios from 'axios';

const MLModels = ({ isDarkMode }) => {
  // State for models and UI
  const [models, setModels] = useState([]);
  const [modelTypes, setModelTypes] = useState([]);
  const [expandedTypes, setExpandedTypes] = useState([]);
  const [editingModelId, setEditingModelId] = useState(null);
  const [editedModel, setEditedModel] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Map model types to icons
  const getIconForType = (type) => {
    const iconMap = {
      'workout': Activity,
      'pose': Dumbbell,
      'muscle': Heart
    };
    return iconMap[type] || Activity;
  };
  
  // Toggle expansion of a model type
  const toggleTypeExpansion = (typeId) => {
    setExpandedTypes(prev => {
      if (prev.includes(typeId)) {
        return prev.filter(id => id !== typeId);
      } else {
        return [...prev, typeId];
      }
    });
  };
  
  // Start editing a model
  const startEditing = (model) => {
    // Format model data for editing form
    const formattedModel = {
      ...model,
      hyperparameters: {
        learningRate: model.learning_rate,
        epochs: model.epochs,
        batchSize: model.batch_size
      }
    };
    
    setEditingModelId(model.id);
    setEditedModel(formattedModel);
  };
  
  // Cancel editing
  const cancelEditing = () => {
    setEditingModelId(null);
    setEditedModel(null);
  };
  
  // Update edited model
  const updateEditedModel = (field, value) => {
    if (field.startsWith('hyperparameters.')) {
      const paramName = field.split('.')[1];
      setEditedModel({
        ...editedModel,
        hyperparameters: {
          ...editedModel.hyperparameters,
          [paramName]: value
        }
      });
    } else {
      setEditedModel({
        ...editedModel,
        [field]: value
      });
    }
  };
  
  // Function to fetch models from API 
  const fetchModels = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const token = localStorage.getItem('access_token');
      const response = await axios.get('http://localhost:8000/api/ml-models/', {
        headers: { Authorization: `Bearer ${token}` }
      });
      
      setModels(response.data);
      
      // Extract unique model types from the data
      const types = [...new Set(response.data.map(model => model.model_type))];
      
      // Create model types array with name and icon
      const typeData = types.map(type => ({
        id: type,
        name: response.data.find(m => m.model_type === type)?.model_type_display || type,
        icon: getIconForType(type)
      }));
      
      setModelTypes(typeData);
      
      // Expand first type if no types are expanded
      if (expandedTypes.length === 0 && types.length > 0) {
        setExpandedTypes([types[0]]);
      }
      
      setLoading(false);
    } catch (err) {
      console.error('Error fetching models:', err);
      
      // Check if this is a database schema error
      if (err.response?.data?.includes && err.response.data.includes('no such column')) {
        setError('Database schema error: The MLModel table is missing required columns. Please run proper migrations on the backend.');
        
        // Set placeholder data for development
        const placeholderTypes = ['workout', 'pose', 'muscle'];
        const typeData = placeholderTypes.map(type => ({
          id: type,
          name: type.charAt(0).toUpperCase() + type.slice(1),
          icon: getIconForType(type)
        }));
        
        setModelTypes(typeData);
        setModels([]);
        
        if (expandedTypes.length === 0) {
          setExpandedTypes([placeholderTypes[0]]);
        }
      } else {
        setError('Failed to load models. Please try again.');
      }
      
      setLoading(false);
    }
  };
  
  // Hook to fetch models on component mount
  useEffect(() => {
    fetchModels();
  }, []);

  // Save edited model
  const saveModel = () => {
    setLoading(true);
    
    const saveModelToAPI = async () => {
      try {
        const token = localStorage.getItem('access_token');
        
        // If updating hyperparameters specifically
        if (editedModel.id) {
          await axios.post(
            `http://localhost:8000/api/ml-models/${editedModel.id}/update_hyperparameters/`, 
            {
              learning_rate: editedModel.hyperparameters.learningRate,
              epochs: editedModel.hyperparameters.epochs,
              batch_size: editedModel.hyperparameters.batchSize
            },
            {
              headers: { Authorization: `Bearer ${token}` }
            }
          );
          
          // Also update general model properties
          await axios.patch(
            `http://localhost:8000/api/ml-models/${editedModel.id}/`, 
            {
              name: editedModel.name
            },
            {
              headers: { 
                Authorization: `Bearer ${token}`,
                'Content-Type': 'application/json'
              }
            }
          );
        } else {
          // Create new model
          await axios.post(
            'http://localhost:8000/api/ml-models/', 
            {
              name: editedModel.name,
              model_type: editedModel.type,
              learning_rate: editedModel.hyperparameters.learningRate,
              epochs: editedModel.hyperparameters.epochs,
              batch_size: editedModel.hyperparameters.batchSize,
              deployed: false
            },
            {
              headers: { 
                Authorization: `Bearer ${token}`,
                'Content-Type': 'application/json'
              }
            }
          );
        }
        
        // Refresh model list
        fetchModels();
        setEditingModelId(null);
        setEditedModel(null);
        setLoading(false);
      } catch (err) {
        console.error('Error saving model:', err);
        setError(`Failed to save model: ${err.response?.data?.message || err.message}`);
        setLoading(false);
      }
    };
    
    saveModelToAPI();
  };
  
  // Deploy a model
  const deployModel = (modelId) => {
    setLoading(true);
    
    const deployModelWithAPI = async () => {
      try {
        const token = localStorage.getItem('access_token');
        await axios.post(
          `http://localhost:8000/api/ml-models/${modelId}/deploy/`, 
          {},
          {
            headers: { Authorization: `Bearer ${token}` }
          }
        );
        
        // Refresh model list after deployment
        fetchModels();
      } catch (err) {
        console.error('Error deploying model:', err);
        setError(`Failed to deploy model: ${err.response?.data?.message || err.message}`);
        setLoading(false);
      }
    };
    
    deployModelWithAPI();
  };
  
  // Add a new model
  const addNewModel = (type) => {
    const newModel = {
      id: null,
      type,
      name: `New${type.charAt(0).toUpperCase() + type.slice(1)}Model`,
      hyperparameters: {
        learningRate: 0.001,
        epochs: 100,
        batchSize: 32
      },
      model_type: type
    };
    
    setEditedModel(newModel);
    setEditingModelId(null);
    setExpandedTypes([...new Set([...expandedTypes, type])]);
  };
  
  // Delete a model
  const deleteModel = (modelId) => {
    if (window.confirm("Are you sure you want to delete this model? This action cannot be undone.")) {
      setLoading(true);
      
      const deleteModelFromAPI = async () => {
        try {
          const token = localStorage.getItem('access_token');
          await axios.delete(
            `http://localhost:8000/api/ml-models/${modelId}/`, 
            {
              headers: { Authorization: `Bearer ${token}` }
            }
          );
          
          // Refresh model list after deletion
          fetchModels();
        } catch (err) {
          console.error('Error deleting model:', err);
          setError(`Failed to delete model: ${err.response?.data?.message || err.message}`);
          setLoading(false);
        }
      };
      
      deleteModelFromAPI();
    }
  };
  
  // Clear error
  const clearError = () => {
    setError(null);
  };
  
  return (
    <div className="space-y-8">
      <div className="flex justify-between items-center">
        <h2 className={`text-2xl font-light ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>
          <span className="text-purple-400 font-medium">ML Models</span> Management
        </h2>
        <button
          onClick={fetchModels}
          className={`p-2 rounded-lg ${
            isDarkMode 
              ? 'bg-purple-900/30 text-purple-400 hover:bg-purple-900/50'
              : 'bg-purple-100 text-purple-600 hover:bg-purple-200'
          }`}
          title="Refresh Models"
        >
          <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>
      
      {/* Error message */}
      <AnimatePresence>
        {error && (
          <motion.div 
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className={`${
              isDarkMode 
                ? 'bg-red-900/50 border border-red-800/50 text-red-100'
                : 'bg-red-50 border border-red-200 text-red-800'
            } p-4 rounded-lg flex items-center justify-between`}
          >
            <div className="flex items-center">
              <AlertCircle className="w-5 h-5 mr-2" />
              <span>{error}</span>
            </div>
            <button 
              onClick={clearError}
              className="p-1 rounded-full hover:bg-black/10"
            >
              <X className="w-4 h-4" />
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Loading state */}
      {loading && modelTypes.length === 0 && (
        <div className={`p-12 rounded-xl border text-center ${
          isDarkMode 
            ? 'bg-gray-800/50 border-gray-700 text-gray-300'
            : 'bg-white border-gray-200 text-gray-600'
        }`}>
          <RefreshCw className="w-8 h-8 mx-auto mb-4 animate-spin opacity-50" />
          <p>Loading model data...</p>
        </div>
      )}
      
      {/* No data state */}
      {!loading && modelTypes.length === 0 && (
        <div className={`p-12 rounded-xl border text-center ${
          isDarkMode 
            ? 'bg-gray-800/50 border-gray-700 text-gray-300'
            : 'bg-white border-gray-200 text-gray-600'
        }`}>
          <AlertCircle className="w-8 h-8 mx-auto mb-4 opacity-50" />
          <p className="mb-4">No model types found.</p>
          <button
            onClick={fetchModels}
            className={`px-4 py-2 rounded-lg ${
              isDarkMode 
                ? 'bg-purple-600 hover:bg-purple-700 text-white'
                : 'bg-purple-500 hover:bg-purple-600 text-white'
            }`}
          >
            <RefreshCw className="w-4 h-4 inline mr-2" />
            Refresh
          </button>
        </div>
      )}
      
      {/* Create new model form */}
      {editedModel && !editingModelId && (
        <div className={`p-6 rounded-xl border ${
          isDarkMode 
            ? 'bg-gray-800/50 border-gray-700'
            : 'bg-white border-gray-200'
        }`}>
          <h3 className={`text-lg font-medium mb-4 ${
            isDarkMode ? 'text-white' : 'text-gray-800'
          }`}>
            Create New Model
          </h3>
          
          <div className="space-y-4">
            {/* Basic info */}
            <div className={`p-4 rounded-lg ${
              isDarkMode 
                ? 'bg-gray-900/50'
                : 'bg-gray-50'
            }`}>
              <div className="grid gap-4 sm:grid-cols-2">
                <div>
                  <label className={`block text-sm font-medium mb-1 ${
                    isDarkMode ? 'text-gray-400' : 'text-gray-700'
                  }`}>
                    Model Name
                  </label>
                  <input
                    type="text"
                    value={editedModel.name}
                    onChange={(e) => updateEditedModel('name', e.target.value)}
                    className={`w-full px-3 py-2 rounded-lg ${
                      isDarkMode
                        ? 'bg-gray-800 border-gray-700 text-white'
                        : 'bg-white border-gray-300 text-gray-800'
                    } border focus:outline-none focus:ring-2 focus:ring-purple-500`}
                  />
                </div>
              </div>
            </div>
            
            {/* Hyperparameters */}
            <div>
              <h4 className={`font-medium mb-2 ${
                isDarkMode ? 'text-gray-300' : 'text-gray-700'
              }`}>
                Hyperparameters
              </h4>
              <div className={`p-4 rounded-lg ${
                isDarkMode 
                  ? 'bg-gray-900/50'
                  : 'bg-gray-50'
              }`}>
                <div className="grid gap-4 sm:grid-cols-3">
                  <div>
                    <label className={`block text-sm font-medium mb-1 ${
                      isDarkMode ? 'text-gray-400' : 'text-gray-700'
                    }`}>
                      Learning Rate
                    </label>
                    <input
                      type="number"
                      min="0.0001"
                      max="1"
                      step="0.0001"
                      value={editedModel.hyperparameters.learningRate}
                      onChange={(e) => updateEditedModel('hyperparameters.learningRate', parseFloat(e.target.value))}
                      className={`w-full px-3 py-2 rounded-lg ${
                        isDarkMode
                          ? 'bg-gray-800 border-gray-700 text-white'
                          : 'bg-white border-gray-300 text-gray-800'
                      } border focus:outline-none focus:ring-2 focus:ring-purple-500`}
                    />
                  </div>
                  
                  <div>
                    <label className={`block text-sm font-medium mb-1 ${
                      isDarkMode ? 'text-gray-400' : 'text-gray-700'
                    }`}>
                      Epochs
                    </label>
                    <input
                      type="number"
                      min="1"
                      step="1"
                      value={editedModel.hyperparameters.epochs}
                      onChange={(e) => updateEditedModel('hyperparameters.epochs', parseInt(e.target.value))}
                      className={`w-full px-3 py-2 rounded-lg ${
                        isDarkMode
                          ? 'bg-gray-800 border-gray-700 text-white'
                          : 'bg-white border-gray-300 text-gray-800'
                      } border focus:outline-none focus:ring-2 focus:ring-purple-500`}
                    />
                  </div>
                  
                  <div>
                    <label className={`block text-sm font-medium mb-1 ${
                      isDarkMode ? 'text-gray-400' : 'text-gray-700'
                    }`}>
                      Batch Size
                    </label>
                    <input
                      type="number"
                      min="1"
                      step="1"
                      value={editedModel.hyperparameters.batchSize}
                      onChange={(e) => updateEditedModel('hyperparameters.batchSize', parseInt(e.target.value))}
                      className={`w-full px-3 py-2 rounded-lg ${
                        isDarkMode
                          ? 'bg-gray-800 border-gray-700 text-white'
                          : 'bg-white border-gray-300 text-gray-800'
                      } border focus:outline-none focus:ring-2 focus:ring-purple-500`}
                    />
                  </div>
                </div>
              </div>
            </div>
            
            {/* Action buttons */}
            <div className="flex justify-end space-x-3">
              <button
                onClick={() => setEditedModel(null)}
                className={`px-4 py-2 rounded-lg ${
                  isDarkMode
                    ? 'bg-gray-700 hover:bg-gray-600 text-white'
                    : 'bg-gray-200 hover:bg-gray-300 text-gray-800'
                }`}
              >
                Cancel
              </button>
              
              <button
                onClick={saveModel}
                disabled={loading}
                className={`px-4 py-2 rounded-lg flex items-center ${
                  isDarkMode
                    ? 'bg-purple-600 hover:bg-purple-700 text-white'
                    : 'bg-purple-500 hover:bg-purple-600 text-white'
                } ${loading ? 'opacity-70' : ''}`}
              >
                {loading ? (
                  <>
                    <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                    Saving...
                  </>
                ) : (
                  <>
                    <Save className="w-4 h-4 mr-2" />
                    Save Model
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
      
      {/* Model types */}
      <div className="space-y-6">
        {modelTypes.map((type) => (
          <div 
            key={type.id}
            className={`rounded-xl border ${
              isDarkMode 
                ? 'bg-gray-800/50 border-gray-700'
                : 'bg-white border-gray-200'
            }`}
          >
            {/* Type header */}
            <div 
              className={`p-4 flex justify-between items-center cursor-pointer ${
                isDarkMode 
                  ? 'border-b border-gray-700'
                  : 'border-b border-gray-200'
              }`}
              onClick={() => toggleTypeExpansion(type.id)}
            >
              <div className="flex items-center space-x-3">
                <div className={`p-2 rounded-lg ${
                  isDarkMode 
                    ? 'bg-purple-900/30 text-purple-400'
                    : 'bg-purple-100 text-purple-600'
                }`}>
                  <type.icon className="w-5 h-5" />
                </div>
                <h3 className="text-lg font-medium">{type.name}</h3>
                <span className={`text-sm px-2 py-0.5 rounded-full ${
                  isDarkMode 
                    ? 'bg-gray-700 text-gray-300'
                    : 'bg-gray-100 text-gray-600'
                }`}>
                  {models.filter(m => m.model_type === type.id).length} models
                </span>
              </div>
              
              <div className="flex items-center space-x-4">
                <button 
                  onClick={(e) => {
                    e.stopPropagation();
                    addNewModel(type.id);
                  }}
                  className={`p-2 rounded-lg ${
                    isDarkMode 
                      ? 'bg-purple-900/30 text-purple-400 hover:bg-purple-900/50'
                      : 'bg-purple-100 text-purple-600 hover:bg-purple-200'
                  }`}
                >
                  <Plus className="w-4 h-4" />
                </button>
                {expandedTypes.includes(type.id) ? (
                  <ChevronUp className="w-5 h-5" />
                ) : (
                  <ChevronDown className="w-5 h-5" />
                )}
              </div>
            </div>
            
            {/* Models list */}
            <AnimatePresence>
              {expandedTypes.includes(type.id) && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.3 }}
                  className="overflow-hidden"
                >
                  <div className="divide-y divide-gray-700">
                    {models
                      .filter(model => model.model_type === type.id)
                      .map(model => (
                        <div 
                          key={model.id}
                          className="p-4"
                        >
                          {editingModelId === model.id ? (
                            // Editing form
                            <div className="space-y-4">
                              {/* Basic info */}
                              <div className={`p-4 rounded-lg ${
                                isDarkMode 
                                  ? 'bg-gray-900/50'
                                  : 'bg-gray-50'
                              }`}>
                                <div className="grid gap-4 sm:grid-cols-2">
                                  <div>
                                    <label className={`block text-sm font-medium mb-1 ${
                                      isDarkMode ? 'text-gray-400' : 'text-gray-700'
                                    }`}>
                                      Model Name
                                    </label>
                                    <input
                                      type="text"
                                      value={editedModel.name}
                                      onChange={(e) => updateEditedModel('name', e.target.value)}
                                      className={`w-full px-3 py-2 rounded-lg ${
                                        isDarkMode
                                          ? 'bg-gray-800 border-gray-700 text-white'
                                          : 'bg-white border-gray-300 text-gray-800'
                                      } border focus:outline-none focus:ring-2 focus:ring-purple-500`}
                                    />
                                  </div>
                                </div>
                              </div>
                              
                              {/* Hyperparameters */}
                              <div>
                                <h4 className={`font-medium mb-2 ${
                                  isDarkMode ? 'text-gray-300' : 'text-gray-700'
                                }`}>
                                  Hyperparameters
                                </h4>
                                <div className={`p-4 rounded-lg ${
                                  isDarkMode 
                                    ? 'bg-gray-900/50'
                                    : 'bg-gray-50'
                                }`}>
                                  <div className="grid gap-4 sm:grid-cols-3">
                                    <div>
                                      <label className={`block text-sm font-medium mb-1 ${
                                        isDarkMode ? 'text-gray-400' : 'text-gray-700'
                                      }`}>
                                        Learning Rate
                                      </label>
                                      <input
                                        type="number"
                                        min="0.0001"
                                        max="1"
                                        step="0.0001"
                                        value={editedModel.hyperparameters.learningRate}
                                        onChange={(e) => updateEditedModel('hyperparameters.learningRate', parseFloat(e.target.value))}
                                        className={`w-full px-3 py-2 rounded-lg ${
                                          isDarkMode
                                            ? 'bg-gray-800 border-gray-700 text-white'
                                            : 'bg-white border-gray-300 text-gray-800'
                                        } border focus:outline-none focus:ring-2 focus:ring-purple-500`}
                                      />
                                    </div>
                                    
                                    <div>
                                      <label className={`block text-sm font-medium mb-1 ${
                                        isDarkMode ? 'text-gray-400' : 'text-gray-700'
                                      }`}>
                                        Epochs
                                      </label>
                                      <input
                                        type="number"
                                        min="1"
                                        step="1"
                                        value={editedModel.hyperparameters.epochs}
                                        onChange={(e) => updateEditedModel('hyperparameters.epochs', parseInt(e.target.value))}
                                        className={`w-full px-3 py-2 rounded-lg ${
                                          isDarkMode
                                            ? 'bg-gray-800 border-gray-700 text-white'
                                            : 'bg-white border-gray-300 text-gray-800'
                                        } border focus:outline-none focus:ring-2 focus:ring-purple-500`}
                                      />
                                    </div>
                                    
                                    <div>
                                      <label className={`block text-sm font-medium mb-1 ${
                                        isDarkMode ? 'text-gray-400' : 'text-gray-700'
                                      }`}>
                                        Batch Size
                                      </label>
                                      <input
                                        type="number"
                                        min="1"
                                        step="1"
                                        value={editedModel.hyperparameters.batchSize}
                                        onChange={(e) => updateEditedModel('hyperparameters.batchSize', parseInt(e.target.value))}
                                        className={`w-full px-3 py-2 rounded-lg ${
                                          isDarkMode
                                            ? 'bg-gray-800 border-gray-700 text-white'
                                            : 'bg-white border-gray-300 text-gray-800'
                                        } border focus:outline-none focus:ring-2 focus:ring-purple-500`}
                                      />
                                    </div>
                                  </div>
                                </div>
                              </div>
                              
                              {/* Action buttons */}
                              <div className="flex justify-end space-x-3">
                                <button
                                  onClick={cancelEditing}
                                  className={`px-4 py-2 rounded-lg ${
                                    isDarkMode
                                      ? 'bg-gray-700 hover:bg-gray-600 text-white'
                                      : 'bg-gray-200 hover:bg-gray-300 text-gray-800'
                                  }`}
                                >
                                  Cancel
                                </button>
                                
                                <button
                                  onClick={saveModel}
                                  disabled={loading}
                                  className={`px-4 py-2 rounded-lg flex items-center ${
                                    isDarkMode
                                      ? 'bg-purple-600 hover:bg-purple-700 text-white'
                                      : 'bg-purple-500 hover:bg-purple-600 text-white'
                                  } ${loading ? 'opacity-70' : ''}`}
                                >
                                  {loading ? (
                                    <>
                                      <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                                      Saving...
                                    </>
                                  ) : (
                                    <>
                                      <Save className="w-4 h-4 mr-2" />
                                      Save Changes
                                    </>
                                  )}
                                </button>
                              </div>
                            </div>
                          ) : (
                            // Model display
                            <div className="flex flex-col sm:flex-row sm:items-center">
                              <div className="flex-grow">
                                <div className="flex items-center space-x-3 mb-2">
                                  <h4 className="font-semibold text-lg">{model.name}</h4>
                                  {model.deployed && (
                                    <span className={`text-xs px-2 py-0.5 rounded-full ${
                                      isDarkMode 
                                        ? 'bg-green-900/40 text-green-400'
                                        : 'bg-green-100 text-green-800'
                                    }`}>
                                      Deployed
                                    </span>
                                  )}
                                </div>
                                
                                <div className="grid gap-4 md:grid-cols-4">
                                 
                                  <div>
                                    <p className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                                      Learning Rate
                                    </p>
                                    <p className="font-medium">
                                      {model.learning_rate}
                                    </p>
                                  </div>
                                  <div>
                                    <p className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                                      Epochs
                                    </p>
                                    <p className="font-medium">
                                      {model.epochs}
                                    </p>
                                  </div>
                                  <div>
                                    <p className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                                      Batch Size
                                    </p>
                                    <p className="font-medium">
                                      {model.batch_size}
                                    </p>
                                  </div>
                                </div>
                              </div>
                              
                              <div className="flex items-center space-x-2 sm:pl-4 mt-4 sm:mt-0">
                                {!model.deployed && (
                                  <button
                                    onClick={() => deployModel(model.id)}
                                    disabled={loading}
                                    className={`p-2 rounded-lg ${
                                      isDarkMode
                                        ? 'bg-blue-900/30 text-blue-400 hover:bg-blue-900/50'
                                        : 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                                    } ${loading ? 'opacity-50' : ''}`}
                                    title="Deploy Model"
                                  >
                                    {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Check className="w-4 h-4" />}
                                  </button>
                                )}
                                
                                <button
                                  onClick={() => startEditing(model)}
                                  className={`p-2 rounded-lg ${
                                    isDarkMode
                                      ? 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                                  }`}
                                  title="Edit Model"
                                >
                                  <Edit className="w-4 h-4" />
                                </button>
                                
                                {!model.deployed && (
                                  <button
                                    onClick={() => deleteModel(model.id)}
                                    className={`p-2 rounded-lg ${
                                      isDarkMode
                                        ? 'bg-red-900/30 text-red-400 hover:bg-red-900/50'
                                        : 'bg-red-100 text-red-700 hover:bg-red-200'
                                    }`}
                                    title="Delete Model"
                                  >
                                    <Trash2 className="w-4 h-4" />
                                  </button>
                                )}
                              </div>
                            </div>
                          )}
                        </div>
                      ))}
                      
                    {models.filter(model => model.model_type === type.id).length === 0 && (
                      <div className={`p-6 text-center ${
                        isDarkMode ? 'text-gray-400' : 'text-gray-500'
                      }`}>
                        <p className="mb-2">No models found for this type</p>
                        <button
                          onClick={() => addNewModel(type.id)}
                          className={`px-4 py-2 rounded-lg ${
                            isDarkMode 
                              ? 'bg-purple-900/30 text-purple-400 hover:bg-purple-900/50'
                              : 'bg-purple-100 text-purple-600 hover:bg-purple-200'
                          }`}
                        >
                          <Plus className="w-4 h-4 inline mr-2" />
                          Add New Model
                        </button>
                      </div>
                    )}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        ))}
      </div>
    </div>
  );
};

export default MLModels; 