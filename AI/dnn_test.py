import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import os

# Constants
max_pose_noise = 0.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model class
class EnhancedPoseModel(nn.Module):
    def __init__(self, input_dim=37, hidden_dim=512, output_dim=36):
        super().__init__()
        
        # Main network architecture
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
        
        # Initialize weights properly
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights with Kaiming normal and zero biases"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        return self.net(x) * max_pose_noise  # Scale tanh output to [-0.1, 0.1] range

def plot_comparison(true, pred, title="Predictions vs True Values"):
    """Enhanced visualization with heatmap-style connecting lines"""
    plt.figure(figsize=(14, 6))
    
    # Calculate errors and normalize for coloring
    errors = np.abs(true - pred)
    max_error = np.max(errors) if np.max(errors) > 0 else 1  # Avoid division by zero
    normalized_errors = errors / max_error
    
    # Create a colormap (red for large errors, green for small)
    cmap = plt.get_cmap('RdYlGn_r')  # Reversed so red=bad, green=good
    colors = cmap(normalized_errors)
    
    # Create indices for plotting
    indices = np.arange(len(true))
    
    # Main scatter plot
    plt.subplot(1, 2, 1)
    sc_true = plt.scatter(indices, true, label='True', alpha=0.8, s=25, color='blue', zorder=3)
    sc_pred = plt.scatter(indices, pred, label='Predicted', alpha=0.8, s=25, marker='x', color='red', zorder=3)
    
    # Add connecting lines with heat colors
    for i in indices:
        plt.plot([i, i], [true[i], pred[i]], 
                color=colors[i], 
                alpha=0.6, 
                linewidth=2 + 2*normalized_errors[i],  # Thicker lines for larger errors
                zorder=1)
    
    # Add colorbar for error magnitude
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_error))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), pad=0.01)
    cbar.set_label('Absolute Error', rotation=270, labelpad=15)
    
    plt.xlabel("Feature Index")
    plt.ylabel("Value")
    plt.title("Value Comparison with Error Heatlines")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Error distribution plot
    plt.subplot(1, 2, 2)
    hist = plt.hist(errors, bins=30, color='green', alpha=0.7, density=True)
    
    # Add KDE plot
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(errors)
    x_vals = np.linspace(0, max_error, 100)
    plt.plot(x_vals, kde(x_vals), color='darkgreen', linewidth=2)
    
    plt.xlabel("Prediction Error (Absolute)")
    plt.ylabel("Density")
    plt.title(f"Error Distribution\nMAE: {np.mean(errors):.4f} | Max Error: {max_error:.4f}")
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    
    # Add overall statistics as text
    stats_text = (f"Mean Error: {np.mean(errors):.4f}\n"
                 f"Median Error: {np.median(errors):.4f}\n"
                 f"Std Dev: {np.std(errors):.4f}\n"
                 f"R² Score: {1 - np.sum(errors**2)/np.sum((true - np.mean(true))**2):.4f}")
    
    plt.gcf().text(0.95, 0.5, stats_text, 
                  bbox=dict(facecolor='white', alpha=0.5), 
                  verticalalignment='center')
    
    plt.show()

def main():
    # Initialize the model with the same architecture
    model = EnhancedPoseModel(input_dim=37, hidden_dim=512, output_dim=36).to(device)
    
    # Load the model weights
    model_path = 'data/best_model.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Set model to evaluation mode
    model.eval()
    
    # Create a sample input (random data for testing)
    # In a real scenario, this would be your actual input data
    sample_input = torch.rand(1, 37, dtype=torch.float32).to(device)
    print(f"Sample input shape: {sample_input.shape}")
    
    # Make a prediction
    with torch.no_grad():
        prediction = model(sample_input)
        print(f"Prediction shape: {prediction.shape}")
    
    # Convert to numpy for display
    input_np = sample_input[0].cpu().numpy()
    prediction_np = prediction[0].cpu().numpy()
    
    print("Sample input values:")
    print(input_np)
    print("\nPrediction values:")
    print(prediction_np)
    
    # Visualize the prediction (comparing with random "true" values for demonstration)
    # In a real scenario, you would compare with actual ground truth
    # Here we just use the input as a stand-in for "true" values (for demonstration only)
    mock_target = input_np[:36]  # Assuming output_dim=36
    plot_comparison(mock_target, prediction_np, title="Sample Prediction Visualization")
    
    print("Test completed successfully!")

if __name__ == "__main__":
    main() 