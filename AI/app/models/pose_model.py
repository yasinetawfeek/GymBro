"""
Pose correction model definition and utilities.
"""
import torch
import torch.nn as nn


class EnhancedPoseModel(nn.Module):
    """Enhanced pose correction model with improved architecture."""
    
    def __init__(self, input_dim=37, hidden_dim=512, output_dim=36):
        super().__init__()
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
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass through the model."""
        # Model outputs values in range [-0.1, 0.1]
        return self.net(x) * 0.1