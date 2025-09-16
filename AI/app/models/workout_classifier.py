"""
Workout classification model definition and utilities.
"""
import torch
import torch.nn as nn


class LSTMWorkoutClassifier(nn.Module):
    """LSTM-based workout classifier with attention mechanism."""
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super(LSTMWorkoutClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers with increased dropout
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
        
        # Additional dropout after LSTM
        self.lstm_dropout = nn.Dropout(dropout + 0.1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 128)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        """Forward pass through the LSTM classifier."""
        # Ensure x has the right shape for sequences
        # x should be [batch_size, sequence_length, features]
        if len(x.shape) == 2:
            # If single frame, reshape to [batch, 1, features]
            x = x.unsqueeze(1)
        
        batch_size, seq_len, _ = x.size()
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))  # lstm_out: [batch_size, seq_len, hidden_size]
        
        # Apply additional dropout to LSTM output
        lstm_out = self.lstm_dropout(lstm_out)
        
        # Apply attention mechanism if sequence length > 1
        if seq_len > 1:
            attention_weights = self.attention(lstm_out)  # [batch_size, seq_len, 1]
            context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # [batch_size, hidden_size]
        else:
            # If single frame, just use the LSTM output directly
            context_vector = lstm_out.squeeze(1)
        
        # Dense layers
        out = self.fc1(context_vector)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out