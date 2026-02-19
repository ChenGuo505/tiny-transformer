import torch
import torch.nn as nn

class FeedForwardNet(nn.Module):
  def __init__(self, d_model, d_ff, dropout=0.1):
    super().__init__()
    self.fc1 = nn.Linear(d_model, d_ff)
    self.fc2 = nn.Linear(d_ff, d_model)
    self.dropout = nn.Dropout(dropout)
    self.layer_norm = nn.LayerNorm(d_model)

  def forward(self, x):
    # x: (batch_size, seq_len, d_model)
    output = self.fc1(x)
    output = torch.relu(output)
    output = self.dropout(output)
    output = self.fc2(output)
    output = self.layer_norm(x + output)  # Residual connection
    return output