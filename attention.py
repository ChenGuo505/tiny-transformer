import torch
import torch.nn as nn
import math

class Attention(nn.Module):
  def __init__(self, dropout=0.1):
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.softmax = nn.Softmax(dim=-1)

  def forward(self, Q, K, V, mask=None):
    # Q: (batch_size, num_heads, seq_len_q, d_k)
    # K: (batch_size, num_heads, seq_len_k, d_k)
    # V: (batch_size, num_heads, seq_len_v, d_v)
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
      scores = scores.masked_fill(mask == 0, float('-inf'))

    attn = self.softmax(scores)
    attn = self.dropout(attn)

    # attn: (batch_size, num_heads, seq_len_q, seq_len_k)
    # output: (batch_size, num_heads, seq_len_q, d_v)
    output = torch.matmul(attn, V)
    return output, attn

class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, d_model, dropout=0.1):
    super().__init__()
    # num_heads: number of attention heads
    # d_model: dimension of the model
    # Ensure d_model is divisible by num_heads
    assert d_model % num_heads == 0
    self.d_k = d_model // num_heads
    self.num_heads = num_heads

    # Define linear layers for Q, K, V
    self.W_q = nn.Linear(d_model, d_model)
    self.W_k = nn.Linear(d_model, d_model)
    self.W_v = nn.Linear(d_model, d_model)

    # Final linear layer to combine heads
    self.fc = nn.Linear(d_model, d_model)

    # Attention module
    self.attention = Attention(dropout)

    # Dropout layer
    self.dropout = nn.Dropout(dropout)

    # Layer normalization
    self.layer_norm = nn.LayerNorm(d_model)

  def forward(self, query, key, value, mask=None):
    batch_size = query.size(0)
    # query, key, value: (batch_size, seq_len, d_model)
    # Apply linear transformations
    Q = self.W_q(query)
    K = self.W_k(key)
    V = self.W_v(value)

    # Q, K, V: (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k)
    # Split into multiple heads
    Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
    K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
    V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    # Apply attention mechanism
    output, attn = self.attention(Q, K, V, mask)

    # output: (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, d_model)
    # Concatenate heads and apply final linear layer
    output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
    output = self.fc(output)

    # Apply dropout and layer normalization
    output = self.dropout(output)
    output = self.layer_norm(output + query)  # Residual connection

    return output, attn