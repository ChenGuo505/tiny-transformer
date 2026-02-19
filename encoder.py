import torch.nn as nn
import math
from attention import MultiHeadAttention
from ffn import FeedForwardNet
from position import PositionalEncoding

class EncoderLayer(nn.Module):
  def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
    super().__init__()
    self.self_attn = MultiHeadAttention(num_heads, d_model, dropout)
    self.ffn = FeedForwardNet(d_model, d_ff, dropout)

  def forward(self, x, mask=None):
    # x: (batch_size, seq_len, d_model)
    output, _ = self.self_attn(x, x, x, mask)
    output = self.ffn(output)
    return output

class Encoder(nn.Module):
  def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1, max_seq_len=5000):
    super().__init__()
    # Initialize embedding and positional encoding
    self.embedding = nn.Embedding(vocab_size, d_model)
    self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

    # Create a stack of encoder layers
    self.layers = nn.ModuleList([
      EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
    ])

  def forward(self, x, mask=None):
    # x: (batch_size, seq_len)
    x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)  # Scale embedding
    x = self.pos_encoding(x)

    for layer in self.layers:
      x = layer(x, mask)

    return x  # (batch_size, seq_len, d_model)