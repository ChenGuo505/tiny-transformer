import torch.nn as nn
import math
from attention import MultiHeadAttention
from ffn import FeedForwardNet
from position import PositionalEncoding

class DecoderLayer(nn.Module):
  def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
    super().__init__()
    self.self_attn = MultiHeadAttention(num_heads, d_model, dropout)
    self.cross_attn = MultiHeadAttention(num_heads, d_model, dropout)
    self.ffn = FeedForwardNet(d_model, d_ff, dropout)

  def forward(self, x, enc_output, self_mask=None, cross_mask=None):
    # x: (batch_size, seq_len_dec, d_model)
    # enc_output: (batch_size, seq_len_enc, d_model)
    output, _ = self.self_attn(x, x, x, self_mask)
    output, _ = self.cross_attn(output, enc_output, enc_output, cross_mask)
    output = self.ffn(output)
    return output
  
class Decoder(nn.Module):
  def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1, max_seq_len=5000):
    super().__init__()
    # Initialize embedding and positional encoding
    self.embedding = nn.Embedding(vocab_size, d_model)
    self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

    # Create a stack of decoder layers
    self.layers = nn.ModuleList([
      DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
    ])

    # Final linear layer to project to vocab size
    self.fc_out = nn.Linear(d_model, vocab_size)

  def forward(self, x, enc_output, self_mask=None, cross_mask=None):
    # x: (batch_size, seq_len_dec)
    x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)  # Scale embedding
    x = self.pos_encoding(x)

    for layer in self.layers:
      x = layer(x, enc_output, self_mask, cross_mask)

    return self.fc_out(x)  # (batch_size, seq_len_dec, vocab_size)