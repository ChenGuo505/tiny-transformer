import torch
import torch.nn as nn
import torch.optim as optim
from transformer import Transformer
from utils import generate_mask

def train():
  # Hyperparameters
  src_vocab_size = 10000
  tgt_vocab_size = 10000
  d_model = 512
  num_heads = 8
  d_ff = 2048
  num_layers_enc = 6
  num_layers_dec = 6
  dropout = 0.1
  max_seq_len = 5000
  batch_size = 32
  num_epochs = 10

  # Initialize model, loss function, and optimizer
  model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_layers_enc, num_layers_dec, dropout, max_seq_len)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  for epoch in range(num_epochs):
    model.train()
    # Dummy data for demonstration (replace with actual data loading)
    src = torch.randint(0, src_vocab_size, (batch_size, max_seq_len))  # (batch_size, seq_len_src)
    tgt_input = torch.randint(0, tgt_vocab_size, (batch_size, max_seq_len))  # (batch_size, seq_len_tgt)
    tgt_output = torch.randint(0, tgt_vocab_size, (batch_size, max_seq_len))  # (batch_size, seq_len_tgt)

    # Generate masks
    src_mask = None
    tgt_mask = generate_mask(tgt_input.size(1)).to(tgt_input.device)  # (seq_len_tgt, seq_len_tgt)
    cross_mask = None

    # Forward pass
    output = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask, cross_mask=cross_mask)  # (batch_size, seq_len_tgt, tgt_vocab_size)

    # Compute loss and backpropagate
    loss = criterion(output.view(-1, tgt_vocab_size), tgt_output.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

if __name__ == "__main__":
  train()