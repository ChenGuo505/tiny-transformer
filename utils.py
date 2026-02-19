import torch

def generate_mask(size):
  # Generate a square mask for the sequence. The masked positions are filled with float('-inf').
  mask = torch.triu(torch.ones(size=(size, size)), diagonal=1).bool()
  return mask == False  # Invert mask: True for allowed positions, False for masked positions