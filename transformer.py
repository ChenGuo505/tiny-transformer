import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from utils import generate_mask

class Transformer(nn.Module):
  def __init__(self,
               src_vocab_size,
               tgt_vocab_size,
               d_model=512,
               num_heads=8,
               d_ff=2048,
               num_layers_enc=6,
               num_layers_dec=6,
               dropout=0.1, max_seq_len=5000):
    super().__init__()
    self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, num_layers_enc, dropout, max_seq_len)
    self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, num_layers_dec, dropout, max_seq_len)

  def forward(self, src, tgt, src_mask=None, tgt_mask=None, cross_mask=None):
    enc_output = self.encoder(src, src_mask)
    dec_output = self.decoder(tgt, enc_output, tgt_mask, cross_mask)
    return dec_output  # (batch_size, seq_len_dec, tgt_vocab_size)

if __name__ == "__main__":
  # Example usage
  src_vocab_size = 10000
  tgt_vocab_size = 10000
  model = Transformer(src_vocab_size, tgt_vocab_size)

  src = torch.randint(0, src_vocab_size, (32, 20))  # (batch_size, seq_len_src)
  tgt = torch.randint(0, tgt_vocab_size, (32, 20))  # (batch_size, seq_len_tgt)

  tgt_mask = generate_mask(tgt.size(1)).to(tgt.device)  # (seq_len_tgt, seq_len_tgt)

  output = model(src, tgt, tgt_mask=tgt_mask)  # (batch_size, seq_len_tgt, tgt_vocab_size)
  print(output.shape)