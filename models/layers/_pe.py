import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        assert max_len % 2 == 0, "max_len must be even"
        pos = torch.arange(max_len)
        dim = torch.arange(0, d_model, 2)
        pe = torch.zeros(max_len, d_model)
        pe[:, ::2] = torch.sin(pos.unsqueeze(-1) / 10000 ** (dim / d_model))
        pe[:, 1::2] = torch.cos(pos.unsqueeze(-1) / 10000 ** (dim / d_model))
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(-2)]
