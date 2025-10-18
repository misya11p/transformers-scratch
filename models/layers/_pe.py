import torch
import torch.nn as nn


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d, max_len=1024, base=10_000):
        super().__init__()
        assert max_len % 2 == 0, "max_len must be even"
        p = torch.arange(max_len)
        i = torch.arange(0, d, 2)
        pe = torch.zeros(max_len, d)
        pe[:, ::2] = torch.sin(p.unsqueeze(-1) / base ** (i / d))
        pe[:, 1::2] = torch.cos(p.unsqueeze(-1) / base ** (i / d))
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(-2)]


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        self.pe = nn.Parameter(torch.randn((max_len, d_model)))

    def forward(self, x):
        return x + self.pe[:x.size(-2)]


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d, max_len=4096, base=10_000):
        super().__init__()
        p = torch.arange(max_len)
        i = torch.arange(0, d, 2)
        theta = p.unsqueeze(-1) / (base ** (i / d))
        self.sin = torch.sin(theta)
        self.cos = torch.cos(theta)
        self.idx = torch.arange(d)
        self.d = d

    def _get_rotation_matrix(self, l):
        R = torch.zeros(l, self.d, self.d)
        R[:, self.idx, self.idx] = self.cos[:l].repeat_interleave(2, -1)
        R[:, self.idx[1::2], self.idx[::2]] = self.sin[:l]
        R[:, self.idx[::2], self.idx[1::2]] = -self.sin[:l]
        return R

    def forward(self, x):
        R = self._get_rotation_matrix(x.size(1)).to(x.device)
        x = torch.einsum("blhd,lvd->blhv", x, R)
        return x
