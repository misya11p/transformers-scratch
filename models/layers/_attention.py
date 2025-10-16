import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(-1)

    def forward(self, q, k, v):
        scale = q.size(-1) ** 0.5
        scores = q @ k.mT
        causal_mask = self._get_causal_mask(q.shape[-2]).to(q.device)
        scores = scores.masked_fill(causal_mask, -torch.inf)
        weights = self.softmax(scores / scale)
        out = weights @ v
        return out

    def _get_causal_mask(self, seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert not d_model % n_heads, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.attention = Attention()
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        h = self.n_heads
        b, l, dm = x.shape
        dk = dm // h
        q = self.w_q(x).view(b, l, h, dk).permute(0, 2, 1, 3)
        k = self.w_k(x).view(b, l, h, dk).permute(0, 2, 1, 3)
        v = self.w_v(x).view(b, l, h, dk).permute(0, 2, 1, 3)
        heads = self.attention(q, k, v).permute(0, 2, 1, 3)
        out = self.w_o(heads.reshape(b, l, dm))
        return out
