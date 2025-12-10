import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, is_causal=False):
        super().__init__()
        self.softmax = nn.Softmax(-1)
        self.is_causal = is_causal

    def forward(self, q, k, v):
        scale = q.size(-1) ** 0.5
        scores = q @ k.mT
        if self.is_causal:
            causal_mask = self._get_causal_mask(q.shape[-2]).to(q.device)
            scores = scores.masked_fill(causal_mask, -torch.inf)
        weights = self.softmax(scores / scale)
        out = weights @ v
        return out

    def _get_causal_mask(self, seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, is_causal=False):
        super().__init__()
        assert not d_model % n_heads, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.attention = Attention(is_causal)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        qs = torch.stack(q.chunk(self.n_heads, -1), 0)
        ks = torch.stack(k.chunk(self.n_heads, -1), 0)
        vs = torch.stack(v.chunk(self.n_heads, -1), 0)
        heads = self.attention(qs, ks, vs).unbind(0)
        out = self.w_o(torch.cat(heads, -1))
        return out
