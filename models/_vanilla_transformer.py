import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.softmax = nn.Softmax(-1)
        self.scale = d ** 0.5

    def forward(self, q, k, v):
        scores = q @ k.mT
        causal_mask = self._get_causal_mask(q.shape[-2]).to(q.device)
        scores = scores.masked_fill(causal_mask, -torch.inf)
        weights = self.softmax(scores / self.scale)
        out = weights @ v
        return out

    def _get_causal_mask(self, seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, bias=True):
        super().__init__()
        self.n_heads = n_heads
        assert not d_model % n_heads, "d_model must be divisible by n_heads"
        d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.attention = Attention(d_k)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        queries = torch.stack(q.chunk(self.n_heads, -1), 0)
        keys = torch.stack(k.chunk(self.n_heads, -1), 0)
        values = torch.stack(v.chunk(self.n_heads, -1), 0)
        heads = self.attention(queries, keys, values).unbind(0)
        out = self.w_o(torch.cat(heads, -1))
        return out


class FeedFowardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc2(self.relu(self.fc1(x)))
        return out


class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedFowardNetwork(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        z1 = self.mha(x)
        z1 = self.dropout(z1)
        z1 = self.norm1(z1 + x)
        z2 = self.ffn(z1)
        z2 = self.dropout(z2)
        z2 = self.norm2(z2 + z1)
        return z2


class VanillaTransformer(nn.Module):
    def __init__(self, vocab_size, max_len, n_layers, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = nn.Parameter(torch.rand((max_len, d_model)))
        self.transformer_layers = nn.Sequential(*[
            TransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x).to(x.device)
        seq_len = x.shape[1]
        x = x + self.pe[:seq_len, :]
        x = self.transformer_layers(x)
        x = self.fc(x)
        return x
