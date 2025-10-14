import torch
import torch.nn as nn

from ._attentions import MultiHeadAttention
from ._ffn import FeedForwardNetwork


class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
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
