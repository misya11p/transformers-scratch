import torch.nn as nn

from .layers import (
    SinusoidalPositionalEmbedding,
    MultiHeadAttention,
    FeedForwardNetwork,
)


class VanillaTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, is_causal=False):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads, is_causal)
        self.ffn = FeedForwardNetwork(d_model, d_ff, activation=nn.ReLU())
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        x = self.dropout(self.mha(x))
        x = self.norm1(x + res)
        res = x
        x = self.dropout(self.ffn(x))
        x = self.norm2(x + res)
        return x


class VanillaTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len=1024,
        n_layers=6,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        dropout=0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = SinusoidalPositionalEmbedding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.transformer_layers = nn.Sequential(*[
            VanillaTransformerLayer(d_model, n_heads, d_ff, dropout, is_causal=True)
            for _ in range(n_layers)
        ])
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pe(x)
        x = self.dropout(x)
        x = self.transformer_layers(x)
        x = self.fc(x)
        return x
