import torch.nn as nn

from .layers import (
    LearnablePositionalEmbedding,
    MultiHeadAttention,
    FeedForwardNetwork,
)


class GPT2TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, is_causal=False):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads, is_causal)
        self.ffn = FeedForwardNetwork(d_model, d_ff, activation=nn.GELU())
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        x = self.norm1(x)
        x = self.dropout(self.mha(x)) + res
        res = x
        x = self.norm2(x)
        x = self.dropout(self.ffn(x)) + res
        return x


class GPT2Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len,
        n_layers,
        d_model,
        n_heads,
        d_ff,
        dropout,
        is_causal=False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = LearnablePositionalEmbedding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.transformer_layers = nn.Sequential(*[
            GPT2TransformerLayer(d_model, n_heads, d_ff, dropout, is_causal)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        x = self.embedding(x)
        x = self.pe(x)
        x = self.dropout(x)
        x = self.transformer_layers(x)
        return x


class GPT2(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len=1024,
        n_layers=12,
        d_model=768,
        n_heads=12,
        d_ff=3072,
        dropout=0.1,
    ):
        super().__init__()
        self.encoder = GPT2Encoder(
            vocab_size,
            max_len,
            n_layers,
            d_model,
            n_heads,
            d_ff,
            dropout,
            is_causal=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.norm(x)
        x = self.fc(x) # For clarity, tied-embedding is not implemented
        return x
