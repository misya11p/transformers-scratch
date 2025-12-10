import torch
import torch.nn as nn
import torchvision.transforms as transforms

from .layers import LearnablePositionalEmbedding
from .gpt2 import GPT2TransformerLayer


class ImageEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, hidden_size, n_channels=3):
        super().__init__()
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        self.resize = transforms.Resize(image_size)
        self.flatten = nn.Flatten(2, -1)
        self.fc = nn.Linear(n_channels * patch_size * patch_size, hidden_size)
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, img):
        img = self.resize(img)
        z = self.unfold(img).mT
        z = self.flatten(z)
        z = self.fc(z)
        return z


class VisionTransformerEncoder(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        hidden_size,
        n_layers,
        mlp_size,
        n_heads,
    ):
        super().__init__()
        self.embedding = ImageEmbedding(image_size, patch_size, hidden_size)
        self.pe = LearnablePositionalEmbedding(
            hidden_size,
            max_len=(image_size // patch_size) ** 2 + 1
        )
        self.transformer_layers = nn.Sequential(*[
            GPT2TransformerLayer(hidden_size, n_heads, mlp_size, is_causal=False)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.cls = nn.Parameter(torch.randn(hidden_size))

    def forward(self, x):
        x = self.embedding(x)
        x = torch.cat([self.cls.repeat(x.size(0), 1, 1), x], dim=1)
        x = self.pe(x)
        x = self.transformer_layers(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        n_classes,
        image_size,
        patch_size,
        hidden_size,
        n_layers,
        mlp_size,
        n_heads,
    ):
        super().__init__()
        self.encoder = VisionTransformerEncoder(
            image_size,
            patch_size,
            hidden_size,
            n_layers,
            mlp_size,
            n_heads,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        x = self.encoder(x)[:, 0]
        x = self.norm(x)
        x = self.mlp(x)
        return x