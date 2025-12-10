import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIP(nn.Module):
    def __init__(self, image_encoder, text_encoder, d_e, d_i, d_t):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.fc_image = nn.Linear(d_i, d_e)
        self.fc_text = nn.Linear(d_t, d_e)
        self.temperature = nn.Parameter(torch.tensor(0.7))

    def forward(self, image, text):
        image_feature = self.image_encoder(image)
        text_feature = self.text_encoder(text)
        image_embed = F.normalize(self.fc_image(image_feature), dim=-1)
        text_embed = F.normalize(self.fc_text(text_feature), dim=-1)
        logits = image_embed @ text_embed.T / self.temperature.exp()
        return logits, image_embed, text_embed
