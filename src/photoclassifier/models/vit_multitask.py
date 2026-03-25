from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import (
    ViT_B_16_Weights,
    ViT_B_32_Weights,
    vit_b_16,
    vit_b_32,
)

class MultiTaskViT(nn.Module):
    """Vision Transformer (ViT) + три линейные головы (type / color / style)."""

    def __init__(
        self,
        num_type: int,
        num_color: int,
        num_style: int,
        variant: str = "b_16",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        variant = variant.lower()
        if variant == "b_16":
            weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = vit_b_16(weights=weights)
        elif variant == "b_32":
            weights = ViT_B_32_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = vit_b_32(weights=weights)
        else:
            raise ValueError(f"Unsupported ViT variant: {variant}")

        feat = backbone.hidden_dim
        # Убираем классификатор (он находится в backbone.heads)
        backbone.heads = nn.Identity()
        self.backbone = backbone
        
        self.head_type = nn.Linear(feat, num_type)
        self.head_color = nn.Linear(feat, num_color)
        self.head_style = nn.Linear(feat, num_style)
        self.variant = variant

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.backbone(x)
        return {
            "type": self.head_type(z),
            "color": self.head_color(z),
            "style": self.head_style(z),
        }
