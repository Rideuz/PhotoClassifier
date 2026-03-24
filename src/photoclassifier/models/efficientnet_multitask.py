from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B2_Weights,
    EfficientNet_B3_Weights,
    efficientnet_b0,
    efficientnet_b2,
    efficientnet_b3,
)


class MultiTaskEfficientNet(nn.Module):
    """EfficientNet + три линейные головы (type / color / style)."""

    def __init__(
        self,
        num_type: int,
        num_color: int,
        num_style: int,
        variant: str = "b0",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        variant = variant.lower()
        if variant == "b0":
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = efficientnet_b0(weights=weights)
        elif variant == "b2":
            weights = EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = efficientnet_b2(weights=weights)
        elif variant == "b3":
            weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = efficientnet_b3(weights=weights)
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {variant}")

        feat = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
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

