from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class MultiTaskResNet(nn.Module):
    """ResNet50 + три линейные головы (тип / цвет / стиль)."""

    def __init__(
        self,
        num_type: int,
        num_color: int,
        num_style: int,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        feat = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head_type = nn.Linear(feat, num_type)
        self.head_color = nn.Linear(feat, num_color)
        self.head_style = nn.Linear(feat, num_style)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.backbone(x)
        return {
            "type": self.head_type(z),
            "color": self.head_color(z),
            "style": self.head_style(z),
        }
