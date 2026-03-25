from __future__ import annotations

import torch
import torch.nn as nn
import open_clip

class MultiTaskCLIP(nn.Module):
    """
    CLIP Image Encoder + три линейные головы (type / color / style).
    Использует библиотеку open_clip.
    """

    def __init__(
        self,
        num_type: int,
        num_color: int,
        num_style: int,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        
        # Загружаем модель CLIP
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained if pretrained else None
        )
        
        # Нам нужен только энкодер картинок
        self.backbone = model.visual
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Размер выходного признака (embedding)
        # У ViT-B/32 это обычно 512, у ViT-L/14 это 768
        feat_dim = model.visual.output_dim
        
        self.head_type = nn.Linear(feat_dim, num_type)
        self.head_color = nn.Linear(feat_dim, num_color)
        self.head_style = nn.Linear(feat_dim, num_style)
        
        self.model_name = model_name
        self.pretrained_dataset = pretrained

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Получаем эмбеддинг картинки из CLIP
        z = self.backbone(x)
        
        # Пропускаем через классификационные головы
        return {
            "type": self.head_type(z),
            "color": self.head_color(z),
            "style": self.head_style(z),
        }
