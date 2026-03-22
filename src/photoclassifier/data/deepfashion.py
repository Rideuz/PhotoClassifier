from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class DeepFashionMultiTaskDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        image_subdir: str,
        paths: list[str],
        y_type: np.ndarray,
        y_color: np.ndarray,
        y_style: np.ndarray,
        transform=None,
    ) -> None:
        self.data_root = data_root
        self.image_subdir = image_subdir
        self.paths = paths
        self.y_type = y_type
        self.y_color = y_color
        self.y_style = y_style
        self.transform = transform

    def _disk_path(self, rel_img_path: str) -> Path:
        # rel_img_path: img/Category/file.jpg -> img_highres/Category/file.jpg
        if not rel_img_path.startswith("img/"):
            raise ValueError(f"Неожиданный формат пути: {rel_img_path}")
        rel = f"{self.image_subdir}/" + rel_img_path[len("img/") :]
        return self.data_root / rel

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self._disk_path(self.paths[idx])
        img = Image.open(p).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {
            "image": img,
            "y_type": int(self.y_type[idx]),
            "y_color": int(self.y_color[idx]),
            "y_style": int(self.y_style[idx]),
            "path": self.paths[idx],
        }
