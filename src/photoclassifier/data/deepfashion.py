from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# DeepFashion может содержать отдельные очень большие изображения.
# Не блокируем загрузку по лимиту PIL и вручную ограничиваем размер ниже.
Image.MAX_IMAGE_PIXELS = None
MAX_SAFE_IMAGE_SIDE = 4096
RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR


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
        img = Image.open(p)
        # Защита от экстремально больших изображений: уменьшаем до безопасного размера
        # до применения torchvision-аугментаций.
        if max(img.size) > MAX_SAFE_IMAGE_SIDE:
            img.thumbnail((MAX_SAFE_IMAGE_SIDE, MAX_SAFE_IMAGE_SIDE), RESAMPLE_BILINEAR)
        img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {
            "image": img,
            "y_type": int(self.y_type[idx]),
            "y_color": int(self.y_color[idx]),
            "y_style": int(self.y_style[idx]),
            "path": self.paths[idx],
        }


class TransformSubsetDataset(Dataset):
    """
    Подмножество по индексам + отдельный transform.
    Класс объявлен на уровне модуля, чтобы DataLoader (num_workers>0) мог pickle на Windows.
    """

    def __init__(self, base: DeepFashionMultiTaskDataset, indices: np.ndarray, transform) -> None:
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        j = int(self.indices[i])
        item = self.base[j]
        item = dict(item)
        item["image"] = self.transform(item["image"])
        return item
