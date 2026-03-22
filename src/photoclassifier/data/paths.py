from __future__ import annotations

from pathlib import Path


def scan_img_highres_keys(data_root: Path, image_subdir: str) -> set[str]:
    """
    Возвращает множество относительных ключей вида 'img/Category/file.jpg',
    соответствующих файлам в data_root/image_subdir.
    """
    base = data_root / image_subdir
    if not base.is_dir():
        raise FileNotFoundError(f"Нет папки изображений: {base}")
    keys: set[str] = set()
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    for cat in base.iterdir():
        if not cat.is_dir():
            continue
        for f in cat.iterdir():
            if f.suffix.lower() not in exts:
                continue
            keys.add(f"img/{cat.name}/{f.name}")
    return keys
