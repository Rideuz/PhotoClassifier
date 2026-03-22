from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def make_stratified_splits(
    y_stratify: np.ndarray,
    indices: np.ndarray | None,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if indices is None:
        indices = np.arange(len(y_stratify), dtype=np.int64)
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio должны суммироваться в 1")

    test_size = test_ratio
    val_fraction_of_trainval = val_ratio / (train_ratio + val_ratio)

    try:
        tv_idx, te_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=y_stratify[indices],
        )
        tr_idx, va_idx = train_test_split(
            tv_idx,
            test_size=val_fraction_of_trainval,
            random_state=seed,
            stratify=y_stratify[tv_idx],
        )
    except ValueError:
        # Редкие классы: fallback без стратификации
        tv_idx, te_idx = train_test_split(
            indices, test_size=test_size, random_state=seed, shuffle=True
        )
        tr_idx, va_idx = train_test_split(
            tv_idx, test_size=val_fraction_of_trainval, random_state=seed, shuffle=True
        )
    return np.sort(tr_idx), np.sort(va_idx), np.sort(te_idx)


def save_splits(path: Path, train: np.ndarray, val: np.ndarray, test: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, train=train, val=val, test=test)


def load_splits(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(path)
    return z["train"], z["val"], z["test"]
