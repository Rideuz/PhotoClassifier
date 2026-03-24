from __future__ import annotations

from collections import Counter

import numpy as np


def build_split_stratify_array(
    y_type: np.ndarray,
    y_style: np.ndarray | None,
    mode: str,
    min_style_freq: int = 5,
) -> np.ndarray:
    """
    Метки только для train_test_split(..., stratify=...).

    - type: как раньше — пропорции 50 категорий одежды.
    - type_coarse_style: тип + «сгрублённый» стиль: все стили с частотой < min_style_freq
      относятся к одному псевдоклассу (только для разбиения, не меняет y_style в обучении).
      Это слегка выравнивает попадание «хвоста» стилей в val/test, но не гарантирует
      каждый редкий класс в каждом сплите (для этого нужна итеративная стратификация).
    """
    y_type = np.asarray(y_type, dtype=np.int64)
    if mode == "type" or y_style is None:
        return y_type

    if mode != "type_coarse_style":
        raise ValueError(f"Неизвестный режим стратификации: {mode}")

    y_style = np.asarray(y_style, dtype=np.int64)
    c = Counter(int(x) for x in y_style)
    rare_id = int(y_style.max()) + 1
    ys = np.array(
        [int(s) if c[int(s)] >= min_style_freq else rare_id for s in y_style],
        dtype=np.int64,
    )
    n_buck = int(ys.max()) + 1
    return y_type * n_buck + ys


def filter_indices_by_style_frequency(
    y_style: np.ndarray,
    min_freq: int,
) -> np.ndarray:
    """Индексы образцов, у которых класс стиля встречается в выборке >= min_freq раз."""
    y_style = np.asarray(y_style, dtype=np.int64)
    c = Counter(int(x) for x in y_style)
    return np.array([c[int(s)] >= min_freq for s in y_style], dtype=bool)


def split_stratify_stats(
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    name: str,
) -> dict:
    all_u = set(np.unique(y).tolist())
    tr = set(np.unique(y[train_idx]).tolist())
    va = set(np.unique(y[val_idx]).tolist())
    te = set(np.unique(y[test_idx]).tolist())
    return {
        "attribute": name,
        "num_classes_total": len(all_u),
        "classes_in_train": len(tr),
        "classes_in_val": len(va),
        "classes_in_test": len(te),
        "missing_in_val": len(all_u - va),
        "missing_in_test": len(all_u - te),
    }
