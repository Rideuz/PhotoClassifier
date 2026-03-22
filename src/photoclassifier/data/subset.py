from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def stratified_subsample_indices(y: np.ndarray, n_target: int, seed: int) -> np.ndarray:
    """
    Случайная подвыборка фиксированного размера с сохранением пропорций классов y
    (стратификация по меткам категории одежды).
    """
    n = y.shape[0]
    if n_target >= n:
        return np.arange(n, dtype=np.int64)
    if n_target < 2:
        return np.array([0], dtype=np.int64)

    X_dummy = np.zeros((n, 1), dtype=np.float32)
    try:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_target, random_state=seed)
        train_ix, _ = next(sss.split(X_dummy, y))
        return np.sort(np.asarray(train_ix, dtype=np.int64))
    except ValueError:
        # Слишком мало объектов на класс для sklearn — чистая случайная подвыборка
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(n, size=n_target, replace=False))


def apply_dataset_subset(
    paths: list[str],
    y_type: np.ndarray,
    attr_mat: np.ndarray,
    max_samples: int | None,
    strategy: str,
    seed: int,
) -> tuple[list[str], np.ndarray, np.ndarray, dict]:
    """
    strategy:
      - stratified — случайная подвыборка с пропорциями классов по y_type;
      - first_n — первые N записей после фильтра по диску (как в ранней версии скрипта).
    """
    n_full = len(paths)
    meta: dict = {
        "full_after_disk_filter": n_full,
        "subset_strategy": strategy,
        "subset_requested": max_samples,
        "subset_applied": False,
    }
    if max_samples is None or max_samples >= n_full:
        meta["subset_size_used"] = n_full
        return paths, y_type, attr_mat, meta

    meta["subset_applied"] = True
    if strategy == "first_n":
        sel = np.arange(min(max_samples, n_full), dtype=np.int64)
    elif strategy == "stratified":
        sel = stratified_subsample_indices(y_type, max_samples, seed)
    else:
        raise ValueError(f"Неизвестная стратегия подвыборки: {strategy}")

    meta["subset_size_used"] = int(sel.shape[0])
    new_paths = [paths[i] for i in sel]
    return new_paths, y_type[sel], attr_mat[sel], meta
