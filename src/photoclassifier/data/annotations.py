from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class AnnotationBundle:
    """Пути, размерности и списки индексов атрибутов для цвета/стиля."""

    paths: list[str]
    category_ids: np.ndarray  # 0..num_types-1
    attr_matrix: np.ndarray  # int8, shape (N, 1000), values in {-1,0,1}
    color_attr_indices: list[int]
    style_attr_indices: list[int]
    color_attr_names: list[str]
    style_attr_names: list[str]
    type_names: list[str]


def _parse_list_attr_cloth(path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        n = int(f.readline().strip())
        f.readline()
        for _ in range(n):
            line = f.readline()
            if not line:
                break
            line = line.strip()
            m = re.match(r"(.+?)\s+(\d)\s*$", line)
            if not m:
                raise ValueError(f"Не удалось разобрать строку атрибута: {line[:80]!r}")
            rows.append((m.group(1).strip(), int(m.group(2))))
    return rows


def _is_color_attribute(name: str) -> bool:
    """
    В coarse DeepFashion мало чистых цветовых тегов; фиксируем согласованный набор
    для головы «цвет» и групповых метрик в отчёте.
    """
    n = name.lower()
    if "colorblock" in n or "ombre" in n:
        return True
    toks = set(n.replace("-", " ").split())
    return bool(toks & {"pink", "red", "rose", "rainbow", "neon"})


def build_label_spaces(attr_cloth_path: Path) -> tuple[list[int], list[int], list[str], list[str]]:
    attrs = _parse_list_attr_cloth(attr_cloth_path)
    color_idx: list[int] = []
    style_idx: list[int] = []
    color_names: list[str] = []
    style_names: list[str] = []
    for i, (name, t) in enumerate(attrs):
        if _is_color_attribute(name):
            color_idx.append(i)
            color_names.append(name)
        elif t == 5:
            style_idx.append(i)
            style_names.append(name)
    return color_idx, style_idx, color_names, style_names


def _parse_category_cloth(path: Path) -> list[str]:
    names: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        n = int(f.readline().strip())
        f.readline()
        for _ in range(n):
            line = f.readline()
            if not line:
                break
            names.append(line.split()[0].strip())
    return names


def load_aligned_tables(
    anno_dir: Path,
    existing_paths: set[str],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Читает list_category_img и list_attr_img (одинаковый порядок строк в датасете).
    Возвращает только те изображения, чей путь (префикс img/) есть в existing_paths.
    """
    cat_file = anno_dir / "list_category_img.txt"
    attr_file = anno_dir / "list_attr_img.txt"

    kept_paths: list[str] = []
    cats: list[int] = []
    attrs_list: list[np.ndarray] = []

    with cat_file.open(encoding="utf-8", errors="replace") as fc, attr_file.open(
        encoding="utf-8", errors="replace"
    ) as fa:
        n_cat = int(fc.readline().strip())
        fc.readline()
        n_attr = int(fa.readline().strip())
        fa.readline()
        if n_cat != n_attr:
            raise ValueError("Число строк category и attribute не совпадает")

        for _ in range(n_cat):
            lc = fc.readline()
            la = fa.readline()
            if not lc or not la:
                break
            pc = lc.split()
            pa = la.split()
            path = pc[0]
            if path not in existing_paths:
                continue
            cat_lab = int(pc[1])
            feats = np.array(pa[1:1001], dtype=np.int8)
            if feats.shape[0] != 1000:
                raise ValueError(f"Ожидалось 1000 атрибутов, получено {feats.shape[0]} для {path}")
            kept_paths.append(path)
            cats.append(cat_lab - 1)
            attrs_list.append(feats)

    if not kept_paths:
        raise RuntimeError("Не найдено ни одного изображения: проверьте img_highres и аннотации.")

    category_ids = np.array(cats, dtype=np.int64)
    attr_matrix = np.stack(attrs_list, axis=0)
    return kept_paths, category_ids, attr_matrix


def build_color_coarse_names(color_attr_names: list[str]) -> list[str]:
    """Грубые цветовые группы для доп. метрики «близость/группа»."""

    def group_for(name: str) -> str:
        low = name.lower()
        if "colorblock" in low or "ombre" in low or "rainbow" in low:
            return "multicolor_pattern"
        if "neon" in low:
            return "neon_accent"
        if "pink" in low:
            return "pink"
        if any(x in low for x in ("red", "rose")):
            return "red_rose"
        return "other_color_tag"

    return [group_for(n) for n in color_attr_names]


def compute_multitask_labels(
    attr_matrix: np.ndarray,
    color_attr_indices: list[int],
    style_attr_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Одна метка на голову: среди положительных (+1) берём минимальный глобальный индекс
    атрибута (детерминированно). Если положительных нет — класс «none» (последний индекс).
    """
    n = attr_matrix.shape[0]
    n_color = len(color_attr_indices) + 1
    n_style = len(style_attr_indices) + 1

    color_map = {g: i for i, g in enumerate(color_attr_indices)}
    style_map = {g: i for i, g in enumerate(style_attr_indices)}

    color_y = np.full(n, n_color - 1, dtype=np.int64)
    style_y = np.full(n, n_style - 1, dtype=np.int64)

    for row_i in range(n):
        row = attr_matrix[row_i]
        c_pos = [j for j in color_attr_indices if row[j] == 1]
        if c_pos:
            best = min(c_pos)
            color_y[row_i] = color_map[best]
        s_pos = [j for j in style_attr_indices if row[j] == 1]
        if s_pos:
            best_s = min(s_pos)
            style_y[row_i] = style_map[best_s]

    return color_y, style_y
