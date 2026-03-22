from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)


def evaluate_multitask(
    preds: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
) -> dict:
    out: dict = {}
    for head in ("type", "color", "style"):
        y = targets[head]
        p = preds[head]
        out[f"{head}_accuracy"] = float(accuracy_score(y, p))
        out[f"{head}_macro_f1"] = float(
            f1_score(y, p, average="macro", zero_division=0)
        )
        out[f"{head}_micro_f1"] = float(
            f1_score(y, p, average="micro", zero_division=0)
        )
    return out


def evaluate_with_logits(
    logits_type: np.ndarray,
    logits_color: np.ndarray,
    logits_style: np.ndarray,
    y_type: np.ndarray,
    y_color: np.ndarray,
    y_style: np.ndarray,
    num_classes_type: int,
    top_k_type: int,
) -> dict:
    pt = logits_type.argmax(axis=1)
    pc = logits_color.argmax(axis=1)
    ps = logits_style.argmax(axis=1)
    metrics = evaluate_multitask(
        {"type": pt, "color": pc, "style": ps},
        {"type": y_type, "color": y_color, "style": y_style},
    )
    k = min(top_k_type, num_classes_type)
    if k >= 2 and logits_type.shape[0] > 0:
        metrics["type_topk_accuracy"] = float(
            top_k_accuracy_score(y_type, logits_type, k=k, labels=np.arange(num_classes_type))
        )
    else:
        metrics["type_topk_accuracy"] = float("nan")
    return metrics


def coarse_color_group_accuracy(
    y_color: np.ndarray,
    pred_color: np.ndarray,
    color_class_to_group: list[int],
    num_groups: int,
) -> float:
    """Accuracy по грубым цветовым группам (индексы групп 0..num_groups-1)."""
    yg = np.array([color_class_to_group[c] for c in y_color], dtype=np.int64)
    pg = np.array([color_class_to_group[c] for c in pred_color], dtype=np.int64)
    return float((yg == pg).mean())


def save_confusion_matrix_csv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    path: Path,
    max_labels: int = 64,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    u = np.unique(np.concatenate([y_true, y_pred]))
    u = np.sort(u)
    if u.size > max_labels:
        cm = confusion_matrix(y_true, y_pred)
        np.savez_compressed(path.with_suffix(".npz"), confusion_matrix=cm, labels=u)
        return
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    # CSV только для умеренного числа классов
    import csv

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + [labels[i] if i < len(labels) else str(i) for i in range(cm.shape[1])])
        for i, row in enumerate(cm):
            lab = labels[i] if i < len(labels) else str(i)
            w.writerow([lab] + list(map(int, row)))


def confusion_matrix_np(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    show_progress: bool = False,
    desc: str = "predict",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    model.eval()
    lt, lc, ls = [], [], []
    yt, yc, ys = [], [], []
    iterator = loader
    if show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(loader, desc=desc, leave=False)
        except ImportError:
            iterator = loader
    for batch in iterator:
        x = batch["image"].to(device, non_blocking=True)
        yt.append(batch["y_type"].detach().cpu().numpy())
        yc.append(batch["y_color"].detach().cpu().numpy())
        ys.append(batch["y_style"].detach().cpu().numpy())
        out = model(x)
        lt.append(out["type"].cpu().numpy())
        lc.append(out["color"].cpu().numpy())
        ls.append(out["style"].cpu().numpy())
    return (
        {
            "type": np.concatenate(yt),
            "color": np.concatenate(yc),
            "style": np.concatenate(ys),
        },
        {
            "type": np.vstack(lt),
            "color": np.vstack(lc),
            "style": np.vstack(ls),
        },
    )
