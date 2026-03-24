"""
Визуализация результатов прогона: кривые обучения, метрики, матрицы ошибок.

  python plot_training.py --run-dir runs/quick
  python plot_training.py   # по умолчанию TrainConfig().output_dir

Требуются matplotlib и seaborn (см. requirements.txt).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import seaborn as sns
except ImportError:
    sns = None

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from photoclassifier.config import TrainConfig


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _category_names_from_meta(meta: dict) -> list[str] | None:
    """Имена категорий типа, если есть в meta (расширяемый формат)."""
    names = meta.get("type_category_names")
    if isinstance(names, list) and names:
        return [str(x) for x in names]
    return None


def plot_losses(df: pd.DataFrame, out: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=dpi)
    ax.plot(df["epoch"], df["train_loss"], "o-", label="Train loss", color="#1f77b4")
    ax.plot(df["epoch"], df["val_loss"], "s-", label="Val loss", color="#ff7f0e")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Loss")
    ax.set_title("Суммарный loss (multitask)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_accuracies(df: pd.DataFrame, out: Path, dpi: int) -> None:
    cols = [c for c in ("val_type_acc", "val_color_acc", "val_style_acc") if c in df.columns]
    if not cols:
        return
    fig, ax = plt.subplots(figsize=(9, 5), dpi=dpi)
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    for i, c in enumerate(cols):
        lab = c.replace("val_", "").replace("_acc", "")
        ax.plot(df["epoch"], df[c], "o-", label=lab, color=colors[i % len(colors)])
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy по головам (валидация)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_macro_f1(df: pd.DataFrame, out: Path, dpi: int) -> None:
    cols = [c for c in ("val_type_macro_f1", "val_color_macro_f1", "val_style_macro_f1") if c in df.columns]
    if not cols:
        return
    fig, ax = plt.subplots(figsize=(9, 5), dpi=dpi)
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    for i, c in enumerate(cols):
        lab = c.replace("val_", "").replace("_macro_f1", "")
        ax.plot(df["epoch"], df[c], "o-", label=lab, color=colors[i % len(colors)])
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Macro-F1 по головам (валидация)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_extras(df: pd.DataFrame, out: Path, dpi: int) -> None:
    has_topk = "val_type_topk_acc" in df.columns
    has_coarse = "val_color_coarse_group_acc" in df.columns
    if not has_topk and not has_coarse:
        return
    fig, ax = plt.subplots(figsize=(9, 5), dpi=dpi)
    if has_topk:
        ax.plot(df["epoch"], df["val_type_topk_acc"], "o-", label="Top-k accuracy (type)", color="#9467bd")
    if has_coarse:
        ax.plot(
            df["epoch"],
            df["val_color_coarse_group_acc"],
            "s-",
            label="Coarse color groups",
            color="#8c564b",
        )
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Accuracy")
    ax.set_title("Доп. метрики (валидация)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_epoch_time(df: pd.DataFrame, out: Path, dpi: int) -> None:
    if "time_epoch_sec" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 4), dpi=dpi)
    ax.bar(df["epoch"].astype(int), df["time_epoch_sec"] / 60.0, color="#17becf", edgecolor="white")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Время, мин")
    ax.set_title("Длительность эпохи")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_confusion_heatmap(
    matrix: np.ndarray,
    labels: list[str] | None,
    out: Path,
    dpi: int,
    title: str,
    max_classes: int = 50,
) -> None:
    n = matrix.shape[0]
    if n == 0:
        return
    if n > max_classes:
        matrix = matrix[:max_classes, :max_classes].copy()
        if labels is not None:
            labels = labels[:max_classes]
        title = f"{title} (первые {max_classes} классов)"

    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm = matrix.astype(float) / row_sums

    fig_w = min(16, max(8, n * 0.22))
    fig_h = min(14, max(6, n * 0.2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    tick_labels = labels if labels is not None else [str(i) for i in range(matrix.shape[0])]
    if sns is not None:
        sns.heatmap(
            norm,
            ax=ax,
            cmap="Blues",
            vmin=0,
            vmax=1,
            square=False,
            linewidths=0,
            cbar_kws={"label": "Доля предсказаний по строке (нормализация по истине)"},
            xticklabels=tick_labels,
            yticklabels=tick_labels,
        )
    else:
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        fig.colorbar(im, ax=ax, label="Доля предсказаний по строке (нормализация по истине)")
        ax.set_xticks(np.arange(len(tick_labels)))
        ax.set_yticks(np.arange(len(tick_labels)))
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)
    ax.set_title(title)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    plt.xticks(rotation=75, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_test_metrics_bar(metrics: dict, out: Path, dpi: int) -> None:
    pairs = []
    for k, v in metrics.items():
        if not k.endswith("_accuracy") and not k.endswith("_macro_f1") and not k.endswith("_micro_f1"):
            continue
        if k.endswith("_note") or "topk" in k.lower():
            continue
        if isinstance(v, (int, float)) and not np.isnan(v):
            pairs.append((k.replace("_", " "), float(v)))
    if not pairs:
        return
    pairs.sort(key=lambda x: x[0])
    names, vals = zip(*pairs)
    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.35)), dpi=dpi)
    y = np.arange(len(names))
    ax.barh(y, vals, color="#2ca02c", height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Значение")
    ax.set_title("Метрики на test (из JSON)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Графики по артефактам train_resnet.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Папка прогона (runs/...)")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--style-confusion-max",
        type=int,
        default=35,
        help="Макс. размер стороны heatmap для style (классов много).",
    )
    args = parser.parse_args()

    run_dir = (args.run_dir or TrainConfig().output_dir).resolve()
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if sns is not None:
        sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)

    log_csv = run_dir / "training_log.csv"
    if not log_csv.exists():
        print(f"Нет {log_csv} — сначала обучите модель или укажите --run-dir.")
        sys.exit(1)

    df = pd.read_csv(log_csv)
    meta = _load_json(run_dir / "dataset_meta.json") or {}

    plot_losses(df, plots_dir / "01_loss_train_val.png", args.dpi)
    plot_accuracies(df, plots_dir / "02_accuracy_val_heads.png", args.dpi)
    plot_macro_f1(df, plots_dir / "03_macro_f1_val_heads.png", args.dpi)
    plot_extras(df, plots_dir / "04_extra_topk_coarse.png", args.dpi)
    plot_epoch_time(df, plots_dir / "05_epoch_duration_min.png", args.dpi)

    # Confusion type
    c_type_npz = run_dir / "confusion_type.npz"
    c_type_csv = run_dir / "confusion_type.csv"
    labels_type = _category_names_from_meta(meta)
    if c_type_npz.exists():
        z = np.load(c_type_npz)
        cm = z["matrix"]
        if labels_type is None and c_type_csv.exists():
            cdf = pd.read_csv(c_type_csv, index_col=0)
            labels_type = [str(x)[:32] for x in cdf.index.tolist()]
        plot_confusion_heatmap(
            cm,
            labels_type,
            plots_dir / "06_confusion_type_normalized.png",
            args.dpi,
            "Матрица ошибок: тип одежды (норм. по строке)",
            max_classes=50,
        )
    elif c_type_csv.exists():
        cdf = pd.read_csv(c_type_csv, index_col=0)
        cm = cdf.values.astype(float)
        labels_type = [str(x)[:32] for x in cdf.index.tolist()]
        plot_confusion_heatmap(
            cm,
            labels_type,
            plots_dir / "06_confusion_type_normalized.png",
            args.dpi,
            "Матрица ошибок: тип одежды (норм. по строке)",
            max_classes=50,
        )

    # Confusion style (урезанная)
    c_style = run_dir / "confusion_style.npz"
    if c_style.exists():
        z = np.load(c_style)
        cm = z["matrix"]
        plot_confusion_heatmap(
            cm,
            None,
            plots_dir / "07_confusion_style_normalized_topk.png",
            args.dpi,
            "Матрица ошибок: стиль (фрагмент)",
            max_classes=args.style_confusion_max,
        )

    # Test metrics bar
    test_m = _load_json(run_dir / "test_metrics.json")
    if test_m:
        plot_test_metrics_bar(test_m, plots_dir / "08_test_metrics_barh.png", args.dpi)
    ev = _load_json(run_dir / "eval_test_metrics.json")
    if ev and not test_m:
        plot_test_metrics_bar(ev, plots_dir / "08_test_metrics_barh.png", args.dpi)

    # Сводная таблица последней эпохи
    if len(df) > 0:
        last = df.iloc[-1]
        summary_path = plots_dir / "summary_last_epoch.txt"
        summary_path.write_text(
            last.to_string(),
            encoding="utf-8",
        )

    print(f"Готово. Графики: {plots_dir}")


if __name__ == "__main__":
    main()
