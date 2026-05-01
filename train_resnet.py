"""
Обучение и оценка multitask ResNet50 на DeepFashion (img_highres).

По умолчанию НЕ все ~289k изображений — берётся стратифицированная выжимка
(default_max_samples в TrainConfig, сейчас 20k), чтобы итерации были терпимыми.

  python train_resnet.py --output-dir runs/dev

Полный датасет (долго, для финальных прогонов):
  python train_resnet.py --full-dataset --output-dir runs/full

Меньше / больше образцов явно:
  python train_resnet.py --max-samples 20000 --epochs 10 --output-dir runs/quick

Первые N строк аннотаций (без стратификации подвыборки):
  python train_resnet.py --max-samples 5000 --subset-selection first_n --rebuild-splits

Скорость: AMP, num_workers для train; val/test с num_workers=0 на Windows.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from photoclassifier.config import TrainConfig
from photoclassifier.data.annotations import (
    build_color_coarse_names,
    build_label_spaces,
    compute_multitask_labels,
    load_aligned_tables,
)
from photoclassifier.data.deepfashion import DeepFashionMultiTaskDataset, TransformSubsetDataset
from photoclassifier.data.paths import scan_img_highres_keys
from photoclassifier.data.stratify_labels import (
    build_split_stratify_array,
    filter_indices_by_style_frequency,
    split_stratify_stats,
)
from photoclassifier.data.subset import apply_dataset_subset
from photoclassifier.data.splits import load_splits, make_stratified_splits, save_splits
from photoclassifier.data.splits import make_style_coverage_splits
from photoclassifier.metrics.multitask import (
    coarse_color_group_accuracy,
    collect_predictions,
    confusion_matrix_np,
    evaluate_with_logits,
    save_confusion_matrix_csv,
)
from photoclassifier.models.resnet_multitask import MultiTaskResNet
from photoclassifier.utils.repro import hardware_and_versions_report, set_seed


def log(msg: str) -> None:
    print(f"[PhotoClassifier] {msg}", flush=True)


def config_to_jsonable(cfg: TrainConfig) -> dict:
    d = vars(cfg).copy()
    for k, v in d.items():
        if isinstance(v, Path):
            d[k] = str(v)
    return d


def resolve_run_dir(base_dir: Path, run_name: str | None, allow_overwrite: bool) -> Path:
    """Return a safe output directory for the current experiment."""
    if run_name:
        candidate = base_dir / run_name
    else:
        candidate = base_dir

    if allow_overwrite or not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    if candidate.is_dir() and not any(candidate.iterdir()):
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name:
        resolved = base_dir / f"{run_name}_{ts}"
    else:
        resolved = base_dir.parent / f"{base_dir.name}_{ts}"
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def verify_split_coverage(
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    name: str,
) -> list[str]:
    warnings: list[str] = []
    all_c = set(np.unique(y).tolist())
    for split, tag in (
        (train_idx, "train"),
        (val_idx, "val"),
        (test_idx, "test"),
    ):
        present = set(np.unique(y[split]).tolist())
        missing = all_c - present
        if missing:
            warnings.append(
                f"{name}: в {tag} нет классов {sorted(missing)[:10]}"
                f"{'...' if len(missing) > 10 else ''} (всего {len(missing)})"
            )
    return warnings


def build_color_group_mapping(color_attr_names: list[str], num_color_classes: int) -> tuple[list[int], list[str]]:
    """Индекс класса цвета -> id грубой группы; последний класс = none."""
    coarse_per_attr = build_color_coarse_names(color_attr_names)
    uniq = sorted(set(coarse_per_attr))
    gid = {g: i for i, g in enumerate(uniq)}
    class_to_group = [gid[c] for c in coarse_per_attr]
    none_gid = len(uniq)
    class_to_group.append(none_gid)
    group_names = uniq + ["none"]
    assert len(class_to_group) == num_color_classes
    return class_to_group, group_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--loss-weight-type", type=float, default=None)
    parser.add_argument("--loss-weight-color", type=float, default=None)
    parser.add_argument("--loss-weight-style", type=float, default=None)
    parser.add_argument("--rebuild-splits", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Имя подкаталога эксперимента внутри --output-dir (например, resnet50_lr3e4_bs32).",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Разрешить запись в существующую непустую папку (иначе будет создана новая с timestamp).",
    )
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="Использовать все доступные на диске изображения (~289k). Без этого флага "
        "берётся стратифицированная выжимка размера default_max_samples из TrainConfig.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Число образцов после фильтра по диску (стратификация по типу). "
        "Если не указано и нет --full-dataset — используется default_max_samples из конфига.",
    )
    parser.add_argument(
        "--subset-selection",
        choices=("stratified", "first_n"),
        default="stratified",
        help="stratified: случайная подвыборка с сохранением долей категорий; "
        "first_n: первые N строк аннотаций (как раньше).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Отключить tqdm (прогресс по батчам и эпохам).",
    )
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--eval-num-workers",
        type=int,
        default=None,
        help="num_workers только для val/test (по умолчанию 0 на Windows — без лишних процессов).",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Отключить mixed precision (AMP) на CUDA.",
    )
    parser.add_argument(
        "--split-stratify",
        choices=("type", "type_coarse_style", "style_coverage"),
        default="type",
        help="Чему соответствует stratify при train/val/test: только тип или тип+грубый стиль "
        "(редкие стили в один бин — см. --style-stratify-min-freq).",
    )
    parser.add_argument(
        "--style-stratify-min-freq",
        type=int,
        default=5,
        help="Для type_coarse_style: стили реже этого числа образцов в текущей выборке "
        "считаются одним «хвостовым» классом только для разбиения.",
    )
    parser.add_argument(
        "--min-style-frequency",
        type=int,
        default=0,
        help="Если >0 — удалить из датасета образцы, чей класс стиля встречается "
        "реже N раз (уменьшает хвост, проще метрики; меньше классов в val).",
    )
    parser.add_argument(
        "--style-balanced-subset",
        action="store_true",
        help="Отобрать подвыборку с уравниванием количества примеров на стиль (кап/равное число на стиль). "
        "Это улучшает macro-F1 по style, уменьшая дисбаланс и эффект отсутствующих стилей.",
    )
    parser.add_argument(
        "--style-balance-min-keep",
        type=int,
        default=3,
        help="Минимальная поддержка стиля (кол-во примеров в текущем подмножестве), чтобы стиль участвовал в "
        "style-balanced подборке и гарантированных split-ах.",
    )
    parser.add_argument(
        "--style-balance-per-style",
        type=int,
        default=None,
        help="Сколько примеров оставлять на каждый стиль в style-balanced режиме. "
        "Если не задано — вычисляется автоматически из max_samples и числа стилей.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Количество эпох без улучшений val_avg_macro_f1 для ранней остановки (0 = отключено).",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.001,
        help="Минимальное улучшение val_avg_macro_f1, чтобы считаться прогрессом.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Применить torch.compile к модели (PyTorch 2.0+). Ускоряет обучение на ~10-30%%, "
             "первая эпоха медленнее из-за компиляции.",
    )
    args = parser.parse_args()

    if args.full_dataset and args.max_samples is not None:
        raise SystemExit("Укажите либо --full-dataset, либо --max-samples, не оба смысла сразу.")

    cfg = TrainConfig()
    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.image_size is not None:
        if args.image_size <= 0:
            raise SystemExit("--image-size должен быть положительным целым.")
        cfg.image_size = args.image_size
    if args.rebuild_splits:
        cfg.rebuild_splits = True
    if args.seed is not None:
        cfg.random_seed = args.seed
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.eval_num_workers is not None:
        cfg.num_workers_eval = args.eval_num_workers
    if args.no_amp:
        cfg.amp = False
    if args.loss_weight_type is not None:
        cfg.loss_weights["type"] = float(args.loss_weight_type)
    if args.loss_weight_color is not None:
        cfg.loss_weights["color"] = float(args.loss_weight_color)
    if args.loss_weight_style is not None:
        cfg.loss_weights["style"] = float(args.loss_weight_style)
    cfg.validate_ratios()

    if args.full_dataset:
        effective_max_samples: int | None = None
    elif args.max_samples is not None:
        if args.max_samples <= 0:
            raise SystemExit("--max-samples должен быть положительным целым.")
        effective_max_samples = args.max_samples
    else:
        effective_max_samples = cfg.default_max_samples

    show_progress = not args.no_progress

    set_seed(cfg.random_seed)
    cfg.output_dir = resolve_run_dir(cfg.output_dir, args.run_name, args.overwrite_output)
    log(f"Каталог эксперимента: {cfg.output_dir}")

    launch_info = {
        "started_at_local": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,
        "config": config_to_jsonable(cfg),
        "cli": {k: v for k, v in vars(args).items() if not isinstance(v, Path)},
    }
    (cfg.output_dir / "experiment_manifest.json").write_text(
        json.dumps(launch_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = hardware_and_versions_report()
    report["random_seed"] = cfg.random_seed
    report["batch_size_train"] = cfg.batch_size
    (cfg.output_dir / "run_environment.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and cfg.amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    log(f"Устройство: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    if device.type != "cuda":
        log("CUDA недоступна — обучение на CPU будет заметно медленнее.")
    elif use_amp:
        log("AMP (mixed precision, float16) на CUDA: включён — обычно быстрее и легче по VRAM.")
    else:
        log("AMP на CUDA выключен (--no-amp).")
    if cfg.num_workers == 0:
        log(
            "Внимание: num_workers=0 — JPEG читает один поток, GPU часто простаивает. "
            "Попробуйте --num-workers 4 (или 8), если нет ошибок multiprocessing."
        )
    else:
        log(
            f"DataLoader train: num_workers={cfg.num_workers}; "
            f"val/test: num_workers={cfg.num_workers_eval} "
            f"(отдельные воркеры там отключены по умолчанию — меньше риск WinError 1455 при загрузке CUDA)"
        )

    anno_dir = cfg.data_root / cfg.anno_subdir
    attr_cloth = anno_dir / "list_attr_cloth.txt"
    category_cloth = anno_dir / "list_category_cloth.txt"

    color_idx, style_idx, color_names, style_names = build_label_spaces(attr_cloth)
    type_names: list[str] = []
    with category_cloth.open(encoding="utf-8", errors="replace") as f:
        n = int(f.readline().strip())
        f.readline()
        for _ in range(n):
            type_names.append(f.readline().split()[0].strip())

    log("Сканирую img_highres и сопоставляю с аннотациями (это может занять время)…")
    existing = scan_img_highres_keys(cfg.data_root, cfg.image_subdir)
    log(f"Найдено файлов на диске (ключей img/...): {len(existing)}")
    paths, y_type, attr_mat = load_aligned_tables(anno_dir, existing)
    log(f"Строк в аннотациях с файлом на диске: {len(paths)}")
    if args.full_dataset:
        log("Режим: полный датасет (--full-dataset).")
    elif args.max_samples is not None:
        log(f"Режим: явная подвыборка --max-samples={args.max_samples}.")
    else:
        log(
            f"Режим: стратифицированная выжимка по умолчанию "
            f"(default_max_samples={cfg.default_max_samples}). Для всех изображений: --full-dataset."
        )
    paths, y_type, attr_mat, subset_meta = apply_dataset_subset(
        paths,
        y_type,
        attr_mat,
        effective_max_samples,
        args.subset_selection,
        cfg.random_seed,
    )
    if subset_meta["subset_applied"]:
        log(
            f"Подвыборка: {subset_meta['subset_strategy']}, "
            f"используется {subset_meta['subset_size_used']} из "
            f"{subset_meta['full_after_disk_filter']} изображений."
        )
    else:
        log("Используется полный доступный датасет (без усечения по --max-samples).")
    y_color, y_style = compute_multitask_labels(attr_mat, color_idx, style_idx)

    if args.min_style_frequency > 0:
        n0 = len(paths)
        keep = filter_indices_by_style_frequency(y_style, args.min_style_frequency)
        paths = [p for p, k in zip(paths, keep) if k]
        y_type = y_type[keep]
        attr_mat = attr_mat[keep]
        y_color = y_color[keep]
        y_style = y_style[keep]
        log(
            f"Фильтр стиля: оставлены классы с частотой ≥{args.min_style_frequency} "
            f"({len(paths)} / {n0} образцов)."
        )

    style_balance_info: dict = {
        "style_balance_enabled": bool(args.style_balanced_subset),
        "style_balance_per_style": None,
        "style_balance_min_keep": args.style_balance_min_keep,
        "style_balance_kept_style_ids": [],
        "style_balance_total_samples": None,
    }

    if args.style_balanced_subset:
        before_n = len(paths)
        counts = np.bincount(y_style.astype(np.int64))
        eligible_style_ids = [int(i) for i, c in enumerate(counts.tolist()) if c >= args.style_balance_min_keep]

        if not eligible_style_ids:
            log(
                "style-balanced: нет стилей с поддержкой "
                f">={args.style_balance_min_keep} — пропускаю уравнивание."
            )
        else:
            k_requested = args.style_balance_per_style
            if k_requested is not None:
                k = int(k_requested)
            else:
                # Авто-K: целимcя в равенство на текущем размере после min_style_frequency фильтра.
                k = int(before_n // max(1, len(eligible_style_ids)))
                k = max(args.style_balance_min_keep, k)

            eligible_style_ids0 = eligible_style_ids
            # Чтобы сохранить равные количества, отбрасываем стили, у которых поддержка меньше k.
            eligible_style_ids = [sid for sid in eligible_style_ids0 if counts[sid] >= k]
            if not eligible_style_ids:
                # fallback: гарантируем не пустой набор
                k = int(args.style_balance_min_keep)
                eligible_style_ids = eligible_style_ids0

            if eligible_style_ids:
                k = int(min(k, min(counts[sid] for sid in eligible_style_ids)))

            rng = np.random.default_rng(cfg.random_seed)
            sel: list[int] = []
            for sid in sorted(eligible_style_ids):
                idxs = np.flatnonzero(y_style == sid).astype(np.int64)
                if idxs.shape[0] < k:
                    continue
                picked = rng.choice(idxs, size=k, replace=False)
                sel.extend(picked.tolist())

            sel = np.array(sorted(set(sel)), dtype=np.int64)
            if sel.shape[0] == 0:
                log("style-balanced: после отбора не осталось данных — пропускаю уравнивание.")
            else:
                paths = [paths[i] for i in sel.tolist()]
                y_type = y_type[sel]
                attr_mat = attr_mat[sel]
                y_color = y_color[sel]
                y_style = y_style[sel]

                style_balance_info["style_balance_per_style"] = int(k)
                style_balance_info["style_balance_kept_style_ids"] = sorted(int(s) for s in set(y_style.tolist()))
                style_balance_info["style_balance_total_samples"] = int(sel.shape[0])

    if len(paths) == 0:
        raise RuntimeError("Нет образцов после фильтров — проверьте данные и --min-style-frequency.")

    num_type = len(type_names)
    num_color = len(color_idx) + 1
    num_style = len(style_idx) + 1
    color_class_to_group, color_group_names = build_color_group_mapping(color_names, num_color)

    n_samples = len(paths)
    all_idx = np.arange(n_samples, dtype=np.int64)
    splits_path = cfg.output_dir / cfg.splits_file
    manifest_path = cfg.output_dir / "subset_manifest.json"
    subset_meta = dict(subset_meta)
    subset_meta["subset_size_used"] = n_samples
    subset_manifest = {
        **subset_meta,
        "random_seed": cfg.random_seed,
        "split_stratify": args.split_stratify,
        "style_stratify_min_freq": args.style_stratify_min_freq,
        "min_style_frequency_filter": args.min_style_frequency,
        **style_balance_info,
        "n_samples": n_samples,
    }

    reuse_splits = splits_path.exists() and not cfg.rebuild_splits
    if reuse_splits:
        if not manifest_path.exists():
            log("Нет subset_manifest.json — пересоздаю train/val/test для согласованности.")
            reuse_splits = False
        else:
            try:
                prev = json.loads(manifest_path.read_text(encoding="utf-8"))
                if prev != subset_manifest:
                    log("Параметры подвыборки/сида изменились — пересоздаю splits.npz.")
                    reuse_splits = False
            except json.JSONDecodeError:
                reuse_splits = False

    if reuse_splits:
        log(f"Загружаю сохранённые сплиты: {splits_path.name}")
        train_idx, val_idx, test_idx = load_splits(splits_path)
    else:
        log(
            f"Делю данные на train/val/test ({cfg.train_ratio:.0%}/"
            f"{cfg.val_ratio:.0%}/{cfg.test_ratio:.0%}), split={args.split_stratify}…"
        )
        if args.split_stratify == "style_coverage":
            train_idx, val_idx, test_idx = make_style_coverage_splits(
                y_style=y_style,
                indices=all_idx,
                seed=cfg.random_seed,
                train_ratio=cfg.train_ratio,
                val_ratio=cfg.val_ratio,
                test_ratio=cfg.test_ratio,
                min_per_split=1,
            )
        else:
            stratify_y = build_split_stratify_array(
                y_type,
                y_style,
                args.split_stratify,
                args.style_stratify_min_freq,
            )
            train_idx, val_idx, test_idx = make_stratified_splits(
                stratify_y,
                all_idx,
                cfg.random_seed,
                cfg.train_ratio,
                cfg.val_ratio,
                cfg.test_ratio,
            )
        save_splits(splits_path, train_idx, val_idx, test_idx)
        manifest_path.write_text(
            json.dumps(subset_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    for w in verify_split_coverage(y_type, train_idx, val_idx, test_idx, "type"):
        print("Стратификация:", w)
    for w in verify_split_coverage(y_color, train_idx, val_idx, test_idx, "color"):
        print("Стратификация:", w)
    for w in verify_split_coverage(y_style, train_idx, val_idx, test_idx, "style"):
        print("Стратификация:", w)

    log(
        "Про «пропуски» классов style в val/test: при ~227 классах и длинном хвосте "
        "случайное разбиение почти всегда оставляет часть классов только в train — "
        "это нормально. Macro-F1 по стилю тогда усредняет и нули для классов без "
        "поддержки в сплите. См. micro-F1/accuracy и опции --split-stratify "
        "type_coarse_style, --min-style-frequency."
    )

    meta = {
        "architecture": "resnet",
        "model_name": "resnet50",
        "num_samples": n_samples,
        "type_category_names": type_names,
        "num_type": num_type,
        "num_color": num_color,
        "num_style": num_style,
        "color_attr_names": color_names,
        "style_attr_count": len(style_names),
        "splits": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "color_coarse_groups": color_group_names,
        "subset": subset_manifest,
        "epochs_planned": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "image_size": cfg.image_size,
        "amp": use_amp,
        "num_workers": cfg.num_workers,
        "num_workers_eval": cfg.num_workers_eval,
        "split_stratify": args.split_stratify,
        "style_stratify_min_freq": args.style_stratify_min_freq,
        "min_style_frequency_filter": args.min_style_frequency,
        "split_coverage_stats": {
            "type": split_stratify_stats(y_type, train_idx, val_idx, test_idx, "type"),
            "color": split_stratify_stats(y_color, train_idx, val_idx, test_idx, "color"),
            "style": split_stratify_stats(y_style, train_idx, val_idx, test_idx, "style"),
        },
        "early_stopping": {
            "patience": args.patience,
            "min_delta": args.min_delta,
        },
    }
    (cfg.output_dir / "dataset_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(cfg.image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(int(cfg.image_size * 1.14)),
            transforms.CenterCrop(cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    full_ds = DeepFashionMultiTaskDataset(
        cfg.data_root,
        cfg.image_subdir,
        paths,
        y_type,
        y_color,
        y_style,
        transform=None,
    )

    train_ds = TransformSubsetDataset(full_ds, train_idx, train_tf)
    val_ds = TransformSubsetDataset(full_ds, val_idx, eval_tf)
    test_ds = TransformSubsetDataset(full_ds, test_idx, eval_tf)

    def _dl_kwargs(nw: int) -> dict:
        kw: dict = {
            "num_workers": nw,
            "pin_memory": device.type == "cuda",
        }
        if nw > 0:
            kw["persistent_workers"] = True
            kw["prefetch_factor"] = 2
        return kw

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        **_dl_kwargs(cfg.num_workers),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        **_dl_kwargs(cfg.num_workers_eval),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        **_dl_kwargs(cfg.num_workers_eval),
    )

    log("Создаю модель: ResNet50 (ImageNet) + головы type/color/style…")
    model = MultiTaskResNet(num_type, num_color, num_style, pretrained=cfg.pretrained)
    model = model.to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
        log("torch.compile включён — первая эпоха медленнее, далее быстрее.")
    elif args.compile:
        log("Предупреждение: --compile указан, но torch.compile недоступен (требуется PyTorch 2.0+).")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    ce = nn.CrossEntropyLoss()
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def _autocast():
        if not use_amp:
            return contextlib.nullcontext()
        try:
            return torch.amp.autocast("cuda", dtype=torch.float16)
        except (TypeError, AttributeError):
            return torch.cuda.amp.autocast()

    def run_epoch(loader, train_mode: bool, desc: str = ""):
        if train_mode:
            model.train()
        else:
            model.eval()
        totals = {"loss": 0.0, "n": 0, "type": 0.0, "color": 0.0, "style": 0.0}
        it = loader
        if show_progress:
            it = tqdm(loader, desc=desc, leave=False)
        for batch in it:
            x = batch["image"].to(device, non_blocking=True)
            yt = batch["y_type"].to(device, non_blocking=True)
            yc = batch["y_color"].to(device, non_blocking=True)
            ys = batch["y_style"].to(device, non_blocking=True)
            if train_mode:
                opt.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train_mode):
                with _autocast():
                    out = model(x)
                    lt = ce(out["type"], yt)
                    lc = ce(out["color"], yc)
                    ls = ce(out["style"], ys)
                    loss = (
                        cfg.loss_weights["type"] * lt
                        + cfg.loss_weights["color"] * lc
                        + cfg.loss_weights["style"] * ls
                    )
            if train_mode:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            bs = x.size(0)
            totals["loss"] += float(loss.item()) * bs
            totals["type"] += float(lt.item()) * bs
            totals["color"] += float(lc.item()) * bs
            totals["style"] += float(ls.item()) * bs
            totals["n"] += bs
        for k in ("loss", "type", "color", "style"):
            totals[k] /= max(totals["n"], 1)
        return totals

    history_rows = []
    best_score = -1.0
    best_path = cfg.output_dir / "best_model.pt"
    epochs_no_improve = 0
    t_train0 = time.perf_counter()

    log(
        f"Старт обучения: {cfg.epochs} эпох, batch_size={cfg.batch_size}, "
        f"lr={cfg.lr}, train/val/test={len(train_idx)}/{len(val_idx)}/{len(test_idx)}."
    )
    epoch_iter = range(1, cfg.epochs + 1)
    if show_progress:
        epoch_iter = tqdm(epoch_iter, desc="Эпохи")

    for epoch in epoch_iter:
        t0 = time.perf_counter()
        tr = run_epoch(train_loader, True, desc=f"train e{epoch}")
        sched.step()
        va = run_epoch(val_loader, False, desc=f"val e{epoch}")
        targets, logits = collect_predictions(
            model,
            val_loader,
            device,
            show_progress=show_progress,
            desc=f"val predict e{epoch}",
            use_amp=use_amp,
        )
        metrics = evaluate_with_logits(
            logits["type"],
            logits["color"],
            logits["style"],
            targets["type"],
            targets["color"],
            targets["style"],
            num_type,
            cfg.top_k_type,
        )
        pred_c = logits["color"].argmax(axis=1)
        metrics["color_coarse_group_accuracy"] = coarse_color_group_accuracy(
            targets["color"],
            pred_c,
            color_class_to_group,
            len(color_group_names),
        )
        row = {
            "epoch": epoch,
            "lr": float(sched.get_last_lr()[0]),
            "train_loss": tr["loss"],
            "val_loss": va["loss"],
            "val_type_acc": metrics["type_accuracy"],
            "val_color_acc": metrics["color_accuracy"],
            "val_style_acc": metrics["style_accuracy"],
            "val_type_macro_f1": metrics["type_macro_f1"],
            "val_color_macro_f1": metrics["color_macro_f1"],
            "val_style_macro_f1": metrics["style_macro_f1"],
            "val_type_topk_acc": metrics["type_topk_accuracy"],
            "val_color_coarse_group_acc": metrics["color_coarse_group_accuracy"],
            "time_epoch_sec": time.perf_counter() - t0,
        }
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(cfg.output_dir / "training_log.csv", index=False)

        score = (
            metrics["type_macro_f1"]
            + metrics["color_macro_f1"]
            + metrics["style_macro_f1"]
        ) / 3.0
        postfix = (
            f"tl={tr['loss']:.3f} vl={va['loss']:.3f} "
            f"t_acc={metrics['type_accuracy']:.2f} f1_avg={score:.3f}"
        )
        if show_progress and isinstance(epoch_iter, tqdm):
            epoch_iter.set_postfix_str(postfix)
        else:
            log(f"Эпоха {epoch}/{cfg.epochs}: {postfix}")

        if score > best_score + args.min_delta:
            best_score = score
            epochs_no_improve = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                    "config": config_to_jsonable(cfg),
                },
                best_path,
            )
        else:
            epochs_no_improve += 1
            if args.patience > 0 and epochs_no_improve >= args.patience:
                log(f"Ранняя остановка на эпохе {epoch}, так как метрика не улучшалась {args.patience} эпох.")
                break

    training_wall = time.perf_counter() - t_train0
    (cfg.output_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "architecture": "resnet",
                "model_name": "resnet50",
                "total_wall_time_sec": training_wall,
                "epochs": cfg.epochs,
                "best_val_avg_macro_f1": best_score,
                "checkpoint": str(best_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    log(f"Обучение закончено за {training_wall:.1f} с. Загружаю лучший чекпоинт и оцениваю test…")
    # Тест с лучшим чекпоинтом
    try:
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    targets, logits = collect_predictions(
        model,
        test_loader,
        device,
        show_progress=show_progress,
        desc="test predict",
        use_amp=use_amp,
    )
    test_metrics = evaluate_with_logits(
        logits["type"],
        logits["color"],
        logits["style"],
        targets["type"],
        targets["color"],
        targets["style"],
        num_type,
        cfg.top_k_type,
    )
    pred_c = logits["color"].argmax(axis=1)
    test_metrics["color_coarse_group_accuracy"] = coarse_color_group_accuracy(
        targets["color"],
        pred_c,
        color_class_to_group,
        len(color_group_names),
    )
    (cfg.output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Confusion matrices (тип и стиль; цвет — малое число классов)
    pt = logits["type"].argmax(axis=1)
    ps = logits["style"].argmax(axis=1)
    np.savez_compressed(
        cfg.output_dir / "confusion_type.npz",
        matrix=confusion_matrix_np(targets["type"], pt, num_type),
        labels=np.arange(num_type),
    )
    np.savez_compressed(
        cfg.output_dir / "confusion_style.npz",
        matrix=confusion_matrix_np(targets["style"], ps, num_style),
        labels=np.arange(num_style),
    )
    save_confusion_matrix_csv(
        targets["type"],
        pt,
        type_names,
        cfg.output_dir / "confusion_type.csv",
        max_labels=60,
    )

    # Размер модели
    param_count = sum(p.numel() for p in model.parameters())
    tmp = cfg.output_dir / "_tmp_weights.pt"
    torch.save(model.state_dict(), tmp)
    disk_mb = tmp.stat().st_size / (1024 * 1024)
    tmp.unlink(missing_ok=True)

    log("Замер скорости инференса (warmup + батчи)…")
    # Инференс
    bench_bs1 = DataLoader(
        TransformSubsetDataset(full_ds, test_idx[: min(512, len(test_idx))], eval_tf),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    bench_bs32 = DataLoader(
        TransformSubsetDataset(full_ds, test_idx[: min(512, len(test_idx))], eval_tf),
        batch_size=min(32, cfg.batch_size),
        shuffle=False,
        num_workers=0,
    )

    def bench_loader(loader: DataLoader, max_images: int = 128) -> float:
        model.eval()
        if device.type == "cuda":
            torch.cuda.synchronize()
        it = iter(loader)
        for _ in range(cfg.inference_warmup_batches):
            batch = next(it, None)
            if batch is None:
                it = iter(loader)
                batch = next(it)
            x = batch["image"].to(device)
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        n_img = 0
        t0 = time.perf_counter()
        it = iter(loader)
        while n_img < max_images:
            batch = next(it, None)
            if batch is None:
                break
            x = batch["image"].to(device)
            model(x)
            n_img += x.size(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        return elapsed / max(n_img, 1) * 1000.0

    inf_report: dict = {
        "param_count": int(param_count),
        "weights_disk_mb": float(disk_mb),
        "note": "Время на изображение после warmup; GPU и версии — в run_environment.json",
    }
    try:
        inf_report["inference_ms_per_image_batch1"] = float(bench_loader(bench_bs1))
        inf_report["inference_ms_per_image_batchN"] = float(bench_loader(bench_bs32))
        inf_report["inference_batch_size_N"] = int(min(32, cfg.batch_size))
    except Exception as e:
        inf_report["inference_error"] = repr(e)

    (cfg.output_dir / "efficiency.json").write_text(
        json.dumps(inf_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary_csv = {
        **{f"test_{k}": v for k, v in test_metrics.items() if isinstance(v, (int, float))},
        **{f"eff_{k}": v for k, v in inf_report.items() if isinstance(v, (int, float))},
    }
    pd.DataFrame([summary_csv]).to_csv(cfg.output_dir / "final_summary.csv", index=False)

    print("Готово. Артефакты в", cfg.output_dir)

    # Авто-обновление leaderboard после каждого прогона
    try:
        import subprocess as _sp
        _runs_root = cfg.output_dir.parent
        _sp.run(
            [sys.executable, str(ROOT / "aggregate_experiments.py"),
             "--runs-root", str(_runs_root),
             "--out-dir", "experiments/reports/latest"],
            check=False, cwd=ROOT,
        )
        log("Leaderboard обновлён: experiments/reports/latest/")
    except Exception as _e:
        log(f"Предупреждение: авто-агрегация не удалась ({_e})")


if __name__ == "__main__":
    main()
