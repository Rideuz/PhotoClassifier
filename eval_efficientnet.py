"""
Оценка чекпоинта EfficientNet без обучения.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

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
from photoclassifier.data.stratify_labels import filter_indices_by_style_frequency
from photoclassifier.data.subset import apply_dataset_subset
from photoclassifier.data.splits import load_splits
from photoclassifier.metrics.multitask import (
    coarse_color_group_accuracy,
    collect_predictions,
    confusion_matrix_np,
    evaluate_with_logits,
    save_confusion_matrix_csv,
)
from photoclassifier.models.efficientnet_multitask import MultiTaskEfficientNet


def _load_type_names(category_cloth: Path) -> list[str]:
    names: list[str] = []
    with category_cloth.open(encoding="utf-8", errors="replace") as f:
        n = int(f.readline().strip())
        f.readline()
        for _ in range(n):
            names.append(f.readline().split()[0].strip())
    return names


def build_color_group_mapping(color_attr_names: list[str], num_color_classes: int) -> tuple[list[int], list[str]]:
    coarse_per_attr = build_color_coarse_names(color_attr_names)
    uniq = sorted(set(coarse_per_attr))
    gid = {g: i for i, g in enumerate(uniq)}
    class_to_group = [gid[c] for c in coarse_per_attr]
    none_gid = len(uniq)
    class_to_group.append(none_gid)
    return class_to_group, uniq + ["none"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("test", "val", "train"), default="test")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    meta = json.loads((run_dir / "dataset_meta.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "subset_manifest.json").read_text(encoding="utf-8"))
    ckpt_path = (args.checkpoint or run_dir / "best_model.pt").resolve()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = TrainConfig()
    if "config" in ckpt and ckpt["config"].get("data_root"):
        cfg.data_root = Path(ckpt["config"]["data_root"])
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    color_idx, style_idx, color_names, _ = build_label_spaces(cfg.data_root / cfg.anno_subdir / "list_attr_cloth.txt")
    type_names = _load_type_names(cfg.data_root / cfg.anno_subdir / "list_category_cloth.txt")
    existing = scan_img_highres_keys(cfg.data_root, cfg.image_subdir)
    paths, y_type, attr_mat = load_aligned_tables(cfg.data_root / cfg.anno_subdir, existing)
    paths, y_type, attr_mat, _ = apply_dataset_subset(
        paths, y_type, attr_mat, manifest.get("subset_requested"), manifest.get("subset_strategy", "stratified"), manifest.get("random_seed", 42)
    )
    y_color, y_style = compute_multitask_labels(attr_mat, color_idx, style_idx)
    min_style = int(manifest.get("min_style_frequency_filter", 0))
    style_balance_enabled = bool(manifest.get("style_balance_enabled", False))
    style_balance_per_style = manifest.get("style_balance_per_style", None)
    style_balance_kept_style_ids = manifest.get("style_balance_kept_style_ids", [])
    seed = int(manifest.get("random_seed", 42))
    if min_style > 0:
        keep = filter_indices_by_style_frequency(y_style, min_style)
        paths = [p for p, k in zip(paths, keep) if k]
        y_type, y_color, y_style = y_type[keep], y_color[keep], y_style[keep]

    if style_balance_enabled:
        if style_balance_per_style is None or not style_balance_kept_style_ids:
            pass
        else:
            k = int(style_balance_per_style)
            rng = np.random.default_rng(seed)
            sel: list[int] = []
            for sid in style_balance_kept_style_ids:
                sid = int(sid)
                idxs = np.flatnonzero(y_style == sid).astype(np.int64)
                if idxs.shape[0] < k:
                    continue
                picked = rng.choice(idxs, size=k, replace=False)
                sel.extend(picked.tolist())
            sel = np.array(sorted(set(sel)), dtype=np.int64)
            if sel.shape[0] > 0:
                paths = [paths[i] for i in sel.tolist()]
                y_type = y_type[sel]
                y_color = y_color[sel]
                y_style = y_style[sel]

    train_idx, val_idx, test_idx = load_splits(run_dir / "splits.npz")
    idx = {"train": train_idx, "val": val_idx, "test": test_idx}[args.split]
    image_size = int(meta.get("image_size", cfg.image_size))
    eval_tf = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    ds = DeepFashionMultiTaskDataset(cfg.data_root, cfg.image_subdir, paths, y_type, y_color, y_style, transform=None)
    loader = DataLoader(TransformSubsetDataset(ds, idx, eval_tf), batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    num_type, num_color, num_style = len(type_names), len(color_idx) + 1, len(style_idx) + 1
    model_name = str(meta.get("model_name", "efficientnet_b0"))
    variant = model_name.split("_")[-1]
    model = MultiTaskEfficientNet(num_type, num_color, num_style, variant=variant, pretrained=False)
    model.load_state_dict(ckpt["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    use_amp = device.type == "cuda" and not args.no_amp
    targets, logits = collect_predictions(model, loader, device, use_amp=use_amp, desc=f"{args.split} predict")
    metrics = evaluate_with_logits(
        logits["type"], logits["color"], logits["style"], targets["type"], targets["color"], targets["style"], num_type, int(meta.get("top_k_type", 3))
    )
    group_map, group_names = build_color_group_mapping(color_names, num_color)
    pred_c = logits["color"].argmax(axis=1)
    metrics["color_coarse_group_accuracy"] = coarse_color_group_accuracy(targets["color"], pred_c, group_map, len(group_names))
    suffix = args.split
    (run_dir / f"eval_{suffix}_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pt = logits["type"].argmax(axis=1)
    ps = logits["style"].argmax(axis=1)
    np.savez_compressed(run_dir / f"confusion_type_{suffix}.npz", matrix=confusion_matrix_np(targets["type"], pt, num_type), labels=np.arange(num_type))
    np.savez_compressed(run_dir / f"confusion_style_{suffix}.npz", matrix=confusion_matrix_np(targets["style"], ps, num_style), labels=np.arange(num_style))
    save_confusion_matrix_csv(targets["type"], pt, type_names, run_dir / f"confusion_type_{suffix}.csv", max_labels=60)
    flat = {f"{suffix}_{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
    pd.DataFrame([flat]).to_csv(run_dir / f"eval_{suffix}_summary.csv", index=False)
    t0 = time.perf_counter()
    _ = model(next(iter(loader))["image"].to(device))
    print(f"Готово. {args.split} metrics сохранены. Warmup={time.perf_counter()-t0:.3f}s")


if __name__ == "__main__":
    main()

