"""
Повторная оценка без обучения: загружает чекпоинт и те же данные/test, что и при train.

Пример:
  python eval_checkpoint.py
  python eval_checkpoint.py --run-dir runs/my_experiment

Нужны в run-dir: best_model.pt, splits.npz, subset_manifest.json, dataset_meta.json
(как после train_resnet.py). Данные на диске — тот же data-root, что при обучении.

  python eval_checkpoint.py --run-dir runs/quick --data-root data/category_and_attribbute_prediction
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
from photoclassifier.models.resnet_multitask import MultiTaskResNet
from photoclassifier.utils.repro import hardware_and_versions_report, set_seed


def log(msg: str) -> None:
    print(f"[PhotoClassifier] {msg}", flush=True)


def build_color_group_mapping(
    color_attr_names: list[str], num_color_classes: int
) -> tuple[list[int], list[str]]:
    coarse_per_attr = build_color_coarse_names(color_attr_names)
    uniq = sorted(set(coarse_per_attr))
    gid = {g: i for i, g in enumerate(uniq)}
    class_to_group = [gid[c] for c in coarse_per_attr]
    none_gid = len(uniq)
    class_to_group.append(none_gid)
    group_names = uniq + ["none"]
    assert len(class_to_group) == num_color_classes
    return class_to_group, group_names


def _load_type_names(category_cloth: Path) -> list[str]:
    names: list[str] = []
    with category_cloth.open(encoding="utf-8", errors="replace") as f:
        n = int(f.readline().strip())
        f.readline()
        for _ in range(n):
            names.append(f.readline().split()[0].strip())
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка сохранённого чекпоинта на test/val.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Папка прогона (best_model.pt, splits.npz, subset_manifest.json, dataset_meta.json). "
        "По умолчанию — output_dir из TrainConfig (обычно runs/resnet50_baseline).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Корень датасета. Если не задан — из чекпоинта config или TrainConfig по умолчанию.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Путь к .pt (по умолчанию run-dir/best_model.pt).",
    )
    parser.add_argument(
        "--split",
        choices=("test", "val", "train"),
        default="test",
        help="На каком сплите считать метрики.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Переопределить размер батча.")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--skip-efficiency", action="store_true", help="Не считать inference benchmark.")
    args = parser.parse_args()

    run_dir = (args.run_dir or TrainConfig().output_dir).resolve()
    log(f"Папка прогона: {run_dir}")
    manifest_path = run_dir / "subset_manifest.json"
    meta_path = run_dir / "dataset_meta.json"
    splits_path = run_dir / "splits.npz"
    ckpt_path = (args.checkpoint or (run_dir / "best_model.pt")).resolve()

    for p, label in (
        (meta_path, "dataset_meta.json"),
        (splits_path, "splits.npz"),
        (ckpt_path, "чекпоинт best_model.pt"),
    ):
        if not p.exists():
            hint = (
                " Укажите папку с артефактами: python eval_checkpoint.py --run-dir runs/ВАША_ПАПКА"
                if args.run_dir is None
                else ""
            )
            raise FileNotFoundError(f"Нет {label}: {p}.{hint}")

    dataset_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    elif isinstance(dataset_meta.get("subset"), dict) and dataset_meta["subset"]:
        manifest = dataset_meta["subset"]
        log("subset_manifest.json нет — беру параметры подвыборки из dataset_meta.json (ключ «subset»).")
    else:
        raise FileNotFoundError(
            f"Нужен либо файл {manifest_path.name}, либо в dataset_meta.json поле «subset» "
            "(так сохраняет train_resnet.py). Переобучите с текущей версией скрипта или укажите --run-dir "
            f"с готовым прогоном. Папка: {run_dir}"
        )

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    ck_cfg = ckpt.get("config") or {}

    cfg = TrainConfig()
    if args.data_root is not None:
        cfg.data_root = args.data_root
    else:
        dr = ck_cfg.get("data_root")
        if dr:
            cfg.data_root = Path(dr)
    cfg.output_dir = run_dir
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.no_amp:
        cfg.amp = False

    seed = int(manifest.get("random_seed", ck_cfg.get("random_seed", TrainConfig().random_seed)))
    set_seed(seed)

    max_samples = manifest.get("subset_requested")
    subset_strategy = manifest.get("subset_strategy", "stratified")
    min_style = int(manifest.get("min_style_frequency_filter", 0))
    n_expected = int(manifest.get("n_samples", dataset_meta.get("num_samples", 0)))

    show_progress = not args.no_progress
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and cfg.amp

    log(f"Устройство: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    log(f"Чекпоинт: {ckpt_path}")
    log(f"Сплит для метрик: {args.split}")

    anno_dir = cfg.data_root / cfg.anno_subdir
    attr_cloth = anno_dir / "list_attr_cloth.txt"
    category_cloth = anno_dir / "list_category_cloth.txt"

    color_idx, style_idx, color_names, _style_names = build_label_spaces(attr_cloth)
    type_names = _load_type_names(category_cloth)

    log("Сканирование диска и загрузка таблиц (как при обучении)…")
    existing = scan_img_highres_keys(cfg.data_root, cfg.image_subdir)
    paths, y_type, attr_mat = load_aligned_tables(anno_dir, existing)
    paths, y_type, attr_mat, _ = apply_dataset_subset(
        paths,
        y_type,
        attr_mat,
        max_samples,
        subset_strategy,
        seed,
    )
    y_color, y_style = compute_multitask_labels(attr_mat, color_idx, style_idx)

    if min_style > 0:
        keep = filter_indices_by_style_frequency(y_style, min_style)
        paths = [p for p, k in zip(paths, keep) if k]
        y_type = y_type[keep]
        attr_mat = attr_mat[keep]
        y_color = y_color[keep]
        y_style = y_style[keep]

    if len(paths) == 0:
        raise RuntimeError("После фильтров нет образцов — проверьте data-root и датасет.")

    if n_expected and len(paths) != n_expected:
        log(
            f"Предупреждение: сейчас {len(paths)} образцов, в manifest было n_samples={n_expected} "
            "(возможно сменились файлы на диске)."
        )

    num_type = len(type_names)
    num_color = len(color_idx) + 1
    num_style = len(style_idx) + 1
    color_class_to_group, color_group_names = build_color_group_mapping(color_names, num_color)

    train_idx, val_idx, test_idx = load_splits(splits_path)
    idx_map = {"train": train_idx, "val": val_idx, "test": test_idx}
    eval_idx = idx_map[args.split]

    image_size = int(dataset_meta.get("image_size", cfg.image_size))
    top_k = int(dataset_meta.get("top_k_type", cfg.top_k_type))

    eval_tf = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
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
    eval_ds = TransformSubsetDataset(full_ds, eval_idx, eval_tf)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers_eval,
        pin_memory=device.type == "cuda",
    )

    model = MultiTaskResNet(num_type, num_color, num_style, pretrained=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)

    if ckpt.get("epoch") is not None:
        log(f"В чекпоинте: эпоха {ckpt['epoch']}")

    log(f"Прогон предсказаний ({len(eval_idx)} образцов)…")
    targets, logits = collect_predictions(
        model,
        eval_loader,
        device,
        show_progress=show_progress,
        desc=f"{args.split} predict",
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
        top_k,
    )
    pred_c = logits["color"].argmax(axis=1)
    metrics["color_coarse_group_accuracy"] = coarse_color_group_accuracy(
        targets["color"],
        pred_c,
        color_class_to_group,
        len(color_group_names),
    )

    suffix = args.split
    (run_dir / f"eval_{suffix}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pt = logits["type"].argmax(axis=1)
    ps = logits["style"].argmax(axis=1)
    np.savez_compressed(
        run_dir / f"confusion_type_{suffix}.npz",
        matrix=confusion_matrix_np(targets["type"], pt, num_type),
        labels=np.arange(num_type),
    )
    np.savez_compressed(
        run_dir / f"confusion_style_{suffix}.npz",
        matrix=confusion_matrix_np(targets["style"], ps, num_style),
        labels=np.arange(num_style),
    )
    save_confusion_matrix_csv(
        targets["type"],
        pt,
        type_names,
        run_dir / f"confusion_type_{suffix}.csv",
        max_labels=60,
    )

    rep = hardware_and_versions_report()
    rep["eval_split"] = args.split
    (run_dir / f"eval_{suffix}_environment.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    inf_report: dict = {"note": "Только eval-скрипт"}
    if not args.skip_efficiency and device.type == "cuda":
        log("Замер инференса (короткий)…")
        bench_bs1 = DataLoader(
            TransformSubsetDataset(full_ds, eval_idx[: min(512, len(eval_idx))], eval_tf),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        bench_bs32 = DataLoader(
            TransformSubsetDataset(full_ds, eval_idx[: min(512, len(eval_idx))], eval_tf),
            batch_size=min(32, cfg.batch_size),
            shuffle=False,
            num_workers=0,
        )

        def bench_loader(loader: DataLoader, max_images: int = 128) -> float:
            model.eval()
            torch.cuda.synchronize()
            it = iter(loader)
            for _ in range(cfg.inference_warmup_batches):
                batch = next(it, None)
                if batch is None:
                    break
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    model(batch["image"].to(device))
            torch.cuda.synchronize()
            n_img = 0
            t0 = time.perf_counter()
            it = iter(loader)
            while n_img < max_images:
                batch = next(it, None)
                if batch is None:
                    break
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    model(batch["image"].to(device))
                n_img += batch["image"].size(0)
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / max(n_img, 1) * 1000.0

        try:
            inf_report["inference_ms_per_image_batch1"] = float(bench_loader(bench_bs1))
            inf_report["inference_ms_per_image_batchN"] = float(bench_loader(bench_bs32))
            inf_report["inference_batch_size_N"] = int(min(32, cfg.batch_size))
        except Exception as e:
            inf_report["inference_error"] = repr(e)

    (run_dir / f"eval_{suffix}_efficiency.json").write_text(
        json.dumps(inf_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    flat = {f"{suffix}_{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
    flat.update({f"eff_{k}": v for k, v in inf_report.items() if isinstance(v, (int, float))})
    pd.DataFrame([flat]).to_csv(run_dir / f"eval_{suffix}_summary.csv", index=False)

    log(f"Готово. Метрики: {run_dir / f'eval_{suffix}_metrics.json'}")


if __name__ == "__main__":
    main()
