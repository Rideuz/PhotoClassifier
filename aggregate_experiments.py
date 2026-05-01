"""
Сводный отчёт по экспериментам из runs/*.

Пример:
  python aggregate_experiments.py --runs-root runs --out-dir experiments/reports/latest
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _safe_get(d: dict, key: str, default=None):
    v = d.get(key, default)
    return v


def _numeric_or_none(v):
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _score_quality(row: pd.Series) -> float | None:
    vals = [
        row.get("test_type_macro_f1"),
        row.get("test_color_macro_f1"),
        row.get("test_style_macro_f1"),
    ]
    vals = [float(v) for v in vals if pd.notna(v)]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _score_realtime(row: pd.Series) -> float | None:
    q = row.get("score_quality_macro_f1_avg")
    ms = row.get("eff_inference_ms_per_image_batch1")
    size = row.get("eff_weights_disk_mb")
    if pd.isna(q) or pd.isna(ms):
        return None
    # Простая интерпретируемая формула: качество / штраф за latency и размер.
    penalty = (1.0 + float(ms) / 20.0) * (1.0 + (float(size) / 100.0 if pd.notna(size) else 0.0))
    return float(q) / penalty


def _score_catalog(row: pd.Series) -> float | None:
    q = row.get("score_quality_macro_f1_avg")
    style = row.get("test_style_macro_f1")
    if pd.isna(q):
        return None
    if pd.isna(style):
        return float(q)
    # Для каталога чуть сильнее учитываем сложный стиль.
    return float(0.7 * float(q) + 0.3 * float(style))


def collect_runs(runs_root: Path) -> list[dict]:
    records: list[dict] = []
    for run_dir in sorted([p for p in runs_root.rglob("*") if p.is_dir()]):
        test_m = _read_json(run_dir / "test_metrics.json")
        eff = _read_json(run_dir / "efficiency.json")
        train_s = _read_json(run_dir / "training_summary.json")
        dmeta = _read_json(run_dir / "dataset_meta.json")
        exp_manifest = _read_json(run_dir / "experiment_manifest.json")

        if not test_m and not train_s:
            continue

        loss_w = dmeta.get("loss_weights", {}) if isinstance(dmeta.get("loss_weights"), dict) else {}
        style_loss_info = dmeta.get("style_loss", {}) if isinstance(dmeta.get("style_loss"), dict) else {}

        rec: dict = {
            "run_dir": str(run_dir),
            "run_name": run_dir.name,
            "architecture": _safe_get(dmeta, "architecture", _safe_get(train_s, "architecture")),
            "model_name": _safe_get(dmeta, "model_name", _safe_get(train_s, "model_name")),
            "started_at_local": _safe_get(exp_manifest, "started_at_local"),
            # --- обучение ---
            "epochs_planned": _numeric_or_none(_safe_get(train_s, "epochs_planned", _safe_get(train_s, "epochs"))),
            "epochs_completed": _numeric_or_none(_safe_get(train_s, "epochs_completed")),
            "early_stopped": _safe_get(train_s, "early_stopped"),
            "best_val_avg_macro_f1": _numeric_or_none(_safe_get(train_s, "best_val_avg_macro_f1")),
            "training_wall_time_sec": _numeric_or_none(_safe_get(train_s, "total_wall_time_sec")),
            # --- данные ---
            "num_samples": _numeric_or_none(_safe_get(dmeta, "num_samples")),
            "batch_size": _numeric_or_none(_safe_get(dmeta, "batch_size")),
            "lr": _numeric_or_none(_safe_get(dmeta, "lr")),
            "weight_decay": _numeric_or_none(_safe_get(dmeta, "weight_decay")),
            "image_size": _numeric_or_none(_safe_get(dmeta, "image_size")),
            "split_stratify": _safe_get(dmeta, "split_stratify"),
            "min_style_frequency_filter": _numeric_or_none(_safe_get(dmeta, "min_style_frequency_filter")),
            # --- веса потерь ---
            "loss_weight_type": _numeric_or_none(loss_w.get("type")),
            "loss_weight_color": _numeric_or_none(loss_w.get("color")),
            "loss_weight_style": _numeric_or_none(loss_w.get("style")),
            "style_loss_mode": _safe_get(style_loss_info, "mode", "ce"),
        }
        # test_metrics — все числовые поля с префиксом test_
        for k, v in test_m.items():
            rec[f"test_{k}"] = _numeric_or_none(v) if isinstance(v, (int, float)) else v
        # efficiency — все числовые поля с префиксом eff_
        for k, v in eff.items():
            rec[f"eff_{k}"] = _numeric_or_none(v) if isinstance(v, (int, float)) else v
        records.append(rec)
    return records


def make_markdown_report(df: pd.DataFrame, out_path: Path, top_k: int) -> None:
    lines: list[str] = []
    lines.append("# Experiment Report")
    lines.append("")
    lines.append(f"- Total runs: **{len(df)}**")
    lines.append("")

    cols_view = [
        "architecture",
        "model_name",
        "run_name",
        "started_at_local",
        # --- composite scores ---
        "score_quality_macro_f1_avg",
        "score_realtime",
        "score_catalog",
        # --- test quality ---
        "test_type_macro_f1",
        "test_color_macro_f1",
        "test_style_macro_f1",
        "test_type_accuracy",
        "test_color_accuracy",
        "test_style_accuracy",
        "test_type_topk_accuracy",
        "test_color_coarse_group_accuracy",
        # --- best val ---
        "best_val_avg_macro_f1",
        # --- efficiency ---
        "eff_inference_ms_per_image_batch1",
        "eff_inference_ms_per_image_batchN",
        "eff_weights_disk_mb",
        "eff_param_count",
        # --- training info ---
        "training_wall_time_sec",
        "epochs_planned",
        "epochs_completed",
        "early_stopped",
        # --- hyperparams ---
        "num_samples",
        "batch_size",
        "lr",
        "weight_decay",
        "image_size",
        "loss_weight_type",
        "loss_weight_color",
        "loss_weight_style",
        "style_loss_mode",
        "split_stratify",
        "min_style_frequency_filter",
    ]
    cols_view = [c for c in cols_view if c in df.columns]

    def _as_md_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "_No data_"
        cols = list(frame.columns)
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        body = []
        for _, row in frame.iterrows():
            vals = []
            for c in cols:
                v = row[c]
                if pd.isna(v):
                    vals.append("")
                elif isinstance(v, float):
                    vals.append(f"{v:.6g}")
                else:
                    vals.append(str(v))
            body.append("| " + " | ".join(vals) + " |")
        return "\n".join([header, sep] + body)

    def _section(title: str, sort_col: str):
        lines.append(f"## {title}")
        lines.append("")
        top = df.sort_values(sort_col, ascending=False).head(top_k)
        lines.append(_as_md_table(top[cols_view]))
        lines.append("")

    if "score_realtime" in df.columns:
        _section("Top Realtime Candidates", "score_realtime")
    if "score_catalog" in df.columns:
        _section("Top Catalog Candidates", "score_catalog")
    if "score_quality_macro_f1_avg" in df.columns:
        _section("Top By Quality (Macro-F1 Avg)", "score_quality_macro_f1_avg")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Сбор полного отчета по runs/*")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/reports/latest"))
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = collect_runs(args.runs_root)
    if not records:
        raise SystemExit(f"Не найдено завершённых прогонов в {args.runs_root}")

    df = pd.DataFrame(records)
    df["score_quality_macro_f1_avg"] = df.apply(_score_quality, axis=1)
    df["score_realtime"] = df.apply(_score_realtime, axis=1)
    df["score_catalog"] = df.apply(_score_catalog, axis=1)

    # Основной мастер-CSV
    df.sort_values("score_quality_macro_f1_avg", ascending=False, na_position="last").to_csv(
        args.out_dir / "leaderboard.csv", index=False
    )
    (args.out_dir / "leaderboard.json").write_text(
        df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8"
    )
    make_markdown_report(df, args.out_dir / "report.md", top_k=args.top_k)
    print(f"Готово. Отчет: {args.out_dir}")


if __name__ == "__main__":
    main()

