"""
Пакетная генерация графиков и сводных таблиц по всем прогонам.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/reports/latest"))
    args = parser.parse_args()

    run_dirs = [p for p in sorted(args.runs_root.rglob("*")) if (p / "training_log.csv").exists()]
    print(f"[reports] runs found: {len(run_dirs)}")
    for run_dir in run_dirs:
        cmd = [sys.executable, "plot_training.py", "--run-dir", str(run_dir)]
        print(f"[plot] {' '.join(cmd)}")
        subprocess.run(cmd, check=False, cwd=Path(__file__).resolve().parent)

    agg = [sys.executable, "aggregate_experiments.py", "--runs-root", str(args.runs_root), "--out-dir", str(args.out_dir)]
    print(f"[aggregate] {' '.join(agg)}")
    subprocess.run(agg, check=False, cwd=Path(__file__).resolve().parent)
    print(f"[done] reports in {args.out_dir}")


if __name__ == "__main__":
    main()

