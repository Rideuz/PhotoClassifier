"""
Автозапуск серии экспериментов из YAML.

Пример:
  python run_experiments.py --plan experiments/plan_quick.yaml --dry-run
  python run_experiments.py --plan experiments/plan_quick.yaml
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


def load_plan(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("YAML-план должен быть объектом (dict) верхнего уровня.")
    if "experiments" not in data or not isinstance(data["experiments"], list) or not data["experiments"]:
        raise ValueError("В плане нужен непустой список experiments.")
    return data


    def _to_cli_flags(params: dict) -> list[str]:
    flags: list[str] = []
    for k, v in params.items():
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                flags.append(flag)
            continue
        if v is None:
            continue
        flags.extend([flag, str(v)])
    return flags


def build_command(base_output_dir: str, exp: dict, common: dict, train_script: str) -> list[str]:
    name = exp.get("name")
    if not name:
        raise ValueError("У каждого эксперимента должен быть name.")
    params = dict(common)
    params.update(exp.get("params", {}))

    cmd = [sys.executable, train_script, "--output-dir", base_output_dir, "--run-name", str(name)]
    cmd.extend(_to_cli_flags(params))
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Автосерия train_resnet.py по YAML-плану.")
    parser.add_argument("--plan", type=Path, required=True, help="YAML-файл серии экспериментов.")
    parser.add_argument("--dry-run", action="store_true", help="Только показать команды без запуска.")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Остановиться на первой ошибке (по умолчанию продолжает и пишет статус).",
    )
    parser.add_argument(
        "--aggregate-after",
        action="store_true",
        help="После серии вызвать aggregate_experiments.py",
    )
    args = parser.parse_args()

    plan = load_plan(args.plan)
    base_output_dir = str(plan.get("output_dir", "runs"))
    train_script = str(plan.get("train_script", "train_resnet.py"))
    common = plan.get("common_params", {})
    if not isinstance(common, dict):
        raise ValueError("common_params должен быть объектом (dict).")

    experiments = plan["experiments"]
    results: list[dict] = []

    print(f"[series] plan={args.plan}")
    print(f"[series] output_dir={base_output_dir}")
    print(f"[series] experiments={len(experiments)}")

    for idx, exp in enumerate(experiments, start=1):
        # Allow experiments to override output_dir (useful for saving to separate logic folders)
        exp_output_dir = exp.get("output_dir", base_output_dir)
        cmd = build_command(exp_output_dir, exp, common, train_script)
        name = str(exp.get("name", f"exp_{idx}"))
        print(f"\n[{idx}/{len(experiments)}] {name}")
        print(" ".join(cmd))

        if args.dry_run:
            results.append({"name": name, "status": "dry_run", "command": cmd})
            continue

        t0 = datetime.now()
        proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parent)
        dt = (datetime.now() - t0).total_seconds()
        status = "ok" if proc.returncode == 0 else f"error({proc.returncode})"
        results.append({"name": name, "status": status, "seconds": dt, "command": cmd})
        print(f"[done] {name}: {status}, {dt:.1f} sec")

        if proc.returncode != 0 and args.stop_on_error:
            break

    out_dir = Path("experiments/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    series_log = out_dir / f"series_run_{stamp}.json"
    series_log.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[series] log: {series_log}")

    if args.aggregate_after and not args.dry_run:
        agg_cmd = [sys.executable, "aggregate_experiments.py", "--runs-root", base_output_dir]
        print(f"[series] aggregate: {' '.join(agg_cmd)}")
        subprocess.run(agg_cmd, cwd=Path(__file__).resolve().parent, check=False)


if __name__ == "__main__":
    main()

