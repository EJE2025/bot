"""Utilities for promoting trained predictive models."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class Report:
    """Structured representation of a model evaluation report."""

    path: Path
    metrics: Dict[str, float]

    @classmethod
    def load(cls, path: Path) -> "Report":
        """Load a JSON report ensuring the required KPIs are present."""

        data = json.loads(path.read_text(encoding="utf-8"))
        required = ["win_rate", "expectancy", "profit_factor"]
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Faltan KPIs en {path}: {missing}")
        metrics = {key: float(data[key]) for key in required}
        return cls(path=path, metrics=metrics)


def compare_kpis(candidate: Report, baseline: Report, keys: List[str]) -> int:
    """Return how many KPIs improved in ``candidate`` versus ``baseline``."""

    better = 0
    for key in keys:
        if candidate.metrics.get(key, float("nan")) > baseline.metrics.get(key, float("nan")):
            better += 1
    return better


def promote(candidate_model: Path, candidate_manifest: Path, prod_model: Path, prod_manifest: Path) -> None:
    """Copy the candidate artifacts into production destinations."""

    prod_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_model, prod_model)
    if candidate_manifest.exists():
        try:
            if candidate_manifest.resolve() == prod_manifest.resolve():
                return
        except OSError:
            # If resolution fails we conservatively attempt the copy below.
            pass
        shutil.copy2(candidate_manifest, prod_manifest)


def main(argv: List[str]) -> int:
    """CLI entry point used by ``python -m trading_bot.model_ops``."""

    parser = argparse.ArgumentParser(prog="model_ops")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    promote_parser = subparsers.add_parser(
        "promote-if-better",
        help="Promueve el modelo si supera baseline en ≥ N KPIs",
    )
    promote_parser.add_argument("--candidate-report", required=True, type=Path)
    promote_parser.add_argument("--baseline-report", required=True, type=Path)
    promote_parser.add_argument("--candidate-model", required=True, type=Path)
    promote_parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=Path("models/manifest.json"),
    )
    promote_parser.add_argument(
        "--prod-model",
        type=Path,
        default=Path("models/model.pkl"),
    )
    promote_parser.add_argument(
        "--prod-manifest",
        type=Path,
        default=Path("models/manifest.json"),
    )
    promote_parser.add_argument("--kpis", default="win_rate,expectancy,profit_factor")
    promote_parser.add_argument("--min-better", type=int, default=3)
    promote_parser.add_argument(
        "--hot-reload-flag",
        type=Path,
        default=Path("models/.reload"),
    )

    args = parser.parse_args(argv)

    if args.cmd == "promote-if-better":
        candidate_report = Report.load(args.candidate_report)
        baseline_report = Report.load(args.baseline_report)
        keys = [key.strip() for key in str(args.kpis).split(",") if key.strip()]
        better = compare_kpis(candidate_report, baseline_report, keys)
        print(f"[model_ops] KPIs: {keys} | better={better}/{len(keys)}", flush=True)
        if better >= args.min_better:
            promote(
                args.candidate_model,
                args.candidate_manifest,
                args.prod_model,
                args.prod_manifest,
            )
            try:
                args.hot_reload_flag.write_text("reload", encoding="utf-8")
            except Exception:
                pass
            print("[model_ops] Promoted ✅", flush=True)
            return 0
        print("[model_ops] Not better, no promotion.", flush=True)
        return 2
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI hook
    sys.exit(main(sys.argv[1:]))
