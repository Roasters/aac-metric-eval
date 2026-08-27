#!/usr/bin/env python3
"""Reconstruct descriptive, correlation, and paired-drop result tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from eval_battery.analysis import summarize_dataset
from eval_battery.config import Settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/camera_ready.yaml")
    parser.add_argument(
        "--datasets", nargs="+", default=["clotho", "audiocaps"],
        choices=("clotho", "audiocaps"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings.load(args.config)
    for dataset in args.datasets:
        result = summarize_dataset(
            dataset,
            settings.perturbation_dir / dataset,
            settings.score_dir / dataset,
            settings.table_dir / dataset,
            expected=settings.dataset(dataset).get("expected", {}),
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
