#!/usr/bin/env python3
"""Validate Clotho inputs or build the five-caption AudioCaps reference CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from eval_battery.config import Settings
from eval_battery.datasets import load_dataset
from eval_battery.datasets.audiocaps import prepare_audiocaps_references
from eval_battery.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/camera_ready.yaml")
    parser.add_argument("--dataset", required=True, choices=("clotho", "audiocaps"))
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="For AudioCaps, construct references_csv before validating it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings.load(args.config)
    if args.dataset == "audiocaps" and args.prepare:
        summary = prepare_audiocaps_references(
            settings.dataset_path("audiocaps", "official_test_csv"),
            settings.dataset_path("audiocaps", "ondisk_eval_csv"),
            settings.dataset_path("audiocaps", "audio_dir"),
            settings.dataset_path("audiocaps", "references_csv"),
        )
        print(f"Prepared AudioCaps references: {summary}")

    records = load_dataset(args.dataset, settings)
    source_ids = {record.source_id for record in records}
    missing_audio = sum(not Path(record.audio_path).is_file() for record in records)
    manifest = {
        "dataset": args.dataset,
        "sources": len(source_ids),
        "leave_one_out_records": len(records),
        "missing_audio_records": missing_audio,
        "caption_csv": str(settings.dataset_path(args.dataset, "references_csv")
                           if args.dataset == "audiocaps"
                           else settings.dataset_path(args.dataset, "captions_csv")),
    }
    output = settings.data_dir / args.dataset / "manifest.json"
    write_json(output, manifest)
    print(f"Validated {args.dataset}: {manifest}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
