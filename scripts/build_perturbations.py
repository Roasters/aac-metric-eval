#!/usr/bin/env python3
"""Build aligned baseline and controlled perturbation pair files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from eval_battery.config import Settings
from eval_battery.datasets import load_dataset
from eval_battery.io import read_records, sha256_file, write_json, write_records
from eval_battery.perturbations import (
    SynonymGenerator,
    adjacent_swaps,
    fixed_nonword_replacements,
    random_nonword_replacements,
    synonym_replacements,
    token_removals,
)


AXES = ("masking", "synonym", "swap_adjacent", "removal", "random_nonword")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/camera_ready.yaml")
    parser.add_argument("--dataset", required=True, choices=("clotho", "audiocaps"))
    parser.add_argument("--axes", nargs="+", choices=AXES)
    parser.add_argument("--max-records", type=int, default=0, help="Smoke-test cap; 0 means full.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--download-nltk", action="store_true")
    return parser.parse_args()


def _write(path: Path, records, *, force: bool) -> dict[str, object]:
    if path.exists() and not force:
        cached = read_records(path)
        return {
            "rows": len(cached),
            "realized": sum(record.perturbation.realized for record in cached),
            "sha256": sha256_file(path),
            "cached": True,
        }
    records = list(records)
    write_records(path, records)
    return {
        "rows": len(records),
        "realized": sum(record.perturbation.realized for record in records),
        "sha256": sha256_file(path),
        "cached": False,
    }


def main() -> None:
    args = parse_args()
    settings = Settings.load(args.config)
    settings.ensure_output_dirs()
    records = load_dataset(args.dataset, settings)
    if args.max_records:
        records = records[: args.max_records]

    perturbation_config = settings.section("perturbations")
    axes = args.axes or tuple(perturbation_config.get("core_axes", AXES[:3]))
    output_dir = settings.perturbation_dir / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "dataset": args.dataset,
        "source_records": len(records),
        "full_run": args.max_records == 0,
        "files": {},
    }
    manifest["files"]["original"] = _write(
        output_dir / "original.jsonl", records, force=args.force
    )

    for axis in axes:
        if axis == "masking":
            variants = fixed_nonword_replacements(
                records, str(perturbation_config.get("fixed_nonword", "xkqjvz"))
            )
        elif axis == "swap_adjacent":
            variants = adjacent_swaps(records)
        elif axis == "removal":
            variants = token_removals(records)
        elif axis == "random_nonword":
            variants = random_nonword_replacements(
                records,
                seed=int(perturbation_config.get("random_seed", 42)),
                length=int(perturbation_config.get("random_nonword_length", 6)),
            )
        else:
            # Global synonym config is the default (Clotho: MLM/RoBERTa); a dataset
            # may override the selector/model (AudioCaps: embedding/SBERT) to match
            # the paper's per-dataset protocol.
            synonym_config = dict(perturbation_config.get("synonym", {}))
            synonym_config.update(settings.dataset(args.dataset).get("synonym", {}) or {})
            generator = SynonymGenerator(
                selector=str(synonym_config.get("selector", "mlm")),
                model_name=str(synonym_config.get("model", "roberta-base")),
                top_k=int(synonym_config.get("top_k", 250)),
                single_token_only=bool(synonym_config.get("single_token_only", False)),
                content_words_only=bool(synonym_config.get("content_words_only", False)),
                device=str(synonym_config.get("device", "auto")),
                download_nltk=args.download_nltk,
            )
            variants = synonym_replacements(records, generator)
        path = output_dir / f"{axis}.jsonl"
        manifest["files"][axis] = _write(path, variants, force=args.force)
        print(f"{axis}: {manifest['files'][axis]}")

    expected = settings.dataset(args.dataset).get("expected", {})
    if args.max_records == 0 and "synonym" in manifest["files"]:
        synonym = manifest["files"]["synonym"]
        manifest["expected"] = dict(expected)
        manifest["verification"] = {
            "synonym_positions": synonym["rows"] == expected.get("synonym_positions"),
            "synonym_replacements": synonym["realized"]
            == expected.get("synonym_replacements"),
        }
    write_json(output_dir / "manifest.json", manifest)
    print(f"Wrote {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
