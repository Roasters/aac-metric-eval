#!/usr/bin/env python3
"""Run established metrics or a resumable CAF/FLEUR pipeline stage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from eval_battery.config import Settings
from eval_battery.io import read_json, read_records, write_json
from eval_battery.metrics import PairScoreCache, score_established_metrics
from eval_battery.metrics.caf import merge_shards, run_clap, run_fleur


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/camera_ready.yaml")
    commands = parser.add_subparsers(dest="command", required=True)

    established = commands.add_parser("established", help="Run the 13 established metrics.")
    established.add_argument("--dataset", required=True, choices=("clotho", "audiocaps"))
    established.add_argument("--sets", nargs="+")
    established.add_argument("--batch-size", type=int)
    established.add_argument("--device")
    established.add_argument("--force", action="store_true")

    register = commands.add_parser("register-caf", help="Register pair files and write CAF sidecars.")
    register.add_argument("--datasets", nargs="+", default=["clotho", "audiocaps"])
    register.add_argument("--sets", nargs="+")
    register.add_argument("--force", action="store_true")

    clap = commands.add_parser("clap", help="Compute the contrastive component of CAF.")
    clap.add_argument("--limit", type=int, default=0)

    fleur = commands.add_parser("fleur", help="Compute one FLEUR/CAF worker shard.")
    fleur.add_argument("--shard", type=int, default=0)
    fleur.add_argument("--num-shards", type=int, default=1)
    fleur.add_argument("--cache")
    fleur.add_argument("--limit", type=int, default=0)

    merge = commands.add_parser("merge-caf", help="Merge FLEUR worker caches.")
    merge.add_argument("--shards", nargs="+", required=True)

    commands.add_parser("finalize-caf", help="Refill every CAF sidecar from the cache.")
    return parser.parse_args()


def _caf_paths(settings: Settings) -> tuple[Path, Path]:
    section = settings.section("caf")
    cache = settings.score_dir / str(section.get("cache_file", "caf_cache.json"))
    pairs = settings.score_dir / str(section.get("pair_manifest", "caf_pairs.jsonl"))
    return cache, pairs


def _set_names(pair_dir: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    return sorted(path.stem for path in pair_dir.glob("*.jsonl"))


def run_established_command(args: argparse.Namespace, settings: Settings) -> None:
    evaluation = settings.section("evaluation")
    pair_dir = settings.perturbation_dir / args.dataset
    output_dir = settings.score_dir / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in _set_names(pair_dir, args.sets):
        input_path = pair_dir / f"{name}.jsonl"
        output_path = output_dir / f"{name}.established.json"
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        if output_path.exists() and not args.force:
            print(f"[cached] {output_path}")
            continue
        records = read_records(input_path)
        artifact = score_established_metrics(
            records,
            metrics=tuple(evaluation.get("metrics", [])),
            batch_size=args.batch_size or int(evaluation.get("batch_size", 1024)),
            device=args.device or str(evaluation.get("device", "auto")),
            spice_failure_policy=str(evaluation.get("spice_failure_policy", "isolate")),
            spice_java_memory=str(evaluation.get("spice_java_memory", "16G")),
        )
        artifact["dataset"] = args.dataset
        artifact["setting"] = name
        write_json(output_path, artifact)
        print(f"Wrote {output_path} ({len(records)} rows)")


def register_caf_command(args: argparse.Namespace, settings: Settings) -> None:
    cache_path, pairs_path = _caf_paths(settings)
    key_mode = str(settings.section("caf").get("key_mode", "basename"))
    cache = PairScoreCache(cache_path, pairs_path, key_mode=key_mode)
    for dataset in args.datasets:
        pair_dir = settings.perturbation_dir / dataset
        output_dir = settings.score_dir / dataset
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in _set_names(pair_dir, args.sets):
            pair_path = pair_dir / f"{name}.jsonl"
            if not pair_path.is_file():
                continue
            output_path = output_dir / f"{name}.caf.json"
            if output_path.exists() and not args.force:
                print(f"[cached] {output_path}")
                continue
            records = read_records(pair_path)
            sidecar = cache.sidecar(records)
            sidecar.update({"dataset": dataset, "setting": name})
            write_json(output_path, sidecar)
            print(f"Registered {dataset}/{name}: {len(records)} aligned rows")
    print(f"Pair manifest: {pairs_path}")


def clap_command(args: argparse.Namespace, settings: Settings) -> None:
    cache_path, pairs_path = _caf_paths(settings)
    section = settings.section("caf")
    result = run_clap(
        pairs_path,
        cache_path,
        settings.caf_source,
        model_name=str(section.get("clap_model", "laionclap")),
        flush_every=int(section.get("clap_flush_every", 2000)),
        limit=args.limit,
    )
    print(json.dumps(result, indent=2))


def fleur_command(args: argparse.Namespace, settings: Settings) -> None:
    base_cache, pairs_path = _caf_paths(settings)
    section = settings.section("caf")
    cache_path = (
        Path(args.cache).resolve()
        if args.cache
        else base_cache.with_name(f"{base_cache.stem}_w{args.shard}{base_cache.suffix}")
    )
    result = run_fleur(
        pairs_path,
        cache_path,
        settings.caf_source,
        base_cache_path=base_cache,
        model_name=str(section.get("fleur_model", "audioflamingo3")),
        alpha=float(section.get("alpha", 0.8)),
        flush_every=int(section.get("fleur_flush_every", 100)),
        max_new_tokens=int(section.get("max_new_tokens", 16)),
        shard=args.shard,
        num_shards=args.num_shards,
        limit=args.limit,
    )
    print(json.dumps({"cache": str(cache_path), **result}, indent=2))


def merge_command(args: argparse.Namespace, settings: Settings) -> None:
    base_cache, _ = _caf_paths(settings)
    result = merge_shards(base_cache, args.shards)
    print(json.dumps(result, indent=2))


def finalize_command(settings: Settings) -> None:
    cache_path, pairs_path = _caf_paths(settings)
    cache = PairScoreCache(
        cache_path,
        pairs_path,
        key_mode=str(settings.section("caf").get("key_mode", "basename")),
    )
    paths = sorted(settings.score_dir.glob("*/*.caf.json"))
    if not paths:
        print("No CAF sidecars found.")
        return
    for path in paths:
        sidecar = cache.refill(read_json(path))
        write_json(path, sidecar)
        values = sidecar["metrics"]["caf_score"]["scores"]
        hits = sum(value is not None for value in values)
        print(f"{path.relative_to(settings.score_dir)}: {hits}/{len(values)}")


def main() -> None:
    args = parse_args()
    settings = Settings.load(args.config)
    settings.ensure_output_dirs()
    if args.command == "established":
        run_established_command(args, settings)
    elif args.command == "register-caf":
        register_caf_command(args, settings)
    elif args.command == "clap":
        clap_command(args, settings)
    elif args.command == "fleur":
        fleur_command(args, settings)
    elif args.command == "merge-caf":
        merge_command(args, settings)
    else:
        finalize_command(settings)


if __name__ == "__main__":
    main()
