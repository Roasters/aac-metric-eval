"""Resumable CLAP, FLEUR, and CAF computation over a registered pair manifest."""

from __future__ import annotations

import json
import shutil
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

from ..io import read_json, write_json
from .cache import pair_key


def load_pairs(path: str | Path) -> list[tuple[str, str, str]]:
    pairs: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("key") or pair_key(row["audio_path"], row["caption"])
            if key in seen:
                continue
            seen.add(key)
            pairs.append((key, row["audio_path"], row["caption"]))
    return pairs


def run_clap(
    pairs_path: str | Path,
    cache_path: str | Path,
    caf_source: str | Path,
    *,
    model_name: str = "laionclap",
    flush_every: int = 2_000,
    limit: int = 0,
) -> dict[str, int]:
    if flush_every <= 0:
        raise ValueError("flush_every must be positive")
    caf_source = str(Path(caf_source).resolve())
    if caf_source not in sys.path:
        sys.path.insert(0, caf_source)
    from src.clap import load_clap

    cache_path = Path(cache_path)
    cache: dict[str, dict[str, Any]] = read_json(cache_path) if cache_path.exists() else {}
    pairs = load_pairs(pairs_path)
    todo = [pair for pair in pairs if cache.get(pair[0], {}).get("clap_score") is None]
    if limit:
        todo = todo[:limit]
    if not todo:
        return {"pairs": len(pairs), "scored": 0, "missing_audio": 0}

    model = load_clap(model_name)
    by_audio: OrderedDict[str, list[tuple[str, str]]] = OrderedDict()
    for key, audio_path, caption in todo:
        by_audio.setdefault(audio_path, []).append((key, caption))

    scored = missing_audio = last_flush = 0
    for audio_path, items in by_audio.items():
        if not Path(audio_path).is_file():
            missing_audio += len(items)
            for key, caption in items:
                entry = cache.setdefault(key, {"audio_path": audio_path, "caption": caption})
                entry.update({"clap_score": None, "error": "audio_not_found"})
            continue
        captions = [caption for _, caption in items]
        try:
            similarities = model.get_similarity(
                audio_path, captions, use_sliding_window=False, pooling="max"
            )
            values = [
                max(float(similarities[0, index].detach().cpu()), 0.0)
                for index in range(len(captions))
            ]
        except Exception as exc:
            for key, caption in items:
                entry = cache.setdefault(key, {"audio_path": audio_path, "caption": caption})
                entry.update({"clap_score": None, "error": f"clap:{exc}"})
            continue
        for (key, caption), value in zip(items, values):
            entry = cache.setdefault(key, {"audio_path": audio_path, "caption": caption})
            entry["clap_score"] = value
            entry.pop("error", None)
            scored += 1
        if scored - last_flush >= flush_every:
            write_json(cache_path, cache)
            last_flush = scored
    write_json(cache_path, cache)
    return {"pairs": len(pairs), "scored": scored, "missing_audio": missing_audio}


def run_fleur(
    pairs_path: str | Path,
    cache_path: str | Path,
    caf_source: str | Path,
    *,
    base_cache_path: str | Path | None = None,
    model_name: str = "audioflamingo3",
    alpha: float = 0.8,
    flush_every: int = 100,
    max_new_tokens: int = 16,
    shard: int = 0,
    num_shards: int = 1,
    limit: int = 0,
) -> dict[str, int]:
    if num_shards <= 0 or not 0 <= shard < num_shards:
        raise ValueError("shard must be in [0, num_shards)")
    if flush_every <= 0:
        raise ValueError("flush_every must be positive")
    cache_path = Path(cache_path)
    if not cache_path.exists() and base_cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base_cache_path, cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"CLAP-filled cache does not exist: {cache_path}")

    caf_source = str(Path(caf_source).resolve())
    if caf_source not in sys.path:
        sys.path.insert(0, caf_source)
    from src.fleur import get_fleur, load_model as load_fleur_model
    import src.fleur.af3 as af3_module

    if max_new_tokens > 0:
        _patch_af3_max_new_tokens(af3_module, max_new_tokens)

    cache: dict[str, dict[str, Any]] = read_json(cache_path)
    pairs = load_pairs(pairs_path)[shard::num_shards]
    todo = [
        pair
        for pair in pairs
        if cache.get(pair[0], {}).get("clap_score") is not None
        and cache.get(pair[0], {}).get("fleur_score") is None
    ]
    if limit:
        todo = todo[:limit]
    missing_clap = sum(
        1 for key, _, _ in pairs if cache.get(key, {}).get("clap_score") is None
    )
    if not todo:
        return {"pairs": len(pairs), "scored": 0, "missing_clap": missing_clap}

    args = type("FleurArgs", (), {"use_think_mode": False, "lalm_model": model_name})()
    model = load_fleur_model(model_name, args)
    scored = 0
    for index, (key, audio_path, caption) in enumerate(todo, start=1):
        try:
            raw_fleur, fleur = get_fleur(model, caption, audio_path)
            fleur = float(fleur) if fleur is not None else None
        except Exception as exc:
            raw_fleur, fleur = None, None
            cache[key]["error"] = f"fleur:{exc}"
        entry = cache[key]
        entry["fleur_score"] = fleur
        entry["raw_fleur_score"] = raw_fleur
        entry["alpha"] = alpha
        entry["caf_score"] = (
            alpha * float(entry["clap_score"]) + (1.0 - alpha) * fleur
            if fleur is not None
            else None
        )
        scored += 1
        if index % flush_every == 0:
            write_json(cache_path, cache)
    write_json(cache_path, cache)
    return {"pairs": len(pairs), "scored": scored, "missing_clap": missing_clap}


def merge_shards(
    base_cache_path: str | Path,
    shard_paths: Iterable[str | Path],
) -> dict[str, int]:
    base_cache_path = Path(base_cache_path)
    base: dict[str, dict[str, Any]] = read_json(base_cache_path)
    fields = ("fleur_score", "raw_fleur_score", "caf_score", "alpha")
    merged = 0
    for shard_path in map(Path, shard_paths):
        if not shard_path.exists():
            continue
        shard: dict[str, dict[str, Any]] = read_json(shard_path)
        for key, value in shard.items():
            if value.get("fleur_score") is None:
                continue
            entry = base.setdefault(key, dict(value))
            for field in fields:
                if value.get(field) is not None:
                    entry[field] = value[field]
            merged += 1
    write_json(base_cache_path, base)
    return {
        "merged": merged,
        "cache_entries": len(base),
        "with_fleur": sum(1 for value in base.values() if value.get("fleur_score") is not None),
    }


def _patch_af3_max_new_tokens(af3_module: Any, max_new_tokens: int) -> None:
    """Patch the vendored AF3 adapter for bounded, dtype-safe score generation."""

    import torch
    from src.fleur.base import calculate_smoothed_score_torch, make_fleur_prompt, parse_raw_score

    def get_fleur_fast(model_wrapper, caption, audio, **_kwargs):
        model, processor = model_wrapper.model, model_wrapper.processor
        rate_to_token = model_wrapper.rate2token
        messages = make_fleur_prompt(audio, caption, audio_key="path")
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True
        ).to(model.device)
        for key, value in list(inputs.items()):
            if torch.is_tensor(value) and torch.is_floating_point(value):
                inputs[key] = value.to(model.dtype)
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            input_length = inputs.input_ids.shape[1]
            generated = outputs.sequences[0, input_length:]
            text = processor.batch_decode(
                outputs.sequences[:, input_length:], skip_special_tokens=True
            )[0].strip()
            if outputs.scores:
                logits = torch.stack(outputs.scores, dim=0).squeeze(1)
                return calculate_smoothed_score_torch(
                    text, logits, generated, rate_to_token
                )
            score = parse_raw_score(text)
            return score, score
        except Exception:
            return None, None

    original_loader = af3_module.load_model

    def load_patched(args, **kwargs):
        wrapper = original_loader(args, **kwargs)
        wrapper._get_fleur_fn = get_fleur_fast
        try:
            wrapper.model.model.audio_tower.to(torch.bfloat16)
        except Exception:
            pass
        return wrapper

    af3_module.get_fleur = get_fleur_fast
    af3_module.load_model = load_patched
