"""
CAF-Score lookup (runs in the BASE env — no LALM/heavy deps).

CAF is expensive (an Audio-Flamingo-3 forward pass per pair), so it is computed once
per unique (audio_path, caption) pair by caf/run_caf_af3.py and cached to
results/caf_cache.json. Experiments call `caf_scores_for(candidates, audio_paths)` to
attach CAF to their outputs.

Two-pass workflow:
  Pass 1 (collect): cache does not exist yet -> every requested pair is registered to
                    results/caf_pairs.jsonl and NaN is returned. Running all
                    experiments once populates the full pair manifest.
  Build:            caf/run_caf_af3.py (caf_af3 env) consumes caf_pairs.jsonl -> caf_cache.json.
  Pass 2 (resolve): cache exists -> real CAF values are returned; any still-missing
                    pair is re-registered so it can be filled on the next build.

Key = sha1(basename(audio_path) + "\\x1f" + caption). Audio basename is used (not the
full path) so Clotho/AudioCaps paths remain stable across machines.
"""

import os
import json
import hashlib
import threading

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.environ.get("CAF_RESULTS_DIR",
                             os.environ.get("RESULTS_DIR", os.path.join(REPO, "results")))
CACHE_PATH = os.path.join(RESULTS_DIR, "caf_cache.json")
PAIRS_PATH = os.path.join(RESULTS_DIR, "caf_pairs.jsonl")

_SEP = "\x1f"
_lock = threading.Lock()
_cache = None
_registered = None  # set of keys already written to caf_pairs.jsonl this session


def caf_key(audio_path: str, caption: str) -> str:
    base = os.path.basename(audio_path) if audio_path else ""
    h = hashlib.sha1((base + _SEP + (caption or "")).encode("utf-8"))
    return h.hexdigest()


def load_cache(force: bool = False) -> dict:
    global _cache
    if _cache is None or force:
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH) as f:
                _cache = json.load(f)
        else:
            _cache = {}
    return _cache


def _load_registered() -> set:
    global _registered
    if _registered is None:
        _registered = set()
        if os.path.exists(PAIRS_PATH):
            with open(PAIRS_PATH) as f:
                for line in f:
                    try:
                        _registered.add(json.loads(line)["key"])
                    except Exception:
                        pass
    return _registered


def _register(pairs):
    """Append unseen (audio_path, caption) pairs to the manifest for later scoring."""
    reg = _load_registered()
    os.makedirs(os.path.dirname(PAIRS_PATH), exist_ok=True)
    with _lock, open(PAIRS_PATH, "a") as f:
        for audio_path, caption, k in pairs:
            if k in reg:
                continue
            reg.add(k)
            f.write(json.dumps({"key": k, "audio_path": audio_path,
                                "caption": caption}) + "\n")


def caf_scores_for(candidates, audio_paths, field: str = "caf_score",
                   register_misses: bool = True):
    """Return a list of CAF (or component) scores aligned to `candidates`.

    Args:
        candidates:  list[str] captions being scored.
        audio_paths: list[str] audio path per candidate (same length).
        field:       which cached component to return
                     ('caf_score' | 'clap_score' | 'fleur_score').
        register_misses: append missing pairs to caf_pairs.jsonl (NaN returned).
    """
    assert len(candidates) == len(audio_paths), "candidates/audio_paths length mismatch"
    cache = load_cache()
    out = []
    misses = []
    for cap, ap in zip(candidates, audio_paths):
        k = caf_key(ap, cap)
        entry = cache.get(k)
        if entry is not None and entry.get(field) is not None:
            out.append(float(entry[field]))
        else:
            out.append(float("nan"))
            if register_misses:
                misses.append((ap, cap, k))
    if register_misses and misses:
        _register(misses)
    return out


def coverage(candidates, audio_paths) -> tuple:
    """(n_hit, n_total) — how many pairs are resolvable from the current cache."""
    cache = load_cache()
    hit = sum(1 for cap, ap in zip(candidates, audio_paths)
              if cache.get(caf_key(ap, cap), {}).get("caf_score") is not None)
    return hit, len(candidates)


def dump_sidecar(candidates, audio_paths, out_path, indices=None):
    """Write a CAF sidecar {caf_score,clap_score,fleur_score: {scores:[...]}} aligned
    to `candidates`, independent of the base-metric result cache.

    Idempotent + cheap (cache dict lookups). On pass 1 (no cache) it registers all
    pairs to caf_pairs.jsonl and writes NaNs; on pass 2 (cache built) it writes real
    values. Returns (n_hit, n_total).

    `indices`, if given, is stored so analysis can align these (possibly subsampled)
    CAF rows back to the full base-score array positions.
    """
    caf = caf_scores_for(candidates, audio_paths, field="caf_score")
    clap = caf_scores_for(candidates, audio_paths, field="clap_score", register_misses=False)
    fleur = caf_scores_for(candidates, audio_paths, field="fleur_score", register_misses=False)
    keys = [caf_key(ap, cap) for cap, ap in zip(candidates, audio_paths)]
    out = {"keys": keys,
           "caf_score": {"scores": caf},
           "clap_score": {"scores": clap},
           "fleur_score": {"scores": fleur}}
    if indices is not None:
        out["indices"] = list(indices)
    os.makedirs(os.path.dirname(str(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f)
    hit = sum(1 for v in caf if v == v)  # non-NaN
    return hit, len(caf)


def refill_sidecar(sidecar_path) -> tuple:
    """Refill an existing CAF sidecar's score lists from the current cache, using the
    stored per-position keys. Lets pass-2 finalize CAF values without re-running the
    (expensive) experiment. Returns (n_hit, n_total)."""
    with open(sidecar_path) as f:
        sc = json.load(f)
    keys = sc.get("keys")
    if not keys:
        raise ValueError(f"{sidecar_path} has no 'keys'; re-run the experiment to regenerate it.")
    cache = load_cache(force=True)
    for field in ("caf_score", "clap_score", "fleur_score"):
        sc[field] = {"scores": [
            (float(cache[k][field]) if (k in cache and cache[k].get(field) is not None)
             else float("nan"))
            for k in keys
        ]}
    with open(sidecar_path, "w") as f:
        json.dump(sc, f)
    hit = sum(1 for v in sc["caf_score"]["scores"] if v == v)
    return hit, len(keys)
