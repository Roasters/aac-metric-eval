"""
Compute the CLAP term of CAF-Score for every pair in the manifest (runs in the BASE
env, where transformers 4.51 + CAF's ClapModel path work correctly).

Writes/updates results/caf_cache.json with `clap_score` per key. FLEUR + final CAF are
added afterwards by caf/run_caf_af3.py (caf_af3 env).

    CUDA_VISIBLE_DEVICES=0 python caf/run_clap.py \
        --pairs results/caf_pairs.jsonl --cache results/caf_cache.json
"""

import os
import sys
import json
import time
import argparse
import hashlib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("CAF_SRC", os.path.join(REPO, "third_party/CAF-Score")))
_SEP = "\x1f"


def caf_key(audio_path, caption):
    base = os.path.basename(audio_path) if audio_path else ""
    return hashlib.sha1((base + _SEP + (caption or "")).encode("utf-8")).hexdigest()


def load_pairs(path):
    pairs, seen = [], set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            k = d.get("key") or caf_key(d["audio_path"], d["caption"])
            if k in seen:
                continue
            seen.add(k)
            pairs.append((k, d["audio_path"], d["caption"]))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=os.path.join(REPO, "results/caf_pairs.jsonl"))
    ap.add_argument("--cache", default=os.path.join(REPO, "results/caf_cache.json"))
    ap.add_argument("--clap_model", default="laionclap")
    ap.add_argument("--flush_every", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from src.clap import load_clap

    os.makedirs(os.path.dirname(args.cache), exist_ok=True)
    cache = {}
    if os.path.exists(args.cache):
        with open(args.cache) as f:
            cache = json.load(f)

    pairs = load_pairs(args.pairs)
    todo = [(k, a, c) for (k, a, c) in pairs
            if k not in cache or cache[k].get("clap_score") is None]
    if args.limit:
        todo = todo[:args.limit]
    print(f"pairs={len(pairs)} need_clap={len(todo)}", flush=True)
    if not todo:
        print("nothing to do."); return

    print(f"loading CLAP={args.clap_model} ...", flush=True)
    clap = load_clap(args.clap_model)

    # Group by audio so each file is loaded/encoded once; captions are batched.
    from collections import OrderedDict
    by_audio = OrderedDict()
    for k, ap, cap in todo:
        by_audio.setdefault(ap, []).append((k, cap))

    t0 = time.time(); done = 0; missing = 0
    n_aud = len(by_audio)
    for ai, (audio_path, items) in enumerate(by_audio.items()):
        if not os.path.exists(audio_path):
            missing += len(items)
            for k, cap in items:
                e = cache.setdefault(k, {"audio_path": audio_path, "caption": cap})
                e["clap_score"] = None; e["error"] = "audio_not_found"
            continue
        caps = [cap for _, cap in items]
        try:
            sim = clap.get_similarity(audio_path, caps, use_sliding_window=False, pooling="max")
            vals = [max(float(sim[0, j].cpu()), 0.0) for j in range(len(caps))]
        except Exception as ex:
            for k, cap in items:
                e = cache.setdefault(k, {"audio_path": audio_path, "caption": cap})
                e["clap_score"] = None; e["error"] = f"clap:{ex}"
            continue
        for (k, cap), v in zip(items, vals):
            e = cache.setdefault(k, {"audio_path": audio_path, "caption": cap})
            e["clap_score"] = v
            done += 1
        if (ai + 1) % max(1, args.flush_every // 20) == 0:
            with open(args.cache, "w") as f:
                json.dump(cache, f)
            rate = done / (time.time() - t0)
            eta = (len(todo) - done) / rate / 60 if rate else -1
            print(f"[audio {ai+1}/{n_aud}] clap_done={done} rate={rate:.1f}/s eta={eta:.1f}min", flush=True)

    with open(args.cache, "w") as f:
        json.dump(cache, f)
    print(f"DONE clap_done={done} missing_audio={missing} rate={done/(time.time()-t0):.1f}/s "
          f"elapsed={(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
