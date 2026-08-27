"""
Build the CAF-Score cache with Audio-Flamingo-3 (runs in the caf_af3 conda env).

Reads results/caf_pairs.jsonl (produced by experiments via caf/caf_lookup.py), computes
CLAP (LAION-CLAP) + FLEUR (Audio-Flamingo-3) + CAF for every unique pair, and writes
results/caf_cache.json keyed the same way as caf_lookup.caf_key.

    CAF = alpha * CLAP + (1 - alpha) * FLEUR      (alpha default 0.8)

AF3 is loaded with device_map="auto"; set CUDA_VISIBLE_DEVICES=0,1 to shard it across
the two RTX 3090s. Run:

    CUDA_VISIBLE_DEVICES=0,1 conda run -n caf_af3 \
        python caf/run_caf_af3.py --pairs results/caf_pairs.jsonl \
                                  --cache results/caf_cache.json

Incremental + resumable: pairs already in the cache are skipped; the cache is flushed
to disk every --flush_every pairs.
"""

import os
import sys
import json
import time
import argparse
import hashlib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAF_SRC = os.environ.get("CAF_SRC", os.path.join(REPO, "third_party/CAF-Score"))
sys.path.insert(0, CAF_SRC)

_SEP = "\x1f"


def caf_key(audio_path: str, caption: str) -> str:
    base = os.path.basename(audio_path) if audio_path else ""
    return hashlib.sha1((base + _SEP + (caption or "")).encode("utf-8")).hexdigest()


def load_pairs(path):
    pairs = []
    seen = set()
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
    ap.add_argument("--lalm_model", default="audioflamingo3")
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--flush_every", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=16,
                    help="cap AF3 generation length (FLEUR needs only the score digits)")
    ap.add_argument("--shard", type=int, default=0, help="this worker's shard index")
    ap.add_argument("--num_shards", type=int, default=1, help="total data-parallel workers")
    ap.add_argument("--limit", type=int, default=0, help="score at most N pairs (0=all); for smoke tests")
    args = ap.parse_args()

    from src.fleur import load_model as load_fleur_model, get_fleur
    # Cap AF3 generation length: FLEUR only needs the score digits ("0.85"), which the
    # prompt forces to appear immediately. 512 tokens (CAF default) is wasteful.
    import src.fleur.af3 as _af3
    if args.max_new_tokens and args.max_new_tokens > 0:
        _patch_af3_max_new_tokens(_af3, args.max_new_tokens)

    os.makedirs(os.path.dirname(args.cache), exist_ok=True)
    cache = {}
    if os.path.exists(args.cache):
        with open(args.cache) as f:
            cache = json.load(f)

    pairs = load_pairs(args.pairs)
    if args.num_shards > 1:
        pairs = pairs[args.shard::args.num_shards]
    # Only score pairs that (a) still lack FLEUR and (b) have a CLAP score from run_clap.
    todo = [(k, a, c) for (k, a, c) in pairs
            if k in cache and cache[k].get("clap_score") is not None
            and cache[k].get("fleur_score") is None]
    no_clap = sum(1 for (k, a, c) in pairs
                  if k not in cache or cache[k].get("clap_score") is None)
    if args.limit:
        todo = todo[:args.limit]
    print(f"pairs={len(pairs)} need_fleur={len(todo)} (missing_clap={no_clap} -> run caf/run_clap.py first)",
          flush=True)
    if not todo:
        print("nothing to do."); return

    class A:
        pass
    a = A(); a.use_think_mode = False; a.lalm_model = args.lalm_model
    print(f"loading LALM={args.lalm_model} (device_map=auto) ...", flush=True)
    fm = load_fleur_model(args.lalm_model, a)

    t0 = time.time(); done = 0
    for i, (k, audio_path, caption) in enumerate(todo):
        try:
            raw_fleur, fleur = get_fleur(fm, caption, audio_path)
            fleur = float(fleur) if fleur is not None else None
        except Exception:
            fleur, raw_fleur = None, None
        entry = cache[k]
        entry["fleur_score"] = fleur
        entry["raw_fleur_score"] = raw_fleur
        entry["alpha"] = args.alpha
        clap_score = entry.get("clap_score")
        entry["caf_score"] = (args.alpha * clap_score + (1 - args.alpha) * fleur
                              if fleur is not None else None)
        done += 1
        if done % args.flush_every == 0:
            with open(args.cache, "w") as f:
                json.dump(cache, f)
            rate = done / (time.time() - t0)
            eta = (len(todo) - i - 1) / rate / 60 if rate > 0 else -1
            print(f"[{i+1}/{len(todo)}] fleur_done={done} rate={rate:.2f}/s eta={eta:.1f}min", flush=True)

    with open(args.cache, "w") as f:
        json.dump(cache, f)
    print(f"DONE fleur_done={done} cache={len(cache)} elapsed={(time.time()-t0)/60:.1f}min -> {args.cache}",
          flush=True)


def _patch_af3_max_new_tokens(af3_module, max_new_tokens):
    """Wrap AF3.get_fleur so model.generate uses a small max_new_tokens."""
    import torch
    from src.fleur.base import parse_raw_score, calculate_smoothed_score_torch, make_fleur_prompt

    def get_fleur_fast(fm, caption, audio, **kwargs):
        model, processor, rate2token = fm.model, fm.processor, fm.rate2token
        messages = make_fleur_prompt(audio, caption, audio_key="path")
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True
        ).to(model.device)
        # AF3-hf processor emits float32 audio input_features while the model runs in
        # bf16 -> conv layers hit "Input type (float) and bias type (BFloat16)". Cast
        # floating-point inputs (not int ids/masks) to the model dtype.
        for _k in list(inputs.keys()):
            _v = inputs[_k]
            if torch.is_tensor(_v) and torch.is_floating_point(_v):
                inputs[_k] = _v.to(model.dtype)
        try:
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                         do_sample=False, return_dict_in_generate=True,
                                         output_scores=True)
            input_len = inputs.input_ids.shape[1]
            gen_ids = outputs.sequences[0, input_len:]
            text = processor.batch_decode(outputs.sequences[:, input_len:],
                                          skip_special_tokens=True)[0].strip()
            if outputs.scores:
                logits = torch.stack(outputs.scores, dim=0).squeeze(1)
                return calculate_smoothed_score_torch(text, logits, gen_ids, rate2token)
            s = parse_raw_score(text); return s, s
        except Exception as e:
            print(f"  gen error: {e}", flush=True)
            return None, None

    af3_module.get_fleur = get_fleur_fast
    # also patch the FleurModel dispatch target used by src.fleur.get_fleur
    import src.fleur as _f
    _orig_loader = af3_module.load_model
    def load_patched(args, **kw):
        fm = _orig_loader(args, **kw)
        fm._get_fleur_fn = get_fleur_fast
        # AF3-hf loads audio_tower.embed_positions in float32 while the rest is bf16;
        # adding bf16 conv output to the fp32 positional embeds throws. Cast the whole
        # audio tower to bf16 so it's uniform with the (bf16-cast) input_features.
        try:
            fm.model.model.audio_tower.to(torch.bfloat16)
        except Exception as e:
            print(f"  [warn] could not cast audio_tower to bf16: {e}", flush=True)
        return fm
    af3_module.load_model = load_patched


if __name__ == "__main__":
    main()
