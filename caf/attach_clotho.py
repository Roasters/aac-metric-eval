"""
Attach CAF sidecars to every Clotho scoring experiment WITHOUT re-running the heavy
metric computation.

Each experiment's perturbed candidate set is reconstructed exactly from the artifacts
it already cached in RESULTS_DIR (the *_sentences.json / *_meta.json companions plus
the leave-one-out originals), in the SAME row order as its *_scores_*.json. For each
experiment we write RESULTS_DIR/<name>_caf.json via caf_lookup.dump_sidecar, which:
  - registers every unique (audio, caption) pair to results/caf_pairs.jsonl (pass 1), and
  - fills CAF/CLAP/FLEUR from results/caf_cache.json once it exists (pass 2, or run
    caf/finalize_sidecars.py).

A length assertion against the base score JSON guards every reconstruction.

Run (base env):
    python caf/attach_clotho.py            # pass 1 (no cache) -> registers pairs
    # ... build cache with caf/run_caf_af3.py ...
    python caf/finalize_sidecars.py        # pass 2 -> fills values
"""

import os
import csv
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from caf_lookup import dump_sidecar, RESULTS_DIR  # noqa: E402
from config import clotho_captions_csv, clotho_audio_dir  # noqa: E402

CLOTHO_EVAL_CSV = str(clotho_captions_csv("evaluation"))
AUDIO_DIR = clotho_audio_dir("evaluation")


def load_originals():
    """Leave-one-out candidates, in the exact order the experiments build all_pred."""
    all_pred, audio_paths = [], []
    with open(CLOTHO_EVAL_CSV) as f:
        reader = csv.reader(f)
        next(reader)  # header
        for r in reader:
            fname = r[1]
            for i in range(2, len(r)):
                all_pred.append(r[i])
                audio_paths.append(os.path.join(AUDIO_DIR, fname))
    return all_pred, audio_paths


def _load(name):
    p = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _base_len(scores_json):
    """Row count of a base score file (first real metric)."""
    if scores_json is None:
        return None
    for m, v in scores_json.items():
        if isinstance(v, dict) and "scores" in v:
            return len(v["scores"])
    return None


def recon_removal(all_pred, audio_paths):
    """masking_removal: drop each token position in turn."""
    cands, aps = [], []
    for pred, ap in zip(all_pred, audio_paths):
        toks = pred.split()
        for j in range(len(toks)):
            cands.append(" ".join(toks[:j] + toks[j + 1:])); aps.append(ap)
    return cands, aps


def recon_wordsub(all_pred, audio_paths, meta, sub_key):
    """masked / random_word: replace token j with the saved substitute word.

    Positions are nested (for each original, for each token) and the substitute list
    is aligned to that nesting.
    """
    subs = meta[sub_key]
    cands, aps = [], []
    k = 0
    for pred, ap in zip(all_pred, audio_paths):
        toks = pred.split()
        for j in range(len(toks)):
            t = toks[:]; t[j] = subs[k]
            cands.append(" ".join(t)); aps.append(ap)
            k += 1
    return cands, aps


def recon_swap(all_pred, audio_paths, meta):
    """swap_adjacent_test: swap tokens (pos, pos+1) for each saved swap position."""
    positions = meta["swap_positions"]
    seq_lens = meta["seq_lens"]  # n_swaps per original
    cands, aps = [], []
    k = 0
    for i, (pred, ap) in enumerate(zip(all_pred, audio_paths)):
        toks = pred.split()
        for _ in range(seq_lens[i]):
            j = positions[k]
            t = toks[:]
            t[j], t[j + 1] = t[j + 1], t[j]
            cands.append(" ".join(t)); aps.append(ap)
            k += 1
    return cands, aps


SAMPLE = int(os.environ.get("CAF_SAMPLE", "0"))  # 0 = full; else cap perturbation rows
# Core = the three perturbations mapping to the paper's three axes (+ originals).
CORE_ONLY = os.environ.get("CAF_CORE_ONLY", "0") == "1"
CORE_NAMES = {"original", "masked", "synonym", "synonym_mlm", "swap_adjacent"}


def _evenly_spaced(n, k):
    """k evenly-spaced indices in [0, n) (deterministic)."""
    if k >= n:
        return list(range(n))
    step = n / k
    return sorted(set(int(i * step) for i in range(k)))


def attach(name, cands, aps, base_scores_file, sampleable=True):
    if CORE_ONLY and name not in CORE_NAMES:
        return
    base = _load(base_scores_file)
    blen = _base_len(base)
    if blen is not None and blen != len(cands):
        print(f"  [WARN] {name}: reconstructed {len(cands)} rows != base {blen} "
              f"({base_scores_file}); SKIPPING to avoid misalignment.")
        return
    indices = None
    if sampleable and SAMPLE and len(cands) > SAMPLE:
        indices = _evenly_spaced(len(cands), SAMPLE)
        cands = [cands[i] for i in indices]
        aps = [aps[i] for i in indices]
    out = os.path.join(RESULTS_DIR, f"{name}_caf.json")
    hit, total = dump_sidecar(cands, aps, out, indices=indices)
    tag = f"(sampled {total} of {blen})" if indices is not None else ""
    print(f"  {name:24s} rows={total:7d} base={blen} cache_hit={hit}/{total} {tag}")


def main():
    all_pred, audio_paths = load_originals()
    print(f"originals (leave-one-out): {len(all_pred)} rows")

    # 1. Originals (shared leave-one-out set) — always full, never sampled
    attach("original", all_pred, audio_paths, "original_scores.json", sampleable=False)

    # 2. Removal
    c, a = recon_removal(all_pred, audio_paths)
    attach("removal", c, a, "removal_scores.json")

    # 3. Masking (non-word substitution, meaning-destroying) / random-word control
    for name, sents_file, sub_key, base in [
        ("masked", "masked_sentences.json", "non_words", "masked_scores.json"),
        ("random_word", "random_word_sentences.json", "random_words", "random_word_scores.json"),
    ]:
        meta = _load(sents_file)
        if meta is None:
            print(f"  [skip] {name}: {sents_file} missing"); continue
        c, a = recon_wordsub(all_pred, audio_paths, meta, sub_key)
        attach(name, c, a, base)

    # 4. Swap adjacent (word-order)
    meta = _load("swap_adjacent_meta.json")
    if meta is not None:
        c, a = recon_swap(all_pred, audio_paths, meta)
        attach("swap_adjacent", c, a, "swap_adjacent_scores.json")

    # 5. Synonym families (meaning-preserving) — candidates saved directly
    for name, sents_file, base in [
        ("synonym", "synonym_sentences.json", "synonym_scores.json"),
        ("synonym_mlm", "synonym_mlm_sentences.json", "synonym_mlm_scores.json"),
        ("synonym_mlm_filtered", "synonym_mlm_filtered_sentences.json", "synonym_mlm_filtered_scores.json"),
    ]:
        meta = _load(sents_file)
        if meta is None:
            print(f"  [skip] {name}: {sents_file} missing"); continue
        attach(name, meta["pred_replaced"], meta["audio_paths_replaced"], base)

    print("\nDone. Unique pairs registered to results/caf_pairs.jsonl for CAF scoring.")


if __name__ == "__main__":
    main()
