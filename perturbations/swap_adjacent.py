"""
Perturbation test: swap each pair of adjacent tokens and measure score change.

For each caption with N words, produces N-1 swapped variants:
    "A machine whines and squeals" ->
        "machine A whines and squeals"   (swap pos 0-1)
        "A whines machine and squeals"   (swap pos 1-2)
        "A machine and whines squeals"   (swap pos 2-3)
        "A machine whines squeals and"   (swap pos 3-4)

Uses the same leave-one-out evaluation setup as masking_test.py.
"""

import numpy as np
import torch
import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Union
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluate import evaluate_aac_batch
from config import (RESULTS_DIR, FIGURES_DIR, clotho_captions_csv,
                    clotho_audio_path)


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# ── 1. Load Clotho evaluation captions (leave-one-out) ─────────────────────
all_pred = []
all_gt = []
all_file_names = []
audio_paths = []

for split in ["evaluation"]:
    with open(clotho_captions_csv(split)) as f:
        reader = csv.reader(f, delimiter=",")
        print(next(reader))  # Skip header
        for r in reader:
            file_name = r[1]
            for i in range(2, len(r)):
                all_pred.append(r[i])
                all_gt.append(r[2:i] + r[i + 1:])
                all_file_names.append(file_name)
                audio_paths.append(clotho_audio_path(file_name, split))

print(f"Loaded {len(all_pred)} leave-one-out caption pairs")

# ── 2. Build swapped sentences ──────────────────────────────────────────────
pred_swapped = []
gts = []
swapped_words = []      # (word_left, word_right) that were swapped
swap_positions = []      # position j where swap(j, j+1) occurred
seq_lens = []            # N-1 swaps per caption of length N
audio_paths_swapped = []
file_names_swapped = []

for i in range(len(all_pred)):
    tokens = all_pred[i].split()
    n_swaps = len(tokens) - 1
    seq_lens.append(n_swaps)

    for j in range(n_swaps):
        tokens_ = tokens[:]
        tokens_[j], tokens_[j + 1] = tokens_[j + 1], tokens_[j]
        pred_swapped.append(" ".join(tokens_))
        gts.append(all_gt[i])
        swapped_words.append((tokens[j], tokens[j + 1]))
        swap_positions.append(j)
        audio_paths_swapped.append(audio_paths[i])
        file_names_swapped.append(all_file_names[i])

total_swaps = len(pred_swapped)
print(f"Total swap positions: {total_swaps}")
print(f"Example: '{all_pred[0]}' -> '{pred_swapped[0]}'")

# ── 3. Compute original scores (reuse cached) ──────────────────────────────
print("\nComputing/loading original scores...")
original_scores_path = RESULTS_DIR / "original_scores.json"
if original_scores_path.exists():
    print("  Loading cached original scores...")
    original_scores = json.load(open(original_scores_path))
else:
    print("  Computing original scores...")
    original_scores = evaluate_aac_batch(
        all_pred, all_gt, batch_size=1045, device=device,
        audio_paths=audio_paths, ignore_all=True,
    )
    with open(original_scores_path, "w") as f:
        json.dump(original_scores, f)

# ── 4. Compute swapped scores ──────────────────────────────────────────────
print(f"\nComputing scores for {total_swaps} swapped sentences...")
swapped_scores_path = RESULTS_DIR / "swap_adjacent_scores.json"
if swapped_scores_path.exists():
    print("  Loading cached swapped scores...")
    swapped_scores = json.load(open(swapped_scores_path))
else:
    swapped_scores = evaluate_aac_batch(
        pred_swapped, gts, batch_size=1045, device=device,
        audio_paths=audio_paths_swapped, ignore_all=True,
    )
    with open(swapped_scores_path, "w") as f:
        json.dump(swapped_scores, f)

# ── 5. Save metadata ───────────────────────────────────────────────────────
meta_path = RESULTS_DIR / "swap_adjacent_meta.json"
with open(meta_path, "w") as f:
    json.dump({
        "seq_lens": seq_lens,
        "swapped_words": swapped_words,
        "swap_positions": swap_positions,
    }, f)
print(f"Saved metadata to: {meta_path}")

# ── 6. Summary statistics ──────────────────────────────────────────────────
METRICS = [
    "bleu_1", "bleu_2", "bleu_3", "bleu_4",
    "rouge_l", "meteor", "cider_d", "spice", "spider",
    "fense", "clap_sim_text", "sbert_sim", "clap_sim_audio",
]
METRIC_LABELS = {
    "bleu_1": "BLEU-1", "bleu_2": "BLEU-2", "bleu_3": "BLEU-3", "bleu_4": "BLEU-4",
    "rouge_l": "ROUGE-L", "meteor": "METEOR", "cider_d": "CIDEr-D",
    "spice": "SPICE", "spider": "SPIDEr",
    "fense": "FENSE", "clap_sim_text": "CLAP-T", "sbert_sim": "SBERT",
    "clap_sim_audio": "CLAP-A",
}

# Expand original scores to match swap-level entries
orig_expanded = {}
for m in METRICS:
    scores = original_scores[m]["scores"]
    expanded = []
    for i, sl in enumerate(seq_lens):
        expanded.extend([scores[i]] * sl)
    orig_expanded[m] = np.array(expanded, dtype=float)

print(f"\n{'Metric':>18s}  {'orig_mean':>10s}  {'swap_mean':>10s}  {'abs_drop':>10s}  {'rel_drop':>10s}")
print("-" * 70)

for m in METRICS:
    if m not in swapped_scores or m not in original_scores:
        continue

    o = orig_expanded[m]
    s = np.array(swapped_scores[m]["scores"], dtype=float)

    # Absolute drop
    valid = ~(np.isnan(o) | np.isnan(s))
    ov, sv = o[valid], s[valid]
    abs_drop = (ov - sv).mean()

    # Relative drop (ratio, clipped, consistent with other experiments)
    o_safe = np.where(o == 0, np.nan, o)
    ratio = np.clip(s / o_safe, 0, 2)
    valid_rel = ~np.isnan(ratio)
    rel_drop = (1.0 - ratio[valid_rel]).mean()

    print(f"{METRIC_LABELS.get(m, m):>18s}  {ov.mean():10.4f}  {sv.mean():10.4f}  {abs_drop:10.4f}  {rel_drop*100:9.2f}%")

print(f"\nDone. Results saved to {RESULTS_DIR}/")
