"""
Masking perturbation (meaning-destroying).

For each caption, masks each token position in turn by replacing the token with a
fixed non-word ("xkqjvz") that carries no meaning and matches no reference n-gram.
This tests whether metrics react to semantic removal or just surface-form change.

Writes RESULTS_DIR/masked_scores.json and RESULTS_DIR/masked_sentences.json.
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

# ── 2. Build masked (non-word) sentences ────────────────────────────────────
pred_replaced = []
gts = []
replaced_words = []
non_words = []
seq_lens = []
audio_paths_replaced = []
file_names_replaced = []

nonword = 'xkqjvz'

for i in range(len(all_pred)):
    pred = all_pred[i]
    gt = all_gt[i]
    tokens = pred.split()
    seq_lens.append(len(tokens))

    for j in range(len(tokens)):
        tokens_ = tokens[:]
        replaced_words.append(tokens_[j])
        non_words.append(nonword)
        tokens_[j] = nonword
        pred_replaced.append(" ".join(tokens_))
        gts.append(gt)
        audio_paths_replaced.append(audio_paths[i])
        file_names_replaced.append(all_file_names[i])

print(f"Total positions: {len(pred_replaced)}")
print(f"Example: '{all_pred[0]}' -> '{pred_replaced[0]}'")

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

# ── 4. Compute masked scores ────────────────────────────────────────────────
print(f"\nComputing scores for {len(pred_replaced)} masked sentences...")
replaced_scores_path = RESULTS_DIR / "masked_scores.json"
if replaced_scores_path.exists():
    print("  Loading cached masked scores...")
    replaced_scores = json.load(open(replaced_scores_path))
else:
    replaced_scores = evaluate_aac_batch(
        pred_replaced, gts, batch_size=1045, device=device,
        audio_paths=audio_paths_replaced, ignore_all=True,
    )
    with open(replaced_scores_path, "w") as f:
        json.dump(replaced_scores, f)

# ── 5. Save metadata ───────────────────────────────────────────────────────
meta_path = RESULTS_DIR / "masked_sentences.json"
with open(meta_path, "w") as f:
    json.dump({
        "seq_lens": seq_lens,
        "replaced_words": replaced_words,
        "non_words": non_words,
    }, f)
print(f"Saved metadata to: {meta_path}")

# ── 6. Summary statistics ──────────────────────────────────────────────────
METRICS = [
    "bleu_1", "bleu_2", "bleu_3", "bleu_4",
    "rouge_l", "meteor", "cider_d", "spice", "spider",
    "fense", "sbert_sim", "clap_sim_text", "clap_sim_audio",
]

# Expand original scores
orig_expanded = {}
for m in METRICS:
    scores = original_scores[m]["scores"]
    expanded = []
    for i, sl in enumerate(seq_lens):
        expanded.extend([scores[i]] * sl)
    orig_expanded[m] = np.array(expanded, dtype=float)

print(f"\n{'Metric':>18s}  {'orig_mean':>10s}  {'rand_mean':>10s}  {'mean_drop':>10s}  {'drop_%':>8s}")
print("-" * 65)

for m in METRICS:
    if m not in replaced_scores or m not in original_scores:
        continue

    o = orig_expanded[m]
    r = np.array(replaced_scores[m]["scores"], dtype=float)

    valid = ~(np.isnan(o) | np.isnan(r))
    ov, rv = o[valid], r[valid]

    orig_mean = ov.mean()
    rand_mean = rv.mean()
    drop = orig_mean - rand_mean
    drop_pct = 100 * drop / orig_mean if orig_mean != 0 else float("nan")

    print(f"{m:>18s}  {orig_mean:10.4f}  {rand_mean:10.4f}  {drop:10.4f}  {drop_pct:7.1f}%")

print(f"\nDone. Results saved to {RESULTS_DIR}/")
