"""
Removal variant of the masking perturbation: removes each token entirely
instead of substituting the non-word (see perturbations/masking.py).

"A machine whines and squeals" ->
    "machine whines and squeals"       (remove pos 0)
    "A whines and squeals"             (remove pos 1)
    "A machine and squeals"            (remove pos 2)
    ...
"""

import numpy as np
import torch
import csv
import json
from pathlib import Path
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
        print(next(reader))
        for r in reader:
            file_name = r[1]
            for i in range(2, len(r)):
                all_pred.append(r[i])
                all_gt.append(r[2:i] + r[i + 1:])
                all_file_names.append(file_name)
                audio_paths.append(clotho_audio_path(file_name, split))

# ── 2. Build token-removed sentences ───────────────────────────────────────
pred_removed = []
gts = []
removed_words = []
seq_lens = []
audio_paths_removed = []
file_names_removed = []

for i in range(len(all_pred)):
    tokens = all_pred[i].split()
    seq_lens.append(len(tokens))

    for j in range(len(tokens)):
        removed_words.append(tokens[j])
        pred_removed.append(" ".join(tokens[:j] + tokens[j + 1:]))
        gts.append(all_gt[i])
        audio_paths_removed.append(audio_paths[i])
        file_names_removed.append(all_file_names[i])

print(f"Total positions: {len(pred_removed)}")
print(f"Example: '{all_pred[0]}' -> '{pred_removed[0]}'")

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

# ── 4. Compute removal scores ──────────────────────────────────────────────
print(f"\nComputing scores for {len(pred_removed)} token-removed sentences...")
removed_scores_path = RESULTS_DIR / "removal_scores.json"
if removed_scores_path.exists():
    print("  Loading cached removal scores...")
    removed_scores = json.load(open(removed_scores_path))
else:
    removed_scores = evaluate_aac_batch(
        pred_removed, gts, batch_size=1045, device=device,
        audio_paths=audio_paths_removed, ignore_all=True,
    )
    with open(removed_scores_path, "w") as f:
        json.dump(removed_scores, f)

# ── 5. Save metadata ───────────────────────────────────────────────────────
meta_path = RESULTS_DIR / "removal_sentences.json"
with open(meta_path, "w") as f:
    json.dump({
        "seq_lens": seq_lens,
        "removed_words": removed_words,
    }, f)
print(f"Saved metadata to: {meta_path}")

# ── 6. Summary statistics ──────────────────────────────────────────────────
METRICS = [
    "bleu_1", "bleu_2", "bleu_3", "bleu_4",
    "rouge_l", "meteor", "cider_d", "spice", "spider",
    "fense", "clap_sim_text", "sbert_sim", "clap_sim_audio",
]

orig_expanded = {}
for m in METRICS:
    scores = original_scores[m]["scores"]
    expanded = []
    for i, sl in enumerate(seq_lens):
        expanded.extend([scores[i]] * sl)
    orig_expanded[m] = np.array(expanded, dtype=float)

print(f"\n{'Metric':>18s}  {'orig_mean':>10s}  {'rem_mean':>10s}  {'mean_drop':>10s}  {'drop_%':>8s}")
print("-" * 65)

for m in METRICS:
    if m not in removed_scores or m not in original_scores:
        continue

    o = orig_expanded[m]
    r = np.array(removed_scores[m]["scores"], dtype=float)

    valid = ~(np.isnan(o) | np.isnan(r))
    ov, rv = o[valid], r[valid]

    orig_mean = ov.mean()
    rem_mean = rv.mean()
    drop = orig_mean - rem_mean
    drop_pct = 100 * drop / orig_mean if orig_mean != 0 else float("nan")

    print(f"{m:>18s}  {orig_mean:10.4f}  {rem_mean:10.4f}  {drop:10.4f}  {drop_pct:7.1f}%")

print(f"\nDone. Results saved to {RESULTS_DIR}/")
