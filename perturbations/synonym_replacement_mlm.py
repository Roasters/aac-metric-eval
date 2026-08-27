"""
Synonym replacement using MLM (masked language model) for synonym ranking
instead of SBERT, to avoid bias toward SBERT-based evaluation metrics
(FENSE, SBERT-sim).

For each word in each caption:
  1. Get all WordNet synonyms (filtered by POS tag)
  2. Mask the target word in the sentence
  3. Run a masked language model (RoBERTa) to get token probabilities
  4. Rank synonyms by their MLM probability in context
  5. Pick the highest-probability synonym

This ensures synonym selection is independent of all 13 evaluation metrics.
"""

import numpy as np
import torch
import csv
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import evaluate_aac_batch
from config import (RESULTS_DIR, FIGURES_DIR, clotho_captions_csv,
                    clotho_audio_path)
from typing import List, Dict, Any, Union
import json
import nltk
from nltk.corpus import wordnet
from transformers import pipeline

nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)


MLM_MODEL = "roberta-base"

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def write_json(data: Union[List[Dict[str, Any]], Dict[str, Any]], path: Path) -> None:
    with path.open("w") as f:
        json.dump(data, f)


# --- Penn Treebank POS → WordNet POS mapping ---
def get_wordnet_pos(treebank_tag: str):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    if treebank_tag.startswith('V'):
        return wordnet.VERB
    if treebank_tag.startswith('N'):
        return wordnet.NOUN
    if treebank_tag.startswith('R'):
        return wordnet.ADV
    return None


def get_all_synonyms(word: str, pos=None) -> list[str]:
    """Return all unique WordNet synonyms of `word` (excluding the word itself)."""
    synsets = wordnet.synsets(word, pos=pos) if pos else wordnet.synsets(word)
    candidates = set()
    for synset in synsets:
        for lemma in synset.lemmas():
            candidate = lemma.name().replace('_', ' ').lower()
            if candidate != word.lower():
                candidates.add(candidate)
    return list(candidates)


def get_best_synonym_mlm(word: str, sentence: str, pos=None, mlm_pipe=None) -> str:
    """Return the WordNet synonym with the highest MLM probability in context.
    Returns None if no synonym is found."""
    candidates = get_all_synonyms(word, pos=pos)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    if mlm_pipe is None:
        return candidates[0]

    tokens = sentence.split()
    try:
        idx = tokens.index(word)
    except ValueError:
        return candidates[0]

    # Build masked sentence
    tokens_masked = tokens[:]
    tokens_masked[idx] = mlm_pipe.tokenizer.mask_token
    masked_sentence = " ".join(tokens_masked)

    # Get MLM predictions — request enough to cover candidates
    try:
        predictions = mlm_pipe(masked_sentence, top_k=250)
    except Exception:
        return candidates[0]

    # Build a map of token_str -> score from MLM output
    token_scores = {}
    for pred in predictions:
        token_str = pred["token_str"].strip().lower()
        token_scores[token_str] = pred["score"]

    # Score each candidate by MLM probability
    best_syn = None
    best_score = -1.0
    for syn in candidates:
        # For multi-word synonyms, use the first word
        syn_key = syn.split()[0].lower()
        score = token_scores.get(syn_key, 0.0)
        if score > best_score:
            best_score = score
            best_syn = syn

    # If no candidate appeared in MLM top-k, fall back to first candidate
    if best_score == 0.0:
        return candidates[0]

    return best_syn


# ── 1. Load Clotho evaluation captions ─────────────────────────────────────
all_pred = []
all_gt = []
all_file_names = []
audio_paths = []

for split in ["evaluation"]:
    with open(clotho_captions_csv(split)) as f:
        reader = csv.reader(f, delimiter=',')
        print(next(reader))
        for r in reader:
            file_name = r[1]
            for i in range(2, len(r)):
                all_pred.append(r[i])
                all_gt.append(r[2:i] + r[i+1:])
                all_file_names.append(file_name)
                audio_paths.append(clotho_audio_path(file_name, split))

print(f"Loaded {len(all_pred)} leave-one-out caption pairs")

# ── 2. Build synonym-replaced sentences ────────────────────────────────────
synonym_sentences_path = RESULTS_DIR / "synonym_mlm_sentences.json"

if synonym_sentences_path.exists():
    print("Loading cached MLM synonym-replaced sentences...")
    cached = json.load(open(synonym_sentences_path, "r"))
    pred_replaced       = cached["pred_replaced"]
    gts                 = cached["gts"]
    replaced_words      = cached["replaced_words"]
    synonym_words       = cached["synonym_words"]
    seq_lens            = cached["seq_lens"]
    audio_paths_replaced = cached["audio_paths_replaced"]
    file_names_replaced  = cached["file_names_replaced"]
    skipped_positions   = cached["skipped_positions"]
else:
    print(f"Loading MLM model ({MLM_MODEL}) for synonym ranking...")
    mlm_pipe = pipeline(
        "fill-mask",
        model=MLM_MODEL,
        device=0 if torch.cuda.is_available() else -1,
    )
    print("  Model loaded.")

    pred_replaced = []
    gts = []
    replaced_words = []
    synonym_words = []
    seq_lens = []
    audio_paths_replaced = []
    file_names_replaced = []
    skipped_positions = []

    for i in range(len(all_pred)):
        pred = all_pred[i]
        gt = all_gt[i]
        tokens = pred.split()
        seq_lens.append(len(tokens))

        tagged = nltk.pos_tag(tokens)

        for j in range(len(tokens)):
            word = tokens[j]
            wn_pos = get_wordnet_pos(tagged[j][1])
            synonym = get_best_synonym_mlm(word, pred, pos=wn_pos, mlm_pipe=mlm_pipe)

            replaced_words.append(word)
            synonym_words.append(synonym if synonym else word)

            tokens_ = tokens[:]
            if synonym:
                tokens_[j] = synonym
                skipped_positions.append(False)
            else:
                skipped_positions.append(True)

            pred_replaced.append(" ".join(tokens_))
            gts.append(gt)
            audio_paths_replaced.append(audio_paths[i])
            file_names_replaced.append(all_file_names[i])

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(all_pred)} captions...")

    print(f"Total positions: {len(pred_replaced)} | No-synonym (skipped): {sum(skipped_positions)}")
    write_json({
        "pred_replaced":        pred_replaced,
        "gts":                  gts,
        "replaced_words":       replaced_words,
        "synonym_words":        synonym_words,
        "seq_lens":             seq_lens,
        "audio_paths_replaced": audio_paths_replaced,
        "file_names_replaced":  file_names_replaced,
        "skipped_positions":    skipped_positions,
    }, synonym_sentences_path)
    print(f"Saved MLM synonym-replaced sentences to {synonym_sentences_path}")

print(f"Total positions: {len(pred_replaced)} | No-synonym (skipped): {sum(skipped_positions)}")

# ── 3. Compare SBERT vs MLM synonym choices ───────────────────────────────
sbert_sentences_path = RESULTS_DIR / "synonym_sentences.json"
if sbert_sentences_path.exists():
    sbert_cached = json.load(open(sbert_sentences_path, "r"))
    sbert_syns = sbert_cached["synonym_words"]
    sbert_skipped = sbert_cached["skipped_positions"]

    n_total = len(synonym_words)
    n_same = sum(1 for a, b in zip(synonym_words, sbert_syns) if a == b)
    n_diff = n_total - n_same

    # Among non-skipped positions in both
    n_both_active = sum(1 for a, b in zip(skipped_positions, sbert_skipped) if not a and not b)
    n_same_active = sum(1 for s_mlm, s_sbert, sk_mlm, sk_sbert in
                        zip(synonym_words, sbert_syns, skipped_positions, sbert_skipped)
                        if not sk_mlm and not sk_sbert and s_mlm == s_sbert)

    print(f"\n=== SBERT vs MLM Synonym Selection Comparison ===")
    print(f"  Total positions: {n_total}")
    print(f"  Same synonym chosen: {n_same} ({100*n_same/n_total:.1f}%)")
    print(f"  Different synonym: {n_diff} ({100*n_diff/n_total:.1f}%)")
    print(f"  Both active (non-skipped): {n_both_active}")
    print(f"  Same among active: {n_same_active} ({100*n_same_active/n_both_active:.1f}%)")

    # Print some examples where they differ
    print(f"\n  Examples of differing selections:")
    count = 0
    for i, (s_mlm, s_sbert, sk_mlm, sk_sbert, word) in enumerate(
            zip(synonym_words, sbert_syns, skipped_positions, sbert_skipped, replaced_words)):
        if not sk_mlm and not sk_sbert and s_mlm != s_sbert and count < 15:
            print(f"    '{word}' -> SBERT: '{s_sbert}' | MLM: '{s_mlm}'")
            count += 1

# ── 4. Compute original scores (reuse cached) ─────────────────────────────
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

# ── 5. Compute MLM synonym scores ─────────────────────────────────────────
print(f"\nComputing scores for {len(pred_replaced)} MLM synonym-replaced sentences...")
mlm_scores_path = RESULTS_DIR / "synonym_mlm_scores.json"
if mlm_scores_path.exists():
    print("  Loading cached MLM synonym scores...")
    replaced_scores = json.load(open(mlm_scores_path))
else:
    replaced_scores = evaluate_aac_batch(
        pred_replaced, gts, batch_size=1045, device=device,
        audio_paths=audio_paths_replaced, ignore_all=True,
    )
    with open(mlm_scores_path, "w") as f:
        json.dump(replaced_scores, f)

# ── 6. Summary statistics ─────────────────────────────────────────────────
METRICS = [
    "bleu_1", "bleu_2", "bleu_3", "bleu_4",
    "rouge_l", "meteor", "cider_d", "spice", "spider",
    "fense", "clap_sim_text", "sbert_sim", "clap_sim_audio",
]

skip = np.array(skipped_positions, dtype=bool)

def expand_orig(scores, sl):
    exp = []
    for i, s in enumerate(sl):
        exp.extend([scores[i]] * s)
    return np.array(exp, dtype=float)

# Also load SBERT synonym scores for comparison
sbert_scores_path = RESULTS_DIR / "synonym_scores.json"
sbert_scores = None
sbert_skip = None
if sbert_scores_path.exists():
    sbert_scores = json.load(open(sbert_scores_path))
    sbert_skip = np.array(sbert_cached["skipped_positions"], dtype=bool)

print(f"\n{'Metric':>18s}  {'MLM_drop':>10s}  {'MLM_drop%':>10s}  {'SBERT_drop':>10s}  {'SBERT_drop%':>10s}")
print("-" * 65)

for m in METRICS:
    if m not in replaced_scores or m not in original_scores:
        continue

    o = expand_orig(original_scores[m]["scores"], seq_lens)
    r = np.array(replaced_scores[m]["scores"], dtype=float)

    valid = ~(np.isnan(o) | np.isnan(r) | skip)
    ov, rv = o[valid], r[valid]
    mlm_drop = (ov - rv).mean()
    mlm_drop_pct = 100 * mlm_drop / ov.mean() if ov.mean() != 0 else float("nan")

    # SBERT comparison
    if sbert_scores is not None and m in sbert_scores:
        sr = np.array(sbert_scores[m]["scores"], dtype=float)
        svalid = ~(np.isnan(o) | np.isnan(sr) | sbert_skip)
        sov, srv = o[svalid], sr[svalid]
        sbert_drop = (sov - srv).mean()
        sbert_drop_pct = 100 * sbert_drop / sov.mean() if sov.mean() != 0 else float("nan")
    else:
        sbert_drop = float("nan")
        sbert_drop_pct = float("nan")

    print(f"{m:>18s}  {mlm_drop:10.4f}  {mlm_drop_pct:9.1f}%  {sbert_drop:10.4f}  {sbert_drop_pct:9.1f}%")

print(f"\nMLM results saved to {RESULTS_DIR}/synonym_mlm_*.json")
