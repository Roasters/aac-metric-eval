"""
Synonym replacement using MLM (RoBERTa) for synonym ranking,
with two additional filters over the original synonym_replacement_mlm.py:

  1. Single-token only — reject any synonym containing spaces.
     Eliminates "a" → "type a", "is" → "make up", "an" → "associate in nursing", etc.

  2. Content words only — only replace nouns, verbs, adjectives, adverbs.
     Skip determiners (DT), prepositions (IN), conjunctions (CC), pronouns (PRP/PRP$),
     particles (RP), modals (MD), existential-there (EX), predeterminers (PDT),
     wh-determiners (WDT), possessive endings (POS), list markers (LS), TO, and
     other function-word POS tags.

These filters are applied without any reference to evaluation metric scores.
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


# POS tags to skip (function words)
SKIP_POS = {
    'DT',    # determiner (a, an, the, this, that)
    'IN',    # preposition or subordinating conjunction (in, of, with, at, by, for, from, on)
    'CC',    # coordinating conjunction (and, or, but)
    'PRP',   # personal pronoun (it, he, she, they)
    'PRP$',  # possessive pronoun (its, his, her, their)
    'WDT',   # wh-determiner (which, that)
    'WP',    # wh-pronoun (who, what)
    'WP$',   # possessive wh-pronoun (whose)
    'WRB',   # wh-adverb (where, when, how) — these are function adverbs
    'MD',    # modal (can, could, will, would, should)
    'TO',    # to
    'EX',    # existential there
    'PDT',   # predeterminer (all, both)
    'POS',   # possessive ending ('s)
    'LS',    # list item marker
    'RP',    # particle (up, off, out)
    'UH',    # interjection
    'FW',    # foreign word
    'SYM',   # symbol
    '$',     # dollar sign
    ',', '.', ':', '(', ')', '``', "''", '#',
}


def get_all_synonyms_filtered(word: str, pos=None) -> list[str]:
    """Return all unique single-token WordNet synonyms of `word` (excluding the word itself)."""
    synsets = wordnet.synsets(word, pos=pos) if pos else wordnet.synsets(word)
    candidates = set()
    for synset in synsets:
        for lemma in synset.lemmas():
            candidate = lemma.name().replace('_', ' ').lower()
            if candidate == word.lower():
                continue
            # Filter 1: single-token only
            if ' ' in candidate:
                continue
            candidates.add(candidate)
    return list(candidates)


def get_best_synonym_mlm_filtered(word: str, sentence: str, pos=None, mlm_pipe=None) -> str:
    """Return the single-token WordNet synonym with the highest MLM probability in context.
    Returns None if no synonym is found."""
    candidates = get_all_synonyms_filtered(word, pos=pos)
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

    # Get MLM predictions
    try:
        predictions = mlm_pipe(masked_sentence, top_k=250)
    except Exception:
        return candidates[0]

    # Build a map of token_str -> score from MLM output
    token_scores = {}
    for pred in predictions:
        token_str = pred["token_str"].strip().lower()
        token_scores[token_str] = pred["score"]

    # Score each candidate by MLM probability (all are single-token now)
    best_syn = None
    best_score = -1.0
    for syn in candidates:
        score = token_scores.get(syn.lower(), 0.0)
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

# ── 2. Build filtered synonym-replaced sentences ─────────────────────────────
synonym_sentences_path = RESULTS_DIR / "synonym_mlm_filtered_sentences.json"

if synonym_sentences_path.exists():
    print("Loading cached filtered MLM synonym-replaced sentences...")
    cached = json.load(open(synonym_sentences_path, "r"))
    pred_replaced        = cached["pred_replaced"]
    gts                  = cached["gts"]
    replaced_words       = cached["replaced_words"]
    synonym_words        = cached["synonym_words"]
    seq_lens             = cached["seq_lens"]
    audio_paths_replaced = cached["audio_paths_replaced"]
    file_names_replaced  = cached["file_names_replaced"]
    skipped_positions    = cached["skipped_positions"]
    skipped_reasons      = cached["skipped_reasons"]
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
    skipped_reasons = []  # track why each position was skipped

    n_skip_pos = 0
    n_skip_no_syn = 0
    n_replaced = 0

    for i in range(len(all_pred)):
        pred = all_pred[i]
        gt = all_gt[i]
        tokens = pred.split()
        seq_lens.append(len(tokens))

        tagged = nltk.pos_tag(tokens)

        for j in range(len(tokens)):
            word = tokens[j]
            pos_tag = tagged[j][1]

            replaced_words.append(word)

            # Filter 2: skip function words
            if pos_tag in SKIP_POS:
                skipped_positions.append(True)
                skipped_reasons.append("function_word")
                synonym_words.append(word)
                pred_replaced.append(" ".join(tokens))
                gts.append(gt)
                audio_paths_replaced.append(audio_paths[i])
                file_names_replaced.append(all_file_names[i])
                n_skip_pos += 1
                continue

            wn_pos = get_wordnet_pos(pos_tag)
            synonym = get_best_synonym_mlm_filtered(word, pred, pos=wn_pos, mlm_pipe=mlm_pipe)

            if synonym:
                tokens_ = tokens[:]
                tokens_[j] = synonym
                skipped_positions.append(False)
                skipped_reasons.append(None)
                synonym_words.append(synonym)
                pred_replaced.append(" ".join(tokens_))
                n_replaced += 1
            else:
                skipped_positions.append(True)
                skipped_reasons.append("no_synonym")
                synonym_words.append(word)
                pred_replaced.append(" ".join(tokens))
                n_skip_no_syn += 1

            gts.append(gt)
            audio_paths_replaced.append(audio_paths[i])
            file_names_replaced.append(all_file_names[i])

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(all_pred)} captions...")

    print(f"\nTotal positions: {len(pred_replaced)}")
    print(f"  Replaced:            {n_replaced}")
    print(f"  Skipped (func word): {n_skip_pos}")
    print(f"  Skipped (no synonym):{n_skip_no_syn}")

    write_json({
        "pred_replaced":        pred_replaced,
        "gts":                  gts,
        "replaced_words":       replaced_words,
        "synonym_words":        synonym_words,
        "seq_lens":             seq_lens,
        "audio_paths_replaced": audio_paths_replaced,
        "file_names_replaced":  file_names_replaced,
        "skipped_positions":    skipped_positions,
        "skipped_reasons":      skipped_reasons,
    }, synonym_sentences_path)
    print(f"Saved filtered synonym-replaced sentences to {synonym_sentences_path}")

print(f"Total positions: {len(pred_replaced)} | Skipped: {sum(skipped_positions)}")

# ── 3. Compare with original (unfiltered) synonym choices ────────────────────
orig_sentences_path = RESULTS_DIR / "synonym_mlm_sentences.json"
if orig_sentences_path.exists():
    orig_cached = json.load(open(orig_sentences_path, "r"))
    orig_syns = orig_cached["synonym_words"]
    orig_skipped = orig_cached["skipped_positions"]

    n_total = len(synonym_words)
    n_orig_active = sum(1 for s in orig_skipped if not s)
    n_filt_active = sum(1 for s in skipped_positions if not s)
    n_both_active = sum(1 for a, b in zip(skipped_positions, orig_skipped) if not a and not b)
    n_same = sum(1 for s_f, s_o, sk_f, sk_o in
                 zip(synonym_words, orig_syns, skipped_positions, orig_skipped)
                 if not sk_f and not sk_o and s_f == s_o)
    n_diff = n_both_active - n_same

    print(f"\n=== Original vs Filtered Synonym Comparison ===")
    print(f"  Total positions: {n_total}")
    print(f"  Original active: {n_orig_active} ({100*n_orig_active/n_total:.1f}%)")
    print(f"  Filtered active: {n_filt_active} ({100*n_filt_active/n_total:.1f}%)")
    print(f"  Both active:     {n_both_active}")
    print(f"  Same synonym:    {n_same} ({100*n_same/n_both_active:.1f}% of both-active)")
    print(f"  Different:       {n_diff} ({100*n_diff/n_both_active:.1f}% of both-active)")

    # Show examples of changed synonyms (different due to multi-word filtering)
    print(f"\n  Examples where filtered synonym differs (multi-word removed):")
    count = 0
    for i, (s_f, s_o, sk_f, sk_o, word) in enumerate(
            zip(synonym_words, orig_syns, skipped_positions, orig_skipped, replaced_words)):
        if not sk_f and not sk_o and s_f != s_o and count < 20:
            print(f"    '{word}' -> original: '{s_o}' | filtered: '{s_f}'")
            count += 1

# ── 4. Compute original scores (reuse cached) ───────────────────────────────
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

# ── 5. Compute filtered synonym scores ──────────────────────────────────────
print(f"\nComputing scores for {len(pred_replaced)} filtered synonym-replaced sentences...")
mlm_scores_path = RESULTS_DIR / "synonym_mlm_filtered_scores.json"
if mlm_scores_path.exists():
    print("  Loading cached filtered synonym scores...")
    replaced_scores = json.load(open(mlm_scores_path))
else:
    replaced_scores = evaluate_aac_batch(
        pred_replaced, gts, batch_size=1045, device=device,
        audio_paths=audio_paths_replaced, ignore_all=True,
    )
    with open(mlm_scores_path, "w") as f:
        json.dump(replaced_scores, f)

# ── 6. Summary statistics ───────────────────────────────────────────────────
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

# Also load original (unfiltered) synonym scores for comparison
orig_scores_path = RESULTS_DIR / "synonym_mlm_scores.json"
orig_syn_scores = None
orig_skip = None
if orig_scores_path.exists() and orig_sentences_path.exists():
    orig_syn_scores = json.load(open(orig_scores_path))
    orig_skip = np.array(orig_cached["skipped_positions"], dtype=bool)

print(f"\n{'Metric':>18s}  {'Filt_drop':>10s}  {'Filt_drop%':>10s}  {'Orig_drop':>10s}  {'Orig_drop%':>10s}")
print("-" * 65)

for m in METRICS:
    if m not in replaced_scores or m not in original_scores:
        continue

    o = expand_orig(original_scores[m]["scores"], seq_lens)
    r = np.array(replaced_scores[m]["scores"], dtype=float)

    valid = ~(np.isnan(o) | np.isnan(r) | skip)
    ov, rv = o[valid], r[valid]
    filt_drop = (ov - rv).mean()
    filt_drop_pct = 100 * filt_drop / ov.mean() if ov.mean() != 0 else float("nan")

    # Original (unfiltered) comparison
    if orig_syn_scores is not None and m in orig_syn_scores:
        sr = np.array(orig_syn_scores[m]["scores"], dtype=float)
        svalid = ~(np.isnan(o) | np.isnan(sr) | orig_skip)
        sov, srv = o[svalid], sr[svalid]
        orig_drop = (sov - srv).mean()
        orig_drop_pct = 100 * orig_drop / sov.mean() if sov.mean() != 0 else float("nan")
    else:
        orig_drop = float("nan")
        orig_drop_pct = float("nan")

    print(f"{m:>18s}  {filt_drop:10.4f}  {filt_drop_pct:9.1f}%  {orig_drop:10.4f}  {orig_drop_pct:9.1f}%")

print(f"\nFiltered results saved to {RESULTS_DIR}/synonym_mlm_filtered_*.json")
