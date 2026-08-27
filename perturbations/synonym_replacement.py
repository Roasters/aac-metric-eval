import numpy as np
import torch
import csv
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from evaluate import evaluate_aac_batch
from config import (RESULTS_DIR, FIGURES_DIR, clotho_captions_csv,
                    clotho_audio_path)
from typing import List, Dict, Any, Union
import json
import nltk
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

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


def get_best_synonym(word: str, sentence: str, pos=None, sbert_model=None) -> str:
    """Return the synonym that produces the sentence most similar to the original,
    as measured by SBERT cosine similarity.
    Returns None if no synonym is found."""
    candidates = get_all_synonyms(word, pos=pos)
    if not candidates:
        return None
    if sbert_model is None or len(candidates) == 1:
        return candidates[0]

    tokens = sentence.split()
    idx = tokens.index(word) if word in tokens else None
    if idx is None:
        return candidates[0]

    # Build candidate sentences
    candidate_sentences = []
    for syn in candidates:
        tokens_ = tokens[:]
        tokens_[idx] = syn
        candidate_sentences.append(" ".join(tokens_))

    # Embed original + all candidates in one batch
    all_sentences = [sentence] + candidate_sentences
    embeddings = sbert_model.encode(all_sentences, convert_to_numpy=True)
    orig_emb = embeddings[0:1]
    cand_embs = embeddings[1:]

    sims = cosine_similarity(orig_emb, cand_embs)[0]
    best_idx = int(np.argmax(sims))
    return candidates[best_idx]


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

all_pred = []
all_gt = []
all_file_names = []
audio_paths = []

for split in ["evaluation"]:
    with open(clotho_captions_csv(split)) as f:
        reader = csv.reader(f, delimiter=',')
        print(next(reader))  # Skip header
        for r in reader:
            file_name = r[1]
            for i in range(2, len(r)):
                all_pred.append(r[i])
                all_gt.append(r[2:i] + r[i+1:])
                all_file_names.append(file_name)
                audio_paths.append(clotho_audio_path(file_name, split))

# --- 1. Process Data & Build Synonym-Replaced Sentences ---
synonym_sentences_path = (RESULTS_DIR / "synonym_sentences.json")

if synonym_sentences_path.exists():
    print("Loading cached synonym-replaced sentences...")
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
    print("Loading SBERT model for synonym selection...")
    sbert_model = SentenceTransformer(SBERT_MODEL)

    pred_replaced = []
    gts = []
    replaced_words = []      # original word
    synonym_words = []       # the synonym used
    seq_lens = []
    audio_paths_replaced = []
    file_names_replaced = []
    skipped_positions = []   # True if no synonym found for this position

    for i in range(len(all_pred)):
        pred = all_pred[i]
        gt = all_gt[i]
        tokens = pred.split()
        seq_lens.append(len(tokens))

        tagged = nltk.pos_tag(tokens)

        for j in range(len(tokens)):
            word = tokens[j]
            wn_pos = get_wordnet_pos(tagged[j][1])
            synonym = get_best_synonym(word, pred, pos=wn_pos, sbert_model=sbert_model)

            replaced_words.append(word)
            synonym_words.append(synonym if synonym else word)  # keep original if no synonym

            tokens_ = tokens[:]
            if synonym:
                tokens_[j] = synonym
                skipped_positions.append(False)
            else:
                # No synonym available — sentence unchanged; mark as skipped
                skipped_positions.append(True)

            pred_replaced.append(" ".join(tokens_))
            gts.append(gt)
            audio_paths_replaced.append(audio_paths[i])
            file_names_replaced.append(all_file_names[i])

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
    print(f"Saved synonym-replaced sentences to {synonym_sentences_path}")

print(f"Total positions: {len(pred_replaced)} | No-synonym (skipped): {sum(skipped_positions)}")

# --- 2. Compute Scores ---
print("Calculating scores for original sentences...")
original_scores_path = (RESULTS_DIR / "original_scores.json")
if not original_scores_path.exists():
    original_scores = evaluate_aac_batch(all_pred, all_gt, batch_size=1045, device=device, audio_paths=audio_paths, ignore_all=True)
    write_json(original_scores, original_scores_path)
else:
    print("Loading cached original scores...")
    original_scores = json.load(open(original_scores_path, "r"))

print("Calculating scores for synonym-replaced sentences (this may take time)...")
replaced_scores_path = (RESULTS_DIR / "synonym_scores.json")
if not replaced_scores_path.exists():
    replaced_scores = evaluate_aac_batch(pred_replaced, gts, batch_size=1045, device=device, audio_paths=audio_paths_replaced, ignore_all=True)
    write_json(replaced_scores, replaced_scores_path)
else:
    print("Loading cached synonym-replaced scores...")
    replaced_scores = json.load(open(replaced_scores_path, "r"))

# --- 3. Build DataFrame ---
original_sentence_ids = []
for idx, length in enumerate(seq_lens):
    original_sentence_ids.extend([idx] * length)

# Detect SPICE-failed samples (NaN) and no-synonym positions
_ref_metric = next(m for m in replaced_scores if m not in ('idxs', 'vocab'))
nan_mask = np.isnan(np.array(replaced_scores[_ref_metric]['scores'], dtype=float))

# Combined mask: exclude SPICE failures AND positions where no synonym was available
skip_mask = nan_mask | np.array(skipped_positions, dtype=bool)

if skip_mask.any():
    valid = ~skip_mask
    n_spice = int(nan_mask.sum())
    n_no_syn = int(np.array(skipped_positions).sum())
    print(f"  Excluded {n_spice} SPICE-failed and {n_no_syn} no-synonym positions.")
    original_sentence_ids = [s for s, v in zip(original_sentence_ids, valid) if v]
    replaced_words  = [w for w, v in zip(replaced_words, valid) if v]
    synonym_words   = [w for w, v in zip(synonym_words, valid) if v]
else:
    valid = ~skip_mask  # all True

pos_tags_replaced = [tag for word, tag in nltk.pos_tag(replaced_words)]

pos_mapping = {
    'NN': 'Noun', 'NNS': 'Noun', 'NNP': 'Noun', 'NNPS': 'Noun',
    'VB': 'Verb', 'VBD': 'Verb', 'VBG': 'Verb', 'VBN': 'Verb', 'VBP': 'Verb', 'VBZ': 'Verb',
    'JJ': 'Adjective', 'JJR': 'Adjective', 'JJS': 'Adjective',
    'RB': 'Adverb', 'RBR': 'Adverb', 'RBS': 'Adverb',
    'PRP': 'Pronoun', 'PRP$': 'Pronoun',
    'DT': 'Determiner', 'PDT': 'Determiner', 'WDT': 'Determiner',
    'IN': 'Preposition/Conjunction', 'CC': 'Preposition/Conjunction',
}
pos_tags_mapped = [pos_mapping.get(t, 'Other') for t in pos_tags_replaced]

df = pd.DataFrame({
    'sentence_id':  original_sentence_ids,
    'original_word': replaced_words,
    'synonym_word':  synonym_words,
    'pos_tag':       pos_tags_mapped,
})

# --- 4. Process Each Metric ---
metric_names = list(original_scores.keys())
metric_names.remove('idxs') if 'idxs' in metric_names else None
metric_names.remove('vocab') if 'vocab' in metric_names else None
metric_names.remove('spider_fl') if 'spider_fl' in metric_names else None
metric_names.remove("sbert_sim") if "sbert_sim" in metric_names else None

for metric in metric_names:
    orig_scores_list = original_scores[metric]["scores"]

    expanded_orig_scores = []
    for idx, length in enumerate(seq_lens):
        expanded_orig_scores.extend([orig_scores_list[idx]] * length)

    if skip_mask.any():
        expanded_orig_scores = [s for s, v in zip(expanded_orig_scores, valid) if v]
        replaced_scores_list = [s for s, v in zip(replaced_scores[metric]["scores"], valid) if v]
    else:
        replaced_scores_list = replaced_scores[metric]["scores"]

    df[f'orig_{metric}']   = expanded_orig_scores
    df[f'syn_{metric}']    = replaced_scores_list
    df[f'impact_{metric}'] = (
        df[f'syn_{metric}'] / df[f'orig_{metric}'].replace(0, float('nan'))
    ).clip(0, 2)  # ratio: 1.0 = no change, <1.0 = drop, >1.0 = gain

# --- 5. Analysis & Statistics ---
impact_cols = [f'impact_{m}' for m in metric_names]
word_stats = df.groupby('original_word')[impact_cols].agg(['mean', 'count'])
word_stats.columns = ['_'.join(col).strip() for col in word_stats.columns.values]
word_stats = word_stats[word_stats[f'impact_{metric_names[0]}_count'] >= 1]

print(f"--- Analysis of {len(metric_names)} Metrics: {metric_names} ---")
for metric in metric_names:
    sorted_by_metric = word_stats.sort_values(by=f'impact_{metric}_mean', ascending=False)
    print(f"\nTop 5 words most impacted by synonym replacement for {metric}:")
    print(sorted_by_metric[[f'impact_{metric}_mean']].head(5))

# --- 6. Correlation of Sensitivity Between Metrics ---
plt.figure(figsize=(8, 6))
impact_df = df[impact_cols].copy()
impact_df.columns = [c.replace('impact_', '') for c in impact_df.columns]
sns.heatmap(impact_df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation of Synonym Sensitivity between Metrics')
plt.tight_layout()
plt.savefig(str(FIGURES_DIR / "synonym_metric_correlation_heatmap.png"), dpi=150)
plt.show()

# --- 7. Relative Performance Drop per Metric ---
relative_remaining = {}
for metric in metric_names:
    orig = df[f'orig_{metric}']
    syn  = df[f'syn_{metric}']
    ratio = (syn / orig.replace(0, np.nan)).clip(0, 2)
    relative_remaining[metric] = ratio.mean()

metrics_list = list(relative_remaining.keys())
values = list(relative_remaining.values())
deltas = [v - 1.0 for v in values]
colors = ['steelblue' if d >= 0 else 'salmon' for d in deltas]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(metrics_list, deltas, bottom=1.0, color=colors)
ax.axhline(1.0, color='black', linewidth=0.8, linestyle='--')
ax.set_title('Relative Score After Synonym Replacement (baseline = 1.0)')
ax.set_ylabel('Relative Score (syn / orig)')
ax.set_xticklabels(metrics_list, rotation=45, ha='right')
plt.tight_layout()
plt.savefig(str(FIGURES_DIR / "synonym_relative_drop.png"), dpi=150)
plt.show()

# --- 8. POS Tag Analysis ---
pos_stats = df.groupby('pos_tag')[impact_cols].agg(['mean', 'count'])
pos_stats.columns = ['_'.join(col).strip() for col in pos_stats.columns.values]
pos_stats = pos_stats[pos_stats[f'{impact_cols[0]}_count'] >= 1]

print("\n--- Average Relative Score by Part of Speech ---")
print(pos_stats[[c for c in pos_stats.columns if 'mean' in c]])

# Plot 1: POS on x-axis, metric as hue
plot_data_pos = df.melt(
    id_vars=['pos_tag'],
    value_vars=impact_cols,
    var_name='Metric',
    value_name='Relative Score'
)
plot_data_pos['Metric'] = plot_data_pos['Metric'].str.replace('impact_', '')
plot_data_pos['Relative Score'] = plot_data_pos['Relative Score'] - 1.0

plt.figure(figsize=(14, 6))
sns.barplot(
    data=plot_data_pos,
    x='pos_tag',
    y='Relative Score',
    hue='Metric',
    errorbar=None
)
plt.title('Synonym Replacement Sensitivity by Part of Speech')
plt.xlabel('Part of Speech Tag')
plt.ylabel('Relative Score (0 = no change, <0 = drop, >0 = gain)')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.legend(title='Evaluation Metric')
plt.tight_layout()
plt.savefig(str(FIGURES_DIR / "synonym_pos_impact_per_tag.png"), dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Metric on x-axis, POS as hue
plot_data_pos2 = df.melt(
    id_vars=['pos_tag'],
    value_vars=impact_cols,
    var_name='Metric',
    value_name='Relative Score'
)
plot_data_pos2['Metric'] = plot_data_pos2['Metric'].str.replace('impact_', '')
plot_data_pos2['Relative Score'] = plot_data_pos2['Relative Score'] - 1.0

plt.figure(figsize=(12, 6))
sns.barplot(
    data=plot_data_pos2,
    x='Metric',
    y='Relative Score',
    hue='pos_tag',
    errorbar=None
)
plt.title('Synonym Replacement Sensitivity by Metric and Part of Speech')
plt.xlabel('Evaluation Metric')
plt.ylabel('Relative Score (0 = no change, <0 = drop, >0 = gain)')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.legend(title='Grammatical Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.xticks(rotation=45)
plt.savefig(str(FIGURES_DIR / "synonym_pos_impact_per_metric.png"), dpi=300, bbox_inches='tight')
plt.show()

# --- 9. Compare Metric Sensitivity for Specific Words ---
sorted_by_metric = word_stats.sort_values(by=f'impact_{metric_names[0]}_mean', ascending=False)
top_words = sorted_by_metric.head(10).index
plot_data = df[df['original_word'].isin(top_words)].melt(
    id_vars=['original_word'],
    value_vars=impact_cols,
    var_name='Metric',
    value_name='Score Drop'
)
plot_data['Metric'] = plot_data['Metric'].str.replace('impact_', '')

plt.figure(figsize=(12, 6))
sns.barplot(data=plot_data, x='original_word', y='Score Drop', hue='Metric')
plt.title('How Different Metrics React to Synonym Replacement of the Same Words')
plt.xticks(rotation=45)
plt.ylabel('Performance Drop (Sensitivity)')
plt.tight_layout()
plt.savefig(str(FIGURES_DIR / "synonym_sensitivity_comparison.png"), dpi=150)
plt.show()
