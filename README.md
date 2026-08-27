# AAC Evaluation Battery

A controlled perturbation battery for probing **reference bias in automated audio
captioning (AAC) evaluation metrics**. Given a set of human reference captions, it
applies meaning-preserving and meaning-destroying single-token edits and measures
how each metric responds, isolating three axes:

- **Acoustic relevance** — does the metric reward captions that match the audio?
- **Lexical-diversity tolerance** — is a valid *paraphrase* penalized?
- **Structural sensitivity** — is a *word-order scramble* penalized?

The battery scores **13 metrics** (BLEU-1..4, ROUGE-L, METEOR, CIDEr-D, SPICE,
SPIDEr, FENSE, CLAP-text, SBERT, CLAP-audio) plus two **reference-free** candidates
(CAF-Score = LAION-CLAP + Audio-Flamingo-3; and an Audio-Flamingo-3 LALM-as-judge).

## Layout

```
aac-metric-eval/
├── config.py              # central path configuration (edit or set env vars)
├── evaluate.py            # 13-metric scoring harness (wraps aac-metrics)
├── score_analysis.py      # per-metric mean/variance + Pearson/Spearman matrices
├── perturbations/         # the three perturbation axes
│   ├── masking.py                    # meaning-destroying: mask tokens with a non-word
│   ├── masking_removal.py            # token removal variant
│   ├── masking_random_word.py        # random real-word replacement variant
│   ├── synonym_replacement.py        # meaning-preserving: WordNet synonyms
│   ├── synonym_replacement_mlm.py    # meaning-preserving: RoBERTa-MLM synonyms (paper's synonym axis)
│   ├── synonym_replacement_mlm_filtered.py  # MLM synonyms + SBERT filter (ablation)
│   └── swap_adjacent.py              # structural: swap adjacent token pairs
├── audiocaps/             # AudioCaps replication of the battery
│   ├── build_audiocaps_refs.py       # build 5-caption AudioCaps eval CSV
│   ├── audiocaps_perturb.py          # generate AudioCaps perturbation sets
│   └── eval_audiocaps_base.py        # 13-metric scoring on AudioCaps sets
├── caf/                   # CAF-Score + LALM-as-judge pipeline
│   ├── caf_lookup.py                 # pair registry + CAF cache lookup (base env)
│   ├── attach_clotho.py              # register Clotho pairs for CAF scoring
│   ├── run_clap.py                   # CLAP term of CAF (base env)
│   ├── run_caf_af3.py                # FLEUR term + final CAF (caf_af3 env)
│   ├── merge_shards.py               # merge multi-GPU CAF cache shards
│   ├── finalize_sidecars.py          # fill CAF values into sidecars (pass 2)
│   └── caf_analysis.py               # paired-drop tables (the paper's CAF/FLEUR results)
└── run_caf_pipeline.sh    # one-command CAF/FLEUR run (CLAP -> N GPU workers -> merge)
```

## Install

Clone with submodules (the reference-free metrics depend on the external
[CAF-Score](https://github.com/inseong00/CAF-Score) repo, vendored as a submodule
at `third_party/CAF-Score`):

```bash
git clone --recurse-submodules <this-repo-url>
# or, if already cloned:
git submodule update --init --recursive
```

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger_eng')"
```

`SPICE` (via `aac-metrics`) needs a Java runtime; `FENSE`/`CLAP` download model
weights on first use.

## Configure paths

All locations live in `config.py` and can be overridden with environment
variables — no path is hardcoded in the scripts.

| Variable | Meaning | Expected layout |
|---|---|---|
| `CLOTHO_ROOT` | Clotho V2 root | `clotho_csv_files/clotho_captions_<split>.csv`, `<split>/<file>.wav` |
| `AUDIOCAPS_ROOT` | AudioCaps root | `eval_text.csv`, `16000/eval/<id>.wav` |
| `RESULTS_DIR` | where `*_scores.json` / CAF caches are written | (created automatically) |
| `FIGURES_DIR` | where figures are saved | (created automatically) |
| `CAF_SRC` | checkout of the external CAF-Score repo | see below |

```bash
export CLOTHO_ROOT=/data/ClothoV2
export AUDIOCAPS_ROOT=/data/AudioCaps
export RESULTS_DIR=$PWD/results
```

## Run the perturbation battery (Clotho)

Each script loads Clotho leave-one-out reference sets, applies its perturbation,
scores all 13 metrics, and writes `*_scores.json` under `RESULTS_DIR`:

```bash
python perturbations/masking.py                          # meaning-destroying
python perturbations/synonym_replacement_mlm.py          # meaning-preserving
python perturbations/swap_adjacent.py                    # word-order
```

Then summarize any scores file (per-metric mean/CV + correlation matrices):

```bash
python score_analysis.py $RESULTS_DIR/synonym_mlm_scores.json
```

## AudioCaps replication

```bash
python audiocaps/build_audiocaps_refs.py   # -> csv_files/audiocaps_captions_evaluation.csv
python audiocaps/audiocaps_perturb.py      # -> RESULTS_DIR/audiocaps/*_candidates.json
python audiocaps/eval_audiocaps_base.py    # -> RESULTS_DIR/audiocaps/*_scores.json
```

`build_audiocaps_refs.py` also expects the official AudioCaps multi-caption test
split at `third_party/audiocaps_meta/test.csv`
(from https://github.com/cdjkim/audiocaps).

Note: `eval_audiocaps_base.py` skips any set whose `<name>_scores.json` already
exists — delete the stale file first to force recomputation.

## CAF-Score + LALM-as-judge

These use the bundled **CAF-Score** submodule (`third_party/CAF-Score`, LAION-CLAP +
Audio-Flamingo-3) — ensure it is checked out (`git submodule update --init`) or
point `CAF_SRC` at your own checkout. Install the AF3 dependencies from that repo's
environment files (a separate `caf_af3` conda env is expected).

Two-pass workflow (pairs are registered first, then scored, then filled back).
The paper evaluates the **full** perturbation sets (~266k unique pairs across both
datasets); registration defaults to full — set `CAF_SAMPLE=N` to subsample.

```bash
# pass 1 — register (audio, caption) pairs into RESULTS_DIR/caf_pairs.jsonl
python caf/attach_clotho.py
python audiocaps/audiocaps_perturb.py   # if also running AudioCaps

# score + pass 2, one command: CLAP term, then NUM_SHARDS Audio-Flamingo-3
# workers (one per ~24GB GPU), then shard merge + sidecar refill
NUM_SHARDS=4 nohup bash run_caf_pipeline.sh > caf_pipeline.log 2>&1 &

# paired-drop tables (the paper's CAF/FLEUR results)
python caf/caf_analysis.py   # -> RESULTS_DIR/caf_summary.csv, caf_paired_drops.csv
```

CAF = `alpha * CLAP + (1 - alpha) * FLEUR` (alpha default 0.8). The pipeline is
resumable — already-scored pairs are skipped on restart. As a reference point,
the full run (~257k FLEUR pairs) took ~15 h on 4× RTX 3090; the CLAP pass ~10 min.

## Citation

If you use this battery, please cite the paper (see the repository root).
Metric implementations are provided by
[`aac-metrics`](https://github.com/Labbeti/aac-metrics).
