# AAC Metric Evaluation Battery

This repository is the camera-ready code release for a controlled perturbation
battery that diagnoses what automated audio-caption metrics actually measure. It
keeps the audio, candidate caption, reference set, and edit location aligned while
changing one property at a time.

The battery isolates three diagnostic dimensions:

1. **Acoustic grounding:** does a score reflect correspondence between the caption
   and its audio, rather than reference wording alone?
2. **Tolerance to valid lexical variation:** does replacing a word with a
   contextually selected synonym avoid an unjustified penalty?
3. **Local word-order sensitivity:** does swapping an adjacent token pair expose
   sensitivity to linguistic structure rather than only the bag of words?

The standard suite contains BLEU-1 through BLEU-4, ROUGE-L, METEOR, CIDEr-D,
SPICE, SPIDEr, FENSE, CLAP text similarity, SBERT similarity, and CLAP audio
similarity. The release also supports the reference-free CAF-Score composite and
its LAION-CLAP and Audio-Flamingo-3/FLEUR components.

## Repository layout

```text
README.md
LICENSE
CITATION.cff
pyproject.toml
requirements.txt
configs/
  camera_ready.yaml
scripts/
  prepare_datasets.py
  build_perturbations.py
  run_evaluation.py
  summarize_results.py
eval_battery/
  datasets/
  perturbations/
  metrics/
  analysis.py
  config.py
  io.py
  records.py
data/
  clotho/
  audiocaps/
results/
  perturbation_pairs/
  cached_scores/
  tables/
third_party/
  CAF-Score/
```

Reusable code lives under `eval_battery/`; `scripts/` contains orchestration only;
`configs/` records experiment choices; `data/` contains instructions and generated
manifests; and `results/` contains generated pairs, score caches, and tables.

## Installation

The release is tested with Python 3.11, PyTorch 2.x, NumPy 1.26, pandas 2.2,
SciPy 1.15, PyYAML 6.0, and `aac-metrics==0.6.0`. Java is required by SPICE.
FENSE, CLAP, RoBERTa, and sentence-transformer weights are downloaded on first use.

```bash
git clone --recurse-submodules https://github.com/Roasters/aac-metric-eval.git
cd aac-metric-eval
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m nltk.downloader wordnet averaged_perceptron_tagger_eng
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

CAF/FLEUR additionally requires the Audio-Flamingo-3 environment supplied by the
CAF-Score submodule. Its large-model dependencies are intentionally not mixed into
the base environment.

## Configuration

All camera-ready choices live in [`configs/camera_ready.yaml`](configs/camera_ready.yaml).
Relative paths resolve from the repository root. The following environment variables
override local defaults:

| Variable | Purpose |
|---|---|
| `CLOTHO_ROOT` | Clotho V2 root |
| `AUDIOCAPS_ROOT` | AudioCaps root |
| `RESULTS_DIR` | Generated artifact root |
| `CAF_SRC` | CAF-Score checkout |

Copy the YAML file before changing perturbations, models, or random seeds. Each
generated perturbation manifest records row counts and SHA-256 checksums.

## Dataset acquisition and preparation

Audio files are not redistributed. Obtain each dataset from its official source and
accept its applicable terms.

### Clotho V2

Expected layout:

```text
$CLOTHO_ROOT/
  clotho_csv_files/clotho_captions_evaluation.csv
  evaluation/*.wav
```

Validate it with:

```bash
export CLOTHO_ROOT=/data/ClothoV2
python scripts/prepare_datasets.py --dataset clotho
```

### AudioCaps

The local AudioCaps copy is expected to contain `eval_text.csv` and
`16000/eval/*.wav`. Place the official multi-caption test CSV from the AudioCaps
repository at `data/audiocaps/official_test.csv`. The preparation step matches the
on-disk caption to the official YouTube identifier, retains one available waveform
per source, and builds the five-caption evaluation CSV.

```bash
export AUDIOCAPS_ROOT=/data/AudioCaps
python scripts/prepare_datasets.py --dataset audiocaps --prepare
```

The commands write `data/<dataset>/manifest.json`. They report unavailable audio but
do not copy or redistribute it.

## Build perturbation pairs

Build each core axis independently so interrupted or resource-heavy stages can be
resumed. `original.jsonl` is generated alongside the requested axis.

```bash
# Meaning-destroying fixed-nonword replacement
python scripts/build_perturbations.py --dataset clotho --axes masking

# Meaning-preserving WordNet synonym ranked by RoBERTa MLM
python scripts/build_perturbations.py --dataset clotho --axes synonym

# Local word-order disruption
python scripts/build_perturbations.py --dataset clotho --axes swap_adjacent
```

Replace `clotho` with `audiocaps` for the replication. To build all camera-ready
axes in one command:

```bash
python scripts/build_perturbations.py --dataset clotho
python scripts/build_perturbations.py --dataset audiocaps
```

Use `--max-records 10 --force` for a small generation smoke test. The optional
`removal` and `random_nonword` controls are available through `--axes`.

### Synonym reporting rule

The JSONL synonym file retains one row for every attempted token position so score
alignment remains explicit. If no WordNet replacement is available, the row has
`perturbation.realized=false` and the candidate is unchanged.

**All reported synonym results are calculated only over positions where a
replacement was realized. Identity rows are never included in the synonym drop.**

Camera-ready full-set counts are:

| Dataset | Attempted positions | Realized replacements | Rate |
|---|---:|---:|---:|
| Clotho | 59,142 | 42,360 | 71.6% |
| AudioCaps | 42,989 | 36,095 | 84.0% |

`manifest.json` and `verification.json` compare generated counts with these values.

## Run the 13 established metrics

Each setting is scored from its released JSONL pairs; the metric layer never rebuilds
or guesses perturbations.

```bash
python scripts/run_evaluation.py established --dataset clotho
python scripts/run_evaluation.py established --dataset audiocaps
```

To run or replace only selected artifacts:

```bash
python scripts/run_evaluation.py established \
  --dataset clotho --sets original masking synonym swap_adjacent --force
```

SPICE failures are isolated per record by default, and missing values remain in place
so every scorer array stays aligned with `record_ids`.

## Run CAF-Score and FLEUR (Audio-Flamingo-3)

CAF uses a two-pass cache. Registration is light-weight and runs in the base
environment; CLAP and FLEUR fill each unique audio-caption key once; finalization
then refills all aligned sidecars.

```bash
# 1. Register every generated pair and create pending sidecars.
python scripts/run_evaluation.py register-caf

# 2. Compute LAION-CLAP in the base environment.
CUDA_VISIBLE_DEVICES=0 python scripts/run_evaluation.py clap

# 3. Compute four independent AF3/FLEUR shards in the CAF environment.
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i conda run -n caf_af3 \
    python scripts/run_evaluation.py fleur --shard $i --num-shards 4 &
done
wait

# 4. Merge worker caches and resolve sidecars.
python scripts/run_evaluation.py merge-caf --shards \
  results/cached_scores/caf_cache_w0.json \
  results/cached_scores/caf_cache_w1.json \
  results/cached_scores/caf_cache_w2.json \
  results/cached_scores/caf_cache_w3.json
python scripts/run_evaluation.py finalize-caf
```

The composite is `CAF = 0.8 × CLAP + 0.2 × FLEUR`. Stages are resumable: existing
component scores are skipped. `--limit N` is available on the CLAP and FLEUR stages
for environment smoke tests.

## Reconstruct paper and appendix tables

```bash
python scripts/summarize_results.py --datasets clotho audiocaps
```

This creates, per dataset:

| Output | Contents |
|---|---|
| `results/tables/<dataset>/metric_stats.csv` | mean, standard deviation, variance, CV, range, and valid count by setting |
| `results/tables/<dataset>/paired_drops.csv` | paired absolute/percentage drops and gain rates; synonyms are replacement-only |
| `results/tables/<dataset>/original_pearson_correlation.csv` | baseline Pearson matrix |
| `results/tables/<dataset>/original_spearman_correlation.csv` | baseline Spearman matrix |
| `results/tables/<dataset>/verification.json` | row-count verification against camera-ready constants |

The corresponding inputs are:

- `results/perturbation_pairs/<dataset>/*.jsonl`
- `results/perturbation_pairs/<dataset>/manifest.json`
- `results/cached_scores/<dataset>/*.established.json`
- `results/cached_scores/<dataset>/*.caf.json`
- `results/cached_scores/caf_pairs.jsonl`
- `results/cached_scores/caf_cache.json`

## Representative verification values

These rounded values are intended as high-level checks, not bitwise checks across
hardware and dependency builds.

| Dataset | Setting | Quantity | Expected value |
|---|---|---|---:|
| Clotho | Original leave-one-out | records | 5,225 |
| Clotho | Original leave-one-out | BLEU-1 mean | 0.6399 |
| Clotho | Original leave-one-out | CIDEr-D mean | 0.9159 |
| Clotho | Original leave-one-out | CLAP-audio mean | 0.4907 |
| AudioCaps | Original leave-one-out | records | 4,060 |
| AudioCaps | Original leave-one-out | BLEU-1 mean | 0.5949 |
| AudioCaps | Original leave-one-out | CIDEr-D mean | 0.7665 |
| AudioCaps | Original leave-one-out | CLAP-audio mean | 0.5808 |

Exact generated pair checksums are recorded locally because absolute source paths and
an explicitly selected configuration are included in release records. A published
archive should ship its own manifests and cached scores from one fixed tagged run.

## Artifact and data-release policy

Release dataset identifiers, retained-example manifests, text perturbation pairs where
permitted, cached scores, aggregate tables, configuration, and random seeds. Do not
redistribute audio or model weights unless their licenses explicitly permit it. The
`.gitignore` protects local audio and generated results; use a dedicated release branch
or `git add -f` only after reviewing artifacts for licensing and path disclosure.

## Versioning and archival

The camera-ready snapshot is intended to be tagged `v1.0-camera-ready`. Record the
exact tag or commit in the paper, archive the tagged repository and permitted artifacts
through Zenodo or an equivalent service, and add the resulting DOI to `CITATION.cff`.
Before tagging, reproduce aggregate tables from a clean environment using either the
released perturbation pairs or the released score cache.

## Tests

Core schema, dataset, perturbation, cache, and paired-analysis tests do not download
models or audio:

```bash
python -m unittest discover -s tests -v
```

The thirteen-metric and CAF/FLEUR stages require their documented external runtimes
and are therefore exercised with `--limit` smoke runs rather than unit downloads.

## Citation

Use [`CITATION.cff`](CITATION.cff) and cite the associated paper. The established
metric implementations are supplied by
[`aac-metrics`](https://github.com/Labbeti/aac-metrics); CAF/FLEUR integration uses
the vendored [`CAF-Score`](https://github.com/inseong00/CAF-Score) submodule.
