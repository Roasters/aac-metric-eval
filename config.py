"""
Central configuration for the AAC evaluation battery.

All input/output locations are resolved here so the scripts contain no
machine-specific absolute paths. Override any of them with environment variables
(recommended) or by editing the defaults below.

Environment variables
---------------------
    RESULTS_DIR / CAF_RESULTS_DIR  Where all *_scores.json / CAF caches are written.
    CLOTHO_ROOT                    Root of the Clotho V2 dataset.
    AUDIOCAPS_ROOT                 Root of the AudioCaps dataset.
    FIGURES_DIR                    Where perturbation figures are saved.
    CAF_SRC                        Path to a checkout of the external CAF-Score repo
                                   (only needed for caf/run_clap.py and
                                   run_caf_af3.py).

Example
-------
    export CLOTHO_ROOT=/data/ClothoV2
    export AUDIOCAPS_ROOT=/data/AudioCaps
    export RESULTS_DIR=$PWD/results
    python perturbations/masking.py
"""

import os
from pathlib import Path

# --- Repository root (this file's directory) ---------------------------------
REPO_ROOT = Path(__file__).resolve().parent

# --- Output locations --------------------------------------------------------
# CAF_RESULTS_DIR kept as an accepted alias for backward compatibility with the
# reference-free scripts, which used it originally.
RESULTS_DIR = Path(
    os.environ.get("RESULTS_DIR")
    or os.environ.get("CAF_RESULTS_DIR")
    or (REPO_ROOT / "results")
)
FIGURES_DIR = Path(os.environ.get("FIGURES_DIR", REPO_ROOT / "figures"))

# --- Clotho V2 ---------------------------------------------------------------
# Expected layout:
#   $CLOTHO_ROOT/clotho_csv_files/clotho_captions_<split>.csv
#   $CLOTHO_ROOT/<split>/<file_name>.wav
CLOTHO_ROOT = Path(os.environ.get("CLOTHO_ROOT", "/path/to/ClothoV2"))


def clotho_captions_csv(split: str = "evaluation") -> Path:
    return CLOTHO_ROOT / "clotho_csv_files" / f"clotho_captions_{split}.csv"


def clotho_audio_path(file_name: str, split: str = "evaluation") -> str:
    return str(CLOTHO_ROOT / split / file_name)


def clotho_audio_dir(split: str = "evaluation") -> str:
    return str(CLOTHO_ROOT / split)


# --- AudioCaps ---------------------------------------------------------------
# Expected layout:
#   $AUDIOCAPS_ROOT/eval_text.csv          (single-caption on-disk eval file)
#   $AUDIOCAPS_ROOT/16000/eval/<id>.wav    (16 kHz audio)
AUDIOCAPS_ROOT = Path(os.environ.get("AUDIOCAPS_ROOT", "/path/to/AudioCaps"))
AUDIOCAPS_ONDISK_EVAL = AUDIOCAPS_ROOT / "eval_text.csv"
AUDIOCAPS_AUDIO_DIR = str(AUDIOCAPS_ROOT / "16000" / "eval")

# The 5-reference AudioCaps eval CSV built by audiocaps/build_audiocaps_refs.py,
# and the official multi-caption test split it is derived from.
AUDIOCAPS_REFS_CSV = REPO_ROOT / "csv_files" / "audiocaps_captions_evaluation.csv"
AUDIOCAPS_OFFICIAL_TEST = REPO_ROOT / "third_party" / "audiocaps_meta" / "test.csv"

# --- External CAF-Score source (Audio-Flamingo-3 / FLEUR + LAION-CLAP) --------
# Clone from the CAF-Score authors' repository into this location, or set CAF_SRC.
CAF_SRC = Path(os.environ.get("CAF_SRC", REPO_ROOT / "third_party" / "CAF-Score"))

# Create output dirs on import so scripts can write immediately.
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
