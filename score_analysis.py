"""
Metric Score Analysis
---------------------
Computes per-metric mean/variance and pairwise Pearson & Spearman
correlations from a JSON file of the form:

    {
        "metric_name": {
            "score": <corpus_float>,
            "scores": [<per_sample_float>, ...]
        },
        ...
    }

Usage:
    python score_analysis.py [path/to/scores.json]
"""

import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from pathlib import Path

# Path to a scores JSON is required as the first CLI argument, e.g. any
# *_scores.json produced by the perturbation scripts under RESULTS_DIR.

def load_scores(path: str) -> dict[str, np.ndarray]:
    with open(path) as f:
        data = json.load(f)
    scores = {metric: np.array(entry["scores"], dtype=float) for metric, entry in data.items()}

    # Drop metrics that are entirely NaN (not computed, e.g. clap_sim_audio without audio paths)
    fully_nan = [m for m, arr in scores.items() if np.all(np.isnan(arr))]
    if fully_nan:
        print(f"  Dropping fully-NaN metrics (not computed): {fully_nan}")
        for m in fully_nan:
            del scores[m]

    # Build valid mask iteratively — exclude samples that are NaN in any metric
    n_samples = len(next(iter(scores.values())))
    valid = np.ones(n_samples, dtype=bool)
    for arr in scores.values():
        valid &= ~np.isnan(arr)
    n_dropped = int((~valid).sum())
    if n_dropped:
        print(f"  Filtered out {n_dropped} samples with NaN scores (SPICE failures).")
        scores = {metric: arr[valid] for metric, arr in scores.items()}

    return scores


def stats_table(scores: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for metric, arr in scores.items():
        mean = float(np.nanmean(arr))
        std  = float(np.nanstd(arr))
        rows.append({
            "metric":   metric,
            "mean":     mean,
            "std":      std,
            "variance": float(np.nanvar(arr)),
            "cv":       std / mean if mean != 0 else float('nan'),
            "min":      float(np.nanmin(arr)),
            "max":      float(np.nanmax(arr)),
        })
    return pd.DataFrame(rows).set_index("metric")


def correlation_matrix(
    scores: dict[str, np.ndarray],
    method: str = "spearman",
) -> pd.DataFrame:
    metrics = list(scores.keys())
    n = len(metrics)
    mat = np.ones((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            a, b = scores[metrics[i]], scores[metrics[j]]
            valid = ~(np.isnan(a) | np.isnan(b))
            if valid.sum() < 2:
                r = float('nan')
            elif method == "pearson":
                r, _ = pearsonr(a[valid], b[valid])
            else:
                r, _ = spearmanr(a[valid], b[valid])
            mat[i, j] = mat[j, i] = r

    return pd.DataFrame(mat, index=metrics, columns=metrics)


def main(path: str):
    print(f"Loading {path} ...")
    scores = load_scores(path)
    metrics = list(scores.keys())
    n_samples = next(iter(scores.values())).shape[0]
    print(f"  {len(metrics)} metrics, {n_samples} samples\n")

    # ── Per-metric statistics ────────────────────────────────────────────────
    stats = stats_table(scores)
    print("=== Per-metric Statistics ===")
    print(stats.round(4).to_string())

    # ── Correlation matrices ─────────────────────────────────────────────────
    pearson_df  = correlation_matrix(scores, method="pearson")
    spearman_df = correlation_matrix(scores, method="spearman")

    print("\n=== Pearson Correlation Matrix ===")
    print(pearson_df.round(3).to_string())

    print("\n=== Spearman Correlation Matrix ===")
    print(spearman_df.round(3).to_string())

    # ── Highlight strong / weak pairs ───────────────────────────────────────
    print("\n=== Spearman: Metric Pairs Sorted by Correlation ===")
    pairs = []
    for i, m1 in enumerate(metrics):
        for j, m2 in enumerate(metrics):
            if j <= i:
                continue
            pairs.append((m1, m2, spearman_df.loc[m1, m2]))
    pairs.sort(key=lambda x: x[2], reverse=True)

    pair_df = pd.DataFrame(pairs, columns=["metric_a", "metric_b", "spearman_r"])
    print(pair_df.round(3).to_string(index=False))

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir = Path(path).parent / "candidate_intra"
    out_dir.mkdir(exist_ok=True)
    stats.round(4).to_csv(out_dir / "metric_stats.csv")
    pearson_df.round(4).to_csv(out_dir / "pearson_corr.csv")
    spearman_df.round(4).to_csv(out_dir / "spearman_corr.csv")
    pair_df.round(4).to_csv(out_dir / "spearman_pairs.csv", index=False)

    print(f"\nSaved to {out_dir}/")
    print("  metric_stats.csv")
    print("  pearson_corr.csv")
    print("  spearman_corr.csv")
    print("  spearman_pairs.csv")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python score_analysis.py <path/to/scores.json>")
    main(sys.argv[1])
