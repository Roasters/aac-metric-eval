"""Aligned descriptive statistics and paired perturbation-drop summaries."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .io import read_json, read_records, write_json
from .records import EvaluationRecord


def metric_arrays(artifact: Mapping[str, Any]) -> dict[str, list[float | None]]:
    """Read both the v1 release schema and legacy metric-name-at-root schema."""

    payload = artifact.get("metrics", artifact)
    arrays: dict[str, list[float | None]] = {}
    for name, value in payload.items():
        if isinstance(value, dict) and isinstance(value.get("scores"), list):
            arrays[name] = value["scores"]
    return arrays


def combine_artifacts(*artifacts: Mapping[str, Any] | None) -> dict[str, Any]:
    record_ids: list[str] | None = None
    metrics: dict[str, Any] = {}
    for artifact in artifacts:
        if not artifact:
            continue
        current_ids = artifact.get("record_ids")
        if current_ids is not None:
            if record_ids is not None and list(current_ids) != record_ids:
                raise ValueError("Cannot combine score artifacts with different record order.")
            record_ids = list(current_ids)
        metrics.update({name: {"scores": values} for name, values in metric_arrays(artifact).items()})
    return {"record_ids": record_ids or [], "metrics": metrics}


def descriptive_rows(
    artifact: Mapping[str, Any], *, dataset: str, setting: str
) -> list[dict[str, Any]]:
    rows = []
    for metric, values in metric_arrays(artifact).items():
        array = np.asarray([np.nan if value is None else value for value in values], dtype=float)
        valid = array[~np.isnan(array)]
        if not valid.size:
            continue
        mean = float(valid.mean())
        rows.append(
            {
                "dataset": dataset,
                "setting": setting,
                "metric": metric,
                "n": int(valid.size),
                "mean": mean,
                "std": float(valid.std()),
                "variance": float(valid.var()),
                "cv": float(valid.std() / mean) if mean else None,
                "min": float(valid.min()),
                "max": float(valid.max()),
            }
        )
    return rows


def correlation_matrix(
    artifact: Mapping[str, Any], *, method: str
) -> tuple[list[str], np.ndarray]:
    arrays = metric_arrays(artifact)
    names = list(arrays)
    matrix = np.eye(len(names), dtype=float)
    if method == "spearman":
        from scipy.stats import spearmanr
    elif method != "pearson":
        raise ValueError("method must be 'pearson' or 'spearman'")
    for left_index, left_name in enumerate(names):
        left = np.asarray(
            [np.nan if value is None else value for value in arrays[left_name]], dtype=float
        )
        for right_index in range(left_index + 1, len(names)):
            right = np.asarray(
                [np.nan if value is None else value for value in arrays[names[right_index]]],
                dtype=float,
            )
            valid = ~(np.isnan(left) | np.isnan(right))
            if valid.sum() < 2:
                correlation = np.nan
            elif method == "pearson":
                correlation = float(np.corrcoef(left[valid], right[valid])[0, 1])
            else:
                correlation = float(spearmanr(left[valid], right[valid]).statistic)
            matrix[left_index, right_index] = matrix[right_index, left_index] = correlation
    return names, matrix


def paired_drop_rows(
    original_records: Sequence[EvaluationRecord],
    perturbed_records: Sequence[EvaluationRecord],
    original_artifact: Mapping[str, Any],
    perturbed_artifact: Mapping[str, Any],
    *,
    dataset: str,
    perturbation: str,
) -> list[dict[str, Any]]:
    """Compute paired drops; synonym identity rows are excluded by construction."""

    original_ids = list(original_artifact.get("record_ids") or [r.record_id for r in original_records])
    perturbed_ids = list(
        perturbed_artifact.get("record_ids") or [r.record_id for r in perturbed_records]
    )
    if len(original_ids) != len(original_records):
        raise ValueError("Original record and score counts differ.")
    if len(perturbed_ids) != len(perturbed_records):
        raise ValueError("Perturbed record and score counts differ.")

    original_arrays = metric_arrays(original_artifact)
    perturbed_arrays = metric_arrays(perturbed_artifact)
    shared_metrics = sorted(set(original_arrays) & set(perturbed_arrays))
    for metric in shared_metrics:
        if len(original_arrays[metric]) != len(original_records):
            raise ValueError(f"Original {metric} score and record counts differ.")
        if len(perturbed_arrays[metric]) != len(perturbed_records):
            raise ValueError(f"Perturbed {metric} score and record counts differ.")
    original_position = {record_id: index for index, record_id in enumerate(original_ids)}
    rows = []
    realized_only = perturbation == "synonym"

    for metric in shared_metrics:
        base_values, perturbed_values = [], []
        for perturbed_index, record in enumerate(perturbed_records):
            if realized_only and not record.perturbation.realized:
                continue
            base_index = original_position.get(record.parent_id)
            if base_index is None:
                raise ValueError(f"Unknown parent_id in {perturbation}: {record.parent_id}")
            base = original_arrays[metric][base_index]
            changed = perturbed_arrays[metric][perturbed_index]
            if base is None or changed is None:
                continue
            base, changed = float(base), float(changed)
            if math.isnan(base) or math.isnan(changed):
                continue
            base_values.append(base)
            perturbed_values.append(changed)
        if not base_values:
            continue
        base_array = np.asarray(base_values)
        perturbed_array = np.asarray(perturbed_values)
        base_mean = float(base_array.mean())
        absolute_drop = float((base_array - perturbed_array).mean())
        rows.append(
            {
                "dataset": dataset,
                "metric": metric,
                "perturbation": perturbation,
                "scope": "realized_replacements" if realized_only else "all_pairs",
                "n": len(base_values),
                "original_mean": base_mean,
                "perturbed_mean": float(perturbed_array.mean()),
                "absolute_drop": absolute_drop,
                "pct_drop": 100.0 * absolute_drop / base_mean if base_mean else None,
                "gain_rate": float((perturbed_array > base_array).mean()),
            }
        )
    return rows


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_matrix(path: str | Path, names: Sequence[str], matrix: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", *names])
        for name, values in zip(names, matrix):
            writer.writerow([name, *values.tolist()])


def load_optional(path: str | Path) -> Mapping[str, Any] | None:
    path = Path(path)
    return read_json(path) if path.is_file() else None


def summarize_dataset(
    dataset: str,
    pair_dir: str | Path,
    score_dir: str | Path,
    table_dir: str | Path,
    *,
    expected: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    pair_dir, score_dir, table_dir = Path(pair_dir), Path(score_dir), Path(table_dir)
    original_records = read_records(pair_dir / "original.jsonl")
    original = combine_artifacts(
        load_optional(score_dir / "original.established.json"),
        load_optional(score_dir / "original.caf.json"),
    )
    if not metric_arrays(original):
        raise FileNotFoundError(f"No original score artifacts found under {score_dir}")

    stats = descriptive_rows(original, dataset=dataset, setting="original")
    drops: list[dict[str, Any]] = []
    verification: dict[str, Any] = {
        "dataset": dataset,
        "original_records": len(original_records),
        "perturbations": {},
    }
    for name in ("masking", "synonym", "swap_adjacent", "removal", "random_nonword"):
        pair_path = pair_dir / f"{name}.jsonl"
        if not pair_path.is_file():
            continue
        records = read_records(pair_path)
        artifact = combine_artifacts(
            load_optional(score_dir / f"{name}.established.json"),
            load_optional(score_dir / f"{name}.caf.json"),
        )
        if metric_arrays(artifact):
            stats.extend(descriptive_rows(artifact, dataset=dataset, setting=name))
            drops.extend(
                paired_drop_rows(
                    original_records,
                    records,
                    original,
                    artifact,
                    dataset=dataset,
                    perturbation=name,
                )
            )
        realized = sum(record.perturbation.realized for record in records)
        verification["perturbations"][name] = {
            "rows": len(records),
            "realized": realized,
        }

    if expected:
        verification["expected"] = dict(expected)
        synonym = verification["perturbations"].get("synonym", {})
        verification["matches_expected"] = {
            "synonym_positions": synonym.get("rows") == expected.get("synonym_positions"),
            "synonym_replacements": synonym.get("realized")
            == expected.get("synonym_replacements"),
        }

    table_dir.mkdir(parents=True, exist_ok=True)
    write_csv(table_dir / "metric_stats.csv", stats)
    write_csv(table_dir / "paired_drops.csv", drops)
    for method in ("pearson", "spearman"):
        names, matrix = correlation_matrix(original, method=method)
        write_matrix(table_dir / f"original_{method}_correlation.csv", names, matrix)
    write_json(table_dir / "verification.json", verification)
    return verification
