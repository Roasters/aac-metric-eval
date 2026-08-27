"""Unified adapter for the thirteen established AAC metrics."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from ..records import EvaluationRecord


ESTABLISHED_METRICS = (
    "bleu_1",
    "bleu_2",
    "bleu_3",
    "bleu_4",
    "rouge_l",
    "meteor",
    "cider_d",
    "spice",
    "spider",
    "fense",
    "clap_sim_text",
    "sbert_sim",
    "clap_sim_audio",
)

_BASE_METRICS = ("bleu_1", "bleu_2", "bleu_3", "bleu_4", "rouge_l", "meteor")


def _values(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]


def _resolve_device(requested: str):
    import torch

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def _score_batch(
    records: Sequence[EvaluationRecord],
    selected: set[str],
    *,
    device: Any,
    spice_failure_policy: str,
    spice_java_memory: str,
) -> dict[str, list[float]]:
    import numpy as np
    from aac_metrics import evaluate
    from aac_metrics.functional import cider_d, clap_sim, fense, spice
    from aac_metrics.utils.tokenization import preprocess_mono_sents, preprocess_mult_sents

    candidates = preprocess_mono_sents([record.candidate for record in records])
    references = preprocess_mult_sents([list(record.references) for record in records])
    audio_paths = [record.audio_path for record in records]
    size = len(records)
    result = {name: [float("nan")] * size for name in selected}

    base = [name for name in _BASE_METRICS if name in selected]
    if base:
        _, sentence_scores = evaluate(candidates, references, metrics=base)
        for name in base:
            result[name] = _values(sentence_scores[name])

    spice_values: list[float] | None = None
    if {"spice", "spider"} & selected:
        try:
            _, sentence_scores = spice(
                candidates, references, java_max_memory=spice_java_memory
            )
            spice_values = _values(sentence_scores["spice"])
        except Exception:
            if spice_failure_policy == "raise":
                raise
            if spice_failure_policy != "isolate":
                spice_values = [float("nan")] * size
            else:
                spice_values = []
                for candidate, reference in zip(candidates, references):
                    try:
                        _, sentence_scores = spice(
                            [candidate], [reference], java_max_memory=spice_java_memory
                        )
                        spice_values.append(float(_values(sentence_scores["spice"])[0]))
                    except Exception:
                        spice_values.append(float("nan"))
        if "spice" in selected:
            result["spice"] = spice_values

    cider_values: list[float] | None = None
    if {"cider_d", "spider"} & selected:
        _, sentence_scores = cider_d(candidates, references)
        cider_values = _values(sentence_scores["cider_d"])
        if "cider_d" in selected:
            result["cider_d"] = cider_values

    if "spider" in selected:
        assert spice_values is not None and cider_values is not None
        result["spider"] = [
            float("nan") if np.isnan(spice_value) or np.isnan(cider_value)
            else 0.5 * spice_value + 0.5 * cider_value
            for spice_value, cider_value in zip(spice_values, cider_values)
        ]

    if {"fense", "sbert_sim"} & selected:
        _, sentence_scores = fense(candidates, references, device=device)
        if "fense" in selected:
            result["fense"] = _values(sentence_scores["fense"])
        if "sbert_sim" in selected:
            result["sbert_sim"] = _values(sentence_scores["sbert_sim"])

    if "clap_sim_text" in selected:
        _, sentence_scores = clap_sim(candidates, references, device=device)
        result["clap_sim_text"] = _values(sentence_scores["clap_sim"])

    if "clap_sim_audio" in selected:
        _, sentence_scores = clap_sim(
            candidates,
            references,
            clap_method="audio",
            device=device,
            audio_paths=audio_paths,
        )
        result["clap_sim_audio"] = _values(sentence_scores["clap_sim"])

    for name, values in result.items():
        if len(values) != size:
            raise RuntimeError(f"Metric {name} returned {len(values)} values for {size} rows")
    return result


def score_established_metrics(
    records: Iterable[EvaluationRecord],
    *,
    metrics: Sequence[str] = ESTABLISHED_METRICS,
    batch_size: int = 1024,
    device: str = "auto",
    spice_failure_policy: str = "isolate",
    spice_java_memory: str = "16G",
) -> dict[str, Any]:
    """Score records without allowing a failed metric to change row alignment."""

    import numpy as np

    records = list(records)
    selected = set(metrics)
    unknown = selected.difference(ESTABLISHED_METRICS)
    if unknown:
        raise ValueError(f"Unknown established metrics: {sorted(unknown)}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    score_arrays = {name: [] for name in metrics}
    torch_device = _resolve_device(device)
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        batch_scores = _score_batch(
            batch,
            selected,
            device=torch_device,
            spice_failure_policy=spice_failure_policy,
            spice_java_memory=spice_java_memory,
        )
        for name in metrics:
            score_arrays[name].extend(batch_scores[name])

    metric_payload: dict[str, dict[str, Any]] = {}
    for name, values in score_arrays.items():
        array = np.asarray(values, dtype=float)
        mean = float(np.nanmean(array)) if array.size and not np.all(np.isnan(array)) else None
        metric_payload[name] = {"score": mean, "scores": values}
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metric_suite": "established-13",
        "record_ids": [record.record_id for record in records],
        "parent_ids": [record.parent_id for record in records],
        "metrics": metric_payload,
        "evaluation": {
            "batch_size": batch_size,
            "device": str(torch_device),
            "spice_failure_policy": spice_failure_policy,
            "spice_java_memory": spice_java_memory,
        },
    }
