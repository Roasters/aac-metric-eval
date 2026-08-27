"""Dataset adapters producing a common leave-one-out record schema."""

from __future__ import annotations

from ..config import Settings
from ..records import EvaluationRecord
from .audiocaps import load_audiocaps
from .clotho import load_clotho


def load_dataset(name: str, settings: Settings) -> list[EvaluationRecord]:
    if name == "clotho":
        return load_clotho(settings)
    if name == "audiocaps":
        return load_audiocaps(settings)
    raise ValueError(f"Unsupported dataset: {name}")


__all__ = ["load_dataset", "load_audiocaps", "load_clotho"]
