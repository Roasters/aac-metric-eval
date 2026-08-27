"""Clotho V2 evaluation adapter."""

from __future__ import annotations

from ..config import Settings
from ..records import EvaluationRecord
from .common import load_five_caption_csv


def load_clotho(settings: Settings) -> list[EvaluationRecord]:
    return load_five_caption_csv(
        settings.dataset_path("clotho", "captions_csv"),
        settings.dataset_path("clotho", "audio_dir"),
        dataset="clotho",
    )
