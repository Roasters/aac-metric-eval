"""Reusable controlled-perturbation battery for audio-caption metrics."""

from .config import Settings
from .records import EvaluationRecord, Perturbation

__all__ = ["EvaluationRecord", "Perturbation", "Settings"]
__version__ = "1.0.0"
