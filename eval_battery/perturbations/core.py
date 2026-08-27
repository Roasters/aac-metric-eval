"""Metric-independent destructive and structural perturbations."""

from __future__ import annotations

import random
import string
from collections.abc import Iterable, Iterator

from ..records import EvaluationRecord


def fixed_nonword_replacements(
    records: Iterable[EvaluationRecord], nonword: str = "xkqjvz"
) -> Iterator[EvaluationRecord]:
    for record in records:
        tokens = record.candidate.split()
        for position, original in enumerate(tokens):
            variant = tokens.copy()
            variant[position] = nonword
            yield record.variant(
                name="masking",
                axis="meaning_destroying",
                position=position,
                candidate=" ".join(variant),
                original=original,
                replacement=nonword,
                metadata={"sequence_length": len(tokens)},
            )


def token_removals(records: Iterable[EvaluationRecord]) -> Iterator[EvaluationRecord]:
    for record in records:
        tokens = record.candidate.split()
        for position, original in enumerate(tokens):
            variant = tokens[:position] + tokens[position + 1 :]
            yield record.variant(
                name="removal",
                axis="meaning_destroying",
                position=position,
                candidate=" ".join(variant),
                original=original,
                replacement="",
                metadata={"sequence_length": len(tokens)},
            )


def random_nonword_replacements(
    records: Iterable[EvaluationRecord], *, seed: int = 42, length: int = 6
) -> Iterator[EvaluationRecord]:
    rng = random.Random(seed)
    for record in records:
        tokens = record.candidate.split()
        for position, original in enumerate(tokens):
            replacement = "".join(rng.choices(string.ascii_lowercase, k=length))
            variant = tokens.copy()
            variant[position] = replacement
            yield record.variant(
                name="random_nonword",
                axis="meaning_destroying",
                position=position,
                candidate=" ".join(variant),
                original=original,
                replacement=replacement,
                metadata={"sequence_length": len(tokens), "seed": seed},
            )


def adjacent_swaps(records: Iterable[EvaluationRecord]) -> Iterator[EvaluationRecord]:
    for record in records:
        tokens = record.candidate.split()
        for position in range(max(0, len(tokens) - 1)):
            left, right = tokens[position], tokens[position + 1]
            variant = tokens.copy()
            variant[position], variant[position + 1] = right, left
            yield record.variant(
                name="swap_adjacent",
                axis="word_order",
                position=position,
                candidate=" ".join(variant),
                original=f"{left} {right}",
                replacement=f"{right} {left}",
                metadata={"sequence_length": len(tokens), "right_position": position + 1},
            )
