"""Stable, serializable records shared by datasets, perturbations, and metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class Perturbation:
    name: str
    axis: str
    position: int | None = None
    original: str | None = None
    replacement: str | None = None
    realized: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Perturbation":
        return cls(
            name=str(value["name"]),
            axis=str(value["axis"]),
            position=value.get("position"),
            original=value.get("original"),
            replacement=value.get("replacement"),
            realized=bool(value.get("realized", True)),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class EvaluationRecord:
    record_id: str
    parent_id: str
    dataset: str
    source_id: str
    audio_path: str
    candidate: str
    references: tuple[str, ...]
    perturbation: Perturbation
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.record_id:
            raise ValueError("record_id must be non-empty")
        if not self.parent_id:
            raise ValueError("parent_id must be non-empty")
        if not isinstance(self.references, tuple):
            object.__setattr__(self, "references", tuple(self.references))

    @classmethod
    def original(
        cls,
        *,
        record_id: str,
        dataset: str,
        source_id: str,
        audio_path: str,
        candidate: str,
        references: Sequence[str],
        metadata: Mapping[str, Any] | None = None,
    ) -> "EvaluationRecord":
        return cls(
            record_id=record_id,
            parent_id=record_id,
            dataset=dataset,
            source_id=source_id,
            audio_path=audio_path,
            candidate=candidate,
            references=tuple(references),
            perturbation=Perturbation(name="original", axis="baseline"),
            metadata=dict(metadata or {}),
        )

    def variant(
        self,
        *,
        name: str,
        axis: str,
        position: int,
        candidate: str,
        original: str,
        replacement: str,
        realized: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> "EvaluationRecord":
        return EvaluationRecord(
            record_id=f"{self.parent_id}:{name}:{position}",
            parent_id=self.parent_id,
            dataset=self.dataset,
            source_id=self.source_id,
            audio_path=self.audio_path,
            candidate=candidate,
            references=self.references,
            perturbation=Perturbation(
                name=name,
                axis=axis,
                position=position,
                original=original,
                replacement=replacement,
                realized=realized,
                metadata=dict(metadata or {}),
            ),
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["references"] = list(self.references)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvaluationRecord":
        return cls(
            record_id=str(value["record_id"]),
            parent_id=str(value["parent_id"]),
            dataset=str(value["dataset"]),
            source_id=str(value["source_id"]),
            audio_path=str(value.get("audio_path", "")),
            candidate=str(value["candidate"]),
            references=tuple(str(item) for item in value.get("references", [])),
            perturbation=Perturbation.from_dict(value["perturbation"]),
            metadata=dict(value.get("metadata", {})),
        )
