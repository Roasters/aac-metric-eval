"""Shared multi-reference CSV parsing."""

from __future__ import annotations

import csv
from pathlib import Path

from ..records import EvaluationRecord


def load_five_caption_csv(
    csv_path: str | Path,
    audio_dir: str | Path,
    *,
    dataset: str,
) -> list[EvaluationRecord]:
    """Load a Clotho-shaped CSV and return five leave-one-out rows per source."""

    csv_path, audio_dir = Path(csv_path), Path(audio_dir)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Caption CSV does not exist: {csv_path}")

    records: list[EvaluationRecord] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Caption CSV has no header: {csv_path}")
        file_column = next(
            (name for name in reader.fieldnames if name.strip().lower() == "file_name"),
            None,
        )
        caption_columns = [
            name for name in reader.fieldnames if name.strip().lower().startswith("caption")
        ]
        if file_column is None or len(caption_columns) < 2:
            raise ValueError(
                f"Expected file_name and caption_* columns in {csv_path}; "
                f"found {reader.fieldnames}"
            )
        for source_index, row in enumerate(reader):
            source_id = row[file_column].strip()
            captions = [row[name].strip() for name in caption_columns if row.get(name, "").strip()]
            if len(captions) < 2:
                continue
            audio_path = str((audio_dir / source_id).resolve())
            for caption_index, candidate in enumerate(captions):
                record_id = f"{dataset}:{source_id}:caption-{caption_index}"
                records.append(
                    EvaluationRecord.original(
                        record_id=record_id,
                        dataset=dataset,
                        source_id=source_id,
                        audio_path=audio_path,
                        candidate=candidate,
                        references=captions[:caption_index] + captions[caption_index + 1 :],
                        metadata={"source_index": source_index, "caption_index": caption_index},
                    )
                )
    return records
