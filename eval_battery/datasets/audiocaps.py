"""AudioCaps preparation and leave-one-out adapter."""

from __future__ import annotations

import collections
import csv
import re
from pathlib import Path

from ..config import Settings
from ..records import EvaluationRecord
from .common import load_five_caption_csv


def normalize_caption(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def prepare_audiocaps_references(
    official_test_csv: str | Path,
    ondisk_eval_csv: str | Path,
    audio_dir: str | Path,
    output_csv: str | Path,
) -> dict[str, int]:
    """Match an on-disk one-caption evaluation set to official five-caption rows."""

    official_test_csv = Path(official_test_csv)
    ondisk_eval_csv = Path(ondisk_eval_csv)
    audio_dir = Path(audio_dir)
    output_csv = Path(output_csv)

    youtube_to_captions: dict[str, list[str]] = collections.OrderedDict()
    with official_test_csv.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            captions = youtube_to_captions.setdefault(row["youtube_id"], [])
            if row["caption"] not in captions:
                captions.append(row["caption"])

    caption_to_youtube: dict[str, set[str]] = collections.defaultdict(set)
    for youtube_id, captions in youtube_to_captions.items():
        for caption in captions:
            caption_to_youtube[normalize_caption(caption)].add(youtube_id)

    youtube_to_wav: dict[str, str] = {}
    ambiguous = unmatched = 0
    with ondisk_eval_csv.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split(",", 1)
            if len(parts) != 2:
                continue
            wav_id, caption = parts
            matches = caption_to_youtube.get(normalize_caption(caption), set())
            if not matches:
                unmatched += 1
                continue
            if len(matches) != 1:
                ambiguous += 1
                continue
            youtube_id = next(iter(matches))
            wav_name = f"{wav_id}.wav"
            if (audio_dir / wav_name).is_file() and youtube_id not in youtube_to_wav:
                youtube_to_wav[youtube_id] = wav_name

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    written = skipped_short = 0
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", "file_name", "caption_1", "caption_2", "caption_3", "caption_4", "caption_5"])
        for youtube_id, captions in youtube_to_captions.items():
            if youtube_id not in youtube_to_wav:
                continue
            if len(captions) < 5:
                skipped_short += 1
                continue
            writer.writerow([written, youtube_to_wav[youtube_id], *captions[:5]])
            written += 1
    return {
        "official_clips": len(youtube_to_captions),
        "ambiguous": ambiguous,
        "unmatched": unmatched,
        "short_reference_sets": skipped_short,
        "written_clips": written,
    }


def load_audiocaps(settings: Settings) -> list[EvaluationRecord]:
    return load_five_caption_csv(
        settings.dataset_path("audiocaps", "references_csv"),
        settings.dataset_path("audiocaps", "audio_dir"),
        dataset="audiocaps",
    )
