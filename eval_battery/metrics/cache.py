"""Stable-key cache for expensive source-candidate metric components."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..io import read_json, write_json
from ..records import EvaluationRecord


_SEPARATOR = "\x1f"


def pair_key(audio_path: str, caption: str, *, key_mode: str = "basename") -> str:
    if key_mode == "basename":
        source = os.path.basename(audio_path) if audio_path else ""
    elif key_mode == "path":
        source = os.path.abspath(audio_path) if audio_path else ""
    else:
        raise ValueError(f"Unknown cache key mode: {key_mode}")
    return hashlib.sha1(f"{source}{_SEPARATOR}{caption or ''}".encode("utf-8")).hexdigest()


class PairScoreCache:
    """Register missing pairs and resolve component scores without row drift."""

    def __init__(
        self,
        cache_path: str | Path,
        manifest_path: str | Path,
        *,
        key_mode: str = "basename",
    ) -> None:
        self.cache_path = Path(cache_path)
        self.manifest_path = Path(manifest_path)
        self.key_mode = key_mode
        self._lock = threading.Lock()

    def load(self) -> dict[str, dict[str, Any]]:
        if not self.cache_path.exists():
            return {}
        value = read_json(self.cache_path)
        if not isinstance(value, dict):
            raise ValueError(f"CAF cache must be a mapping: {self.cache_path}")
        return value

    def save(self, cache: dict[str, dict[str, Any]]) -> None:
        write_json(self.cache_path, cache)

    def registered_keys(self) -> set[str]:
        keys: set[str] = set()
        if not self.manifest_path.exists():
            return keys
        with self.manifest_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    keys.add(json.loads(line)["key"])
                except (json.JSONDecodeError, KeyError):
                    continue
        return keys

    def register(self, records: Iterable[EvaluationRecord]) -> tuple[list[str], int]:
        rows = []
        keys = []
        for record in records:
            key = pair_key(record.audio_path, record.candidate, key_mode=self.key_mode)
            keys.append(key)
            rows.append(
                {
                    "key": key,
                    "audio_path": record.audio_path,
                    "caption": record.candidate,
                    "source_id": record.source_id,
                }
            )
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        added = 0
        with self._lock:
            known = self.registered_keys()
            with self.manifest_path.open("a", encoding="utf-8") as handle:
                for row in rows:
                    if row["key"] in known:
                        continue
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    known.add(row["key"])
                    added += 1
        return keys, added

    def sidecar(
        self,
        records: Sequence[EvaluationRecord],
        *,
        fields: Sequence[str] = ("caf_score", "clap_score", "fleur_score"),
    ) -> dict[str, Any]:
        keys, _ = self.register(records)
        cache = self.load()
        payload: dict[str, Any] = {
            "schema_version": 1,
            "record_ids": [record.record_id for record in records],
            "parent_ids": [record.parent_id for record in records],
            "keys": keys,
            "metrics": {},
        }
        for field in fields:
            values = [
                float(cache[key][field])
                if key in cache and cache[key].get(field) is not None
                else None
                for key in keys
            ]
            payload["metrics"][field] = {"score": _mean(values), "scores": values}
        return payload

    def refill(self, sidecar: dict[str, Any]) -> dict[str, Any]:
        cache = self.load()
        keys = sidecar.get("keys")
        if not isinstance(keys, list):
            raise ValueError("Sidecar does not contain a key list.")
        for field in ("caf_score", "clap_score", "fleur_score"):
            values = [
                float(cache[key][field])
                if key in cache and cache[key].get(field) is not None
                else None
                for key in keys
            ]
            sidecar.setdefault("metrics", {})[field] = {
                "score": _mean(values),
                "scores": values,
            }
        return sidecar


def _mean(values: Sequence[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None
