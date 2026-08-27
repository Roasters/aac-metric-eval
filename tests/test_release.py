from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from eval_battery.analysis import paired_drop_rows
from eval_battery.config import Settings
from eval_battery.datasets.common import load_five_caption_csv
from eval_battery.io import read_records, write_records
from eval_battery.metrics.cache import PairScoreCache, pair_key
from eval_battery.perturbations.core import (
    adjacent_swaps,
    fixed_nonword_replacements,
    random_nonword_replacements,
    token_removals,
)
from eval_battery.records import EvaluationRecord


def original(candidate: str = "a dog barks") -> EvaluationRecord:
    return EvaluationRecord.original(
        record_id="test:clip.wav:caption-0",
        dataset="test",
        source_id="clip.wav",
        audio_path="/data/clip.wav",
        candidate=candidate,
        references=("a canine is barking", "dog sound"),
    )


class RecordAndPerturbationTests(unittest.TestCase):
    def test_record_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            records = [original()]
            self.assertEqual(write_records(path, records), 1)
            self.assertEqual(read_records(path), records)

    def test_core_axes_are_aligned_and_deterministic(self) -> None:
        record = original()
        masks = list(fixed_nonword_replacements([record]))
        removals = list(token_removals([record]))
        swaps = list(adjacent_swaps([record]))
        random_a = list(random_nonword_replacements([record], seed=9))
        random_b = list(random_nonword_replacements([record], seed=9))

        self.assertEqual(len(masks), 3)
        self.assertEqual(len(removals), 3)
        self.assertEqual(len(swaps), 2)
        self.assertEqual(
            [item.candidate for item in random_a],
            [item.candidate for item in random_b],
        )
        self.assertTrue(all(item.parent_id == record.record_id for item in masks))
        self.assertEqual(masks[1].candidate, "a xkqjvz barks")
        self.assertEqual(removals[1].candidate, "a barks")
        self.assertEqual(swaps[0].candidate, "dog a barks")


class DatasetAndConfigTests(unittest.TestCase):
    def test_leave_one_out_csv_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "captions.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["file_name", "caption_1", "caption_2", "caption_3"])
                writer.writerow(["clip.wav", "one", "two", "three"])
            records = load_five_caption_csv(csv_path, root / "audio", dataset="sample")

        self.assertEqual(len(records), 3)
        self.assertEqual(records[1].candidate, "two")
        self.assertEqual(records[1].references, ("one", "three"))
        self.assertEqual(records[1].parent_id, records[1].record_id)

    def test_camera_ready_config_resolves_from_repository(self) -> None:
        settings = Settings.load("configs/camera_ready.yaml")
        self.assertEqual(settings.repository_root, Path.cwd().resolve())
        self.assertEqual(settings.perturbation_dir, Path.cwd() / "results/perturbation_pairs")
        self.assertEqual(
            settings.dataset_path("clotho", "captions_csv"),
            Path.cwd() / "data/clotho/clotho_csv_files/clotho_captions_evaluation.csv",
        )

    def test_synonym_selector_is_per_dataset(self) -> None:
        # Paper protocol: Clotho ranks WordNet candidates by RoBERTa MLM;
        # AudioCaps selects by SBERT cosine similarity.
        settings = Settings.load("configs/camera_ready.yaml")
        base = settings.section("perturbations").get("synonym", {})

        def resolved(dataset: str) -> dict:
            merged = dict(base)
            merged.update(settings.dataset(dataset).get("synonym", {}) or {})
            return merged

        self.assertEqual(resolved("clotho")["selector"], "mlm")
        self.assertEqual(resolved("clotho")["model"], "roberta-base")
        self.assertEqual(resolved("audiocaps")["selector"], "embedding")
        self.assertIn("MiniLM", resolved("audiocaps")["model"])


class CacheAndAnalysisTests(unittest.TestCase):
    def test_pair_key_uses_basename_and_separator(self) -> None:
        self.assertEqual(
            pair_key("/left/clip.wav", "caption"),
            pair_key("/right/clip.wav", "caption"),
        )
        self.assertNotEqual(
            pair_key("/left/clip.wav", "caption"),
            pair_key("/left/clip.wav", "caption!"),
        )

    def test_cache_registration_is_idempotent_and_sidecar_stays_aligned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = PairScoreCache(root / "cache.json", root / "pairs.jsonl")
            records = [original("first"), original("second")]
            keys, added = cache.register(records)
            _, added_again = cache.register(records)
            self.assertEqual((added, added_again), (2, 0))

            (root / "cache.json").write_text(
                json.dumps(
                    {
                        keys[0]: {"caf_score": 0.75},
                        keys[1]: {"caf_score": 0.25},
                    }
                ),
                encoding="utf-8",
            )
            sidecar = cache.sidecar(records, fields=("caf_score",))

        self.assertEqual(sidecar["keys"], keys)
        self.assertEqual(sidecar["metrics"]["caf_score"]["scores"], [0.75, 0.25])
        self.assertEqual(sidecar["metrics"]["caf_score"]["score"], 0.5)

    def test_synonym_drop_excludes_unrealized_rows(self) -> None:
        base = original("dog barks")
        realized = base.variant(
            name="synonym",
            axis="lexical_variation",
            position=0,
            candidate="canine barks",
            original="dog",
            replacement="canine",
        )
        identity = base.variant(
            name="synonym",
            axis="lexical_variation",
            position=1,
            candidate="dog barks",
            original="barks",
            replacement="barks",
            realized=False,
        )
        rows = paired_drop_rows(
            [base],
            [realized, identity],
            {"record_ids": [base.record_id], "metrics": {"m": {"scores": [1.0]}}},
            {
                "record_ids": [realized.record_id, identity.record_id],
                "metrics": {"m": {"scores": [0.6, 1.0]}},
            },
            dataset="test",
            perturbation="synonym",
        )
        self.assertEqual(rows[0]["n"], 1)
        self.assertEqual(rows[0]["scope"], "realized_replacements")
        self.assertAlmostEqual(rows[0]["absolute_drop"], 0.4)


if __name__ == "__main__":
    unittest.main()
