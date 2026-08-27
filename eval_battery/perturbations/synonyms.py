"""WordNet synonym replacement with metric-independent contextual ranking."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from ..records import EvaluationRecord


FUNCTION_WORD_TAGS = {
    "DT", "IN", "CC", "PRP", "PRP$", "WDT", "WP", "WP$", "WRB", "MD",
    "TO", "EX", "PDT", "POS", "LS", "RP", "UH", "FW", "SYM", "$", ",",
    ".", ":", "(", ")", "``", "''", "#",
}


@dataclass
class SynonymGenerator:
    """Generate deterministic WordNet candidates and rank them in context.

    The camera-ready configuration uses ``selector='mlm'`` with RoBERTa, attempts
    every whitespace-token position, and permits multi-word WordNet lemmas. The
    stricter single-token/content-word filters remain available as an ablation.
    """

    selector: str = "mlm"
    model_name: str = "roberta-base"
    top_k: int = 250
    single_token_only: bool = False
    content_words_only: bool = False
    device: str = "auto"
    download_nltk: bool = False
    _mlm: Any = field(default=None, init=False, repr=False)
    _embedder: Any = field(default=None, init=False, repr=False)
    _nltk_module: Any = field(default=None, init=False, repr=False)
    _wordnet: Any = field(default=None, init=False, repr=False)

    def _nltk(self):
        if self._nltk_module is not None:
            return self._nltk_module, self._wordnet
        try:
            import nltk
            from nltk.corpus import wordnet
        except ImportError as exc:  # pragma: no cover - installation error
            raise RuntimeError("NLTK is required for synonym perturbations.") from exc
        if self.download_nltk:
            nltk.download("wordnet", quiet=True)
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        try:
            wordnet.ensure_loaded()
            nltk.pos_tag(["sound"])
        except LookupError as exc:
            raise RuntimeError(
                "Missing NLTK resources. Run: python -m nltk.downloader "
                "wordnet averaged_perceptron_tagger_eng"
            ) from exc
        self._nltk_module, self._wordnet = nltk, wordnet
        return self._nltk_module, self._wordnet

    @staticmethod
    def _wordnet_pos(tag: str, wordnet: Any) -> str | None:
        if tag.startswith("J"):
            return wordnet.ADJ
        if tag.startswith("V"):
            return wordnet.VERB
        if tag.startswith("N"):
            return wordnet.NOUN
        if tag.startswith("R"):
            return wordnet.ADV
        return None

    def candidates(self, word: str, tag: str) -> list[str]:
        _, wordnet = self._nltk()
        pos = self._wordnet_pos(tag, wordnet)
        synsets = wordnet.synsets(word, pos=pos) if pos else wordnet.synsets(word)
        values: list[str] = []
        seen: set[str] = set()
        for synset in synsets:
            for lemma in synset.lemmas():
                candidate = lemma.name().replace("_", " ").lower()
                if candidate == word.lower() or candidate in seen:
                    continue
                if self.single_token_only and " " in candidate:
                    continue
                values.append(candidate)
                seen.add(candidate)
        return values

    def _load_mlm(self):
        if self._mlm is None:
            try:
                import torch
                from transformers import pipeline
            except ImportError as exc:  # pragma: no cover - installation error
                raise RuntimeError("transformers and torch are required for MLM ranking.") from exc
            device = 0 if self.device == "auto" and torch.cuda.is_available() else -1
            if self.device not in {"auto", "cpu"}:
                device = int(self.device)
            self._mlm = pipeline("fill-mask", model=self.model_name, device=device)
        return self._mlm

    def _rank_mlm(self, tokens: list[str], position: int, candidates: list[str]) -> str:
        mlm = self._load_mlm()
        masked = tokens.copy()
        masked[position] = mlm.tokenizer.mask_token
        try:
            predictions = mlm(" ".join(masked), top_k=self.top_k)
        except Exception:
            return candidates[0]
        probabilities = {
            item["token_str"].strip().lower(): float(item["score"])
            for item in predictions
        }
        return max(
            candidates,
            key=lambda value: probabilities.get(value.split()[0].lower(), 0.0),
        )

    def _load_embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - installation error
                raise RuntimeError("sentence-transformers is required for embedding ranking.") from exc
            self._embedder = SentenceTransformer(self.model_name)
        return self._embedder

    def _rank_embedding(
        self, sentence: str, tokens: list[str], position: int, candidates: list[str]
    ) -> str:
        import numpy as np

        alternatives = []
        for candidate in candidates:
            variant = tokens.copy()
            variant[position] = candidate
            alternatives.append(" ".join(variant))
        vectors = self._load_embedder().encode(
            [sentence, *alternatives], convert_to_numpy=True
        )
        base = vectors[0]
        norms = np.linalg.norm(vectors[1:], axis=1) * np.linalg.norm(base)
        similarities = (vectors[1:] @ base) / np.maximum(norms, 1e-12)
        return candidates[int(np.argmax(similarities))]

    def select(
        self, sentence: str, tokens: list[str], position: int, tag: str
    ) -> tuple[str | None, dict[str, Any]]:
        word = tokens[position]
        if self.content_words_only and tag in FUNCTION_WORD_TAGS:
            return None, {"skip_reason": "function_word", "pos": tag}
        candidates = self.candidates(word, tag)
        if not candidates:
            return None, {"skip_reason": "no_synonym", "pos": tag}
        if len(candidates) == 1:
            selected = candidates[0]
        elif self.selector == "mlm":
            selected = self._rank_mlm(tokens, position, candidates)
        elif self.selector == "embedding":
            selected = self._rank_embedding(sentence, tokens, position, candidates)
        elif self.selector == "first":
            selected = candidates[0]
        else:
            raise ValueError(f"Unknown synonym selector: {self.selector}")
        return selected, {
            "pos": tag,
            "selector": self.selector,
            "model": self.model_name,
            "candidate_count": len(candidates),
        }


def synonym_replacements(
    records: Iterable[EvaluationRecord], generator: SynonymGenerator
) -> Iterator[EvaluationRecord]:
    nltk, _ = generator._nltk()
    for record in records:
        tokens = record.candidate.split()
        tagged = nltk.pos_tag(tokens)
        for position, (word, tag) in enumerate(tagged):
            replacement, metadata = generator.select(
                record.candidate, tokens, position, tag
            )
            realized = replacement is not None
            variant = tokens.copy()
            variant[position] = replacement or word
            yield record.variant(
                name="synonym",
                axis="lexical_variation",
                position=position,
                candidate=" ".join(variant),
                original=word,
                replacement=replacement or word,
                realized=realized,
                metadata={"sequence_length": len(tokens), **metadata},
            )
