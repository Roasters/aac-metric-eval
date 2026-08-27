"""Configuration loading with explicit repository-relative path resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class ConfigError(ValueError):
    """Raised when a release configuration is missing or malformed."""


@dataclass(frozen=True)
class Settings:
    """Parsed YAML configuration and its repository root."""

    values: Mapping[str, Any]
    config_path: Path
    repository_root: Path

    @classmethod
    def load(cls, path: str | Path) -> "Settings":
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - installation error
            raise ConfigError("PyYAML is required: pip install PyYAML") from exc

        config_path = Path(path).expanduser().resolve()
        if not config_path.is_file():
            raise ConfigError(f"Configuration file does not exist: {config_path}")
        with config_path.open(encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        if not isinstance(values, dict):
            raise ConfigError("The configuration root must be a mapping.")

        configured_root = values.get("repository_root")
        if configured_root:
            root = Path(os.path.expandvars(str(configured_root))).expanduser()
            if not root.is_absolute():
                root = config_path.parent / root
            root = root.resolve()
        else:
            root = config_path.parent.parent.resolve()
        return cls(values=values, config_path=config_path, repository_root=root)

    def section(self, name: str) -> Mapping[str, Any]:
        value = self.values.get(name, {})
        if not isinstance(value, dict):
            raise ConfigError(f"Configuration section '{name}' must be a mapping.")
        return value

    def dataset(self, name: str) -> Mapping[str, Any]:
        datasets = self.section("datasets")
        value = datasets.get(name)
        if not isinstance(value, dict):
            raise ConfigError(f"Dataset '{name}' is not configured.")
        return value

    def resolve_path(
        self,
        value: str | Path,
        *,
        env: str | None = None,
        base: str | Path | None = None,
    ) -> Path:
        """Resolve a configured path, preferring an optional environment variable."""

        raw = os.environ.get(env) if env else None
        path = Path(os.path.expandvars(raw or str(value))).expanduser()
        if path.is_absolute():
            return path.resolve()
        anchor = self.repository_root if base is None else self.resolve_path(base)
        return (anchor / path).resolve()

    @property
    def data_dir(self) -> Path:
        value = self.section("paths").get("data_dir", "data")
        return self.resolve_path(value)

    @property
    def results_dir(self) -> Path:
        override = os.environ.get("RESULTS_DIR") or os.environ.get("CAF_RESULTS_DIR")
        value = override or self.section("paths").get("results_dir", "results")
        return self.resolve_path(value)

    @property
    def perturbation_dir(self) -> Path:
        return self.results_dir / "perturbation_pairs"

    @property
    def score_dir(self) -> Path:
        return self.results_dir / "cached_scores"

    @property
    def table_dir(self) -> Path:
        return self.results_dir / "tables"

    @property
    def caf_source(self) -> Path:
        section = self.section("caf")
        value = os.environ.get("CAF_SRC") or section.get(
            "source_dir", "third_party/CAF-Score"
        )
        return self.resolve_path(value)

    def dataset_root(self, name: str) -> Path:
        section = self.dataset(name)
        return self.resolve_path(
            section.get("root", f"data/{name}"), env=section.get("root_env")
        )

    def dataset_path(self, name: str, key: str) -> Path:
        section = self.dataset(name)
        if key not in section:
            raise ConfigError(f"Dataset '{name}' has no '{key}' path.")
        value = str(section[key])
        if value.startswith("repo:"):
            return self.resolve_path(value.removeprefix("repo:"))
        return self.resolve_path(value, base=self.dataset_root(name))

    def ensure_output_dirs(self) -> None:
        for path in (self.perturbation_dir, self.score_dir, self.table_dir):
            path.mkdir(parents=True, exist_ok=True)
