"""Named local CSV datasets configured in custom_datasets.py."""
from __future__ import annotations

import runpy
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    """A local ASR test split; paths are relative to custom_datasets.py."""

    name: str
    source: dict[str, str]


def load_config(name: str) -> tuple[DatasetConfig, Path]:
    """Find a named config in the current directory without contacting the Hub."""
    path = Path("custom_datasets.py").resolve()
    if not path.is_file():
        raise ValueError(f"Unknown dataset {name!r}; define it in {path}")
    configs = [value for value in runpy.run_path(str(path)).values()
               if isinstance(value, DatasetConfig) and value.name == name]
    if len(configs) != 1:
        raise ValueError(f"Expected exactly one DatasetConfig named {name!r} in {path}")
    config = configs[0]
    if not isinstance(config.source, dict) or not isinstance(config.source.get("test"), str):
        raise ValueError(f"{name}: source must contain a 'test' CSV path")
    source = (path.parent / config.source["test"]).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{name}: test CSV not found: {source}")
    return config, source
