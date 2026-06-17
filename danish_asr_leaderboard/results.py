"""Leaderboard result schema, parameter-count detection, and JSON output.

The JSON schema written here is the contract consumed by ``scripts/push_results.py``
(which builds ``data/results.parquet``) and by the leaderboard Space. Keep field
names stable.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any


@dataclass
class EvalResult:
    model: str            # markdown link: [org/name](https://huggingface.co/org/name)
    params_b: float
    coral_conversation_wer: float | None
    coral_read_aloud_wer: float | None
    cv17_da_wer: float | None
    fleurs_da_wer: float | None
    ftspeech_wer: float | None
    mean_wer: float       # macro-average over the core-5 datasets
    coral_conversation_cer: float | None
    coral_read_aloud_cer: float | None
    cv17_da_cer: float | None
    fleurs_da_cer: float | None
    ftspeech_cer: float | None
    mean_cer: float       # macro-average over the core-5 datasets
    speed_x: float | None  # audio_duration / inference_time (higher = faster)
    submitted: str        # ISO date
    access: str = "open"  # "open" | "proprietary"


def mean(values: list[float | None]) -> float:
    """Macro-average of the non-null values, rounded to 2dp (0.0 if all null)."""
    valid = [v for v in values if v is not None]
    if not valid:
        return 0.0
    return round(sum(valid) / len(valid), 2)


def model_link(model_id: str) -> str:
    """Wrap an HF model id in a markdown link; leave local paths untouched."""
    if model_id.startswith("/") or Path(model_id).exists():
        return model_id
    clean = model_id.rstrip("/")
    return f"[{clean}](https://huggingface.co/{clean})"


def fetch_params_b(model_id: str, model: Any | None = None) -> float | None:
    """Best-effort parameter count in billions.

    1. For HF-hosted models, read safetensors metadata (no weight download).
    2. Otherwise count parameters on the loaded model object.
    Returns None if neither works.
    """
    if not Path(model_id).exists():
        try:
            from huggingface_hub import model_info as hf_model_info

            info = hf_model_info(model_id)
            if info.safetensors and info.safetensors.total:
                return round(info.safetensors.total / 1e9, 3)
        except Exception:
            pass
    if model is not None:
        try:
            if hasattr(model, "parameters"):
                return round(sum(p.numel() for p in model.parameters()) / 1e9, 3)
            if hasattr(model, "model") and hasattr(model.model, "parameters"):
                return round(sum(p.numel() for p in model.model.parameters()) / 1e9, 3)
        except Exception:
            pass
    return None


def slugify(model_id: str) -> str:
    """Filesystem-safe slug for a result filename."""
    return re.sub(r"[^a-zA-Z0-9_-]", "__", model_id.strip("/"))


def write_result_json(result: EvalResult, out_dir: Path, model_id: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slugify(model_id)}.json"
    out_path.write_text(
        json.dumps(asdict(result), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return out_path


def today_iso() -> str:
    return date.today().isoformat()
