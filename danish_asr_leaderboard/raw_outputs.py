"""Persist and load per-sample raw model outputs for offline re-scoring.

At eval time the harness writes, for each model, one JSONL file per dataset under
``<outputs_dir>/<model-slug>/<column>.jsonl``. Each line is one sample:

    {"id": "<audio_path>", "reference": "<raw ref>", "hypothesis": "<raw hyp>"}

The text is stored exactly as produced (no normalisation), so ``scripts/rescore.py``
can recompute WER/CER under any normaliser configuration — different Unicode forms,
a future word<->digit converter, etc. — without ever re-running a model.

A small ``meta.json`` alongside the per-dataset files records the model id, params
and run date so a re-scored result JSON can be rebuilt with the same metadata.
"""
from __future__ import annotations

import json
from pathlib import Path

from danish_asr_leaderboard.results import slugify


def outputs_root(outputs_dir: Path | str, model_id: str) -> Path:
    """Directory holding one model's raw outputs."""
    return Path(outputs_dir) / slugify(model_id)


def write_dataset_outputs(
    outputs_dir: Path | str, model_id: str, column: str, records: list[dict]
) -> Path:
    """Write one dataset's raw ``records`` to ``<outputs_dir>/<slug>/<column>.jsonl``."""
    root = outputs_root(outputs_dir, model_id)
    root.mkdir(parents=True, exist_ok=True)
    out_path = root / f"{column}.jsonl"
    with open(out_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out_path


def write_meta(outputs_dir: Path | str, model_id: str, meta: dict) -> Path:
    """Write the run-level metadata sidecar (``meta.json``)."""
    root = outputs_root(outputs_dir, model_id)
    root.mkdir(parents=True, exist_ok=True)
    out_path = root / "meta.json"
    out_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return out_path


def read_dataset_outputs(jsonl_path: Path | str) -> list[dict]:
    """Read a ``<column>.jsonl`` raw-output file back into records."""
    records: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def read_meta(outputs_dir: Path | str, model_id: str) -> dict:
    """Read a model's ``meta.json`` (empty dict if absent)."""
    meta_path = outputs_root(outputs_dir, model_id) / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}
