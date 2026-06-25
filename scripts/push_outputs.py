#!/usr/bin/env python3
"""Convert saved raw model outputs to per-model Parquet and push to HuggingFace.

The eval harness writes raw, un-normalised output to
``outputs/<model-slug>/<dataset>.jsonl`` (one ``{id, reference, hypothesis}`` per
line) plus a ``meta.json``. This script rolls each model's per-dataset JSONL files
into a single Parquet — ``outputs/<model-slug>.parquet`` with columns
``dataset, id, reference, hypothesis`` — and uploads it to the dataset repo under
the ``outputs/`` prefix, mirroring how ``push_results.py`` publishes the scores.

One Parquet *per model* (rather than one combined file) keeps each push
incremental: re-run one model, push only that model's outputs. The ``dataset``
column makes each file self-describing, so the HF viewer can filter by test set.

Run after one or more evals have produced ``outputs/<slug>/``:
  python scripts/push_outputs.py                 # all models under outputs/
  python scripts/push_outputs.py --model openai/whisper-large-v3

Requires:
  uv run --with pandas --with pyarrow --with huggingface_hub python scripts/push_outputs.py
  # or: pip install pandas pyarrow huggingface_hub && python scripts/push_outputs.py
  huggingface-cli login  (with RyeAI org write token)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from danish_asr_leaderboard.raw_outputs import read_dataset_outputs
from danish_asr_leaderboard.results import slugify

DATASET_REPO_ID = "RyeAI/danish-asr-leaderboard"
COLUMNS = ["dataset", "id", "reference", "hypothesis"]


def model_dirs(outputs_dir: Path, model: str) -> list[Path]:
    if model:
        d = outputs_dir / slugify(model)
        if not d.is_dir():
            print(f"ERROR: no raw outputs for {model!r} at {d}", file=sys.stderr)
            sys.exit(1)
        return [d]
    dirs = sorted(d for d in outputs_dir.iterdir() if d.is_dir())
    if not dirs:
        print(f"ERROR: no model output dirs under {outputs_dir}", file=sys.stderr)
        sys.exit(1)
    return dirs


def build_parquet(model_dir: Path) -> tuple[Path, int]:
    """Roll one model's per-dataset JSONL files into a single Parquet."""
    import pandas as pd

    rows: list[dict] = []
    for jsonl_path in sorted(model_dir.glob("*.jsonl")):
        dataset = jsonl_path.stem
        for rec in read_dataset_outputs(jsonl_path):
            rows.append({
                "dataset": dataset,
                "id": rec.get("id"),
                "reference": rec.get("reference"),
                "hypothesis": rec.get("hypothesis"),
            })
    if not rows:
        return Path(), 0
    parquet_path = model_dir.parent / f"{model_dir.name}.parquet"
    pd.DataFrame(rows, columns=COLUMNS).to_parquet(str(parquet_path), index=False)
    return parquet_path, len(rows)


def model_label(model_dir: Path) -> str:
    meta = model_dir / "meta.json"
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8")).get("model", model_dir.name)
    return model_dir.name


def main() -> None:
    ap = argparse.ArgumentParser(description="Push raw model outputs to HF as per-model Parquet")
    ap.add_argument("--outputs-dir", default="outputs")
    ap.add_argument("--model", default="", help="Only this model id (default: all under outputs-dir)")
    ap.add_argument("--no-upload", action="store_true",
                    help="Build the Parquet files locally but skip the HF upload")
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.is_dir():
        print(f"ERROR: outputs dir not found: {outputs_dir}", file=sys.stderr)
        sys.exit(1)

    dirs = model_dirs(outputs_dir, args.model)
    built: list[tuple[Path, str, int]] = []
    for d in dirs:
        parquet_path, n = build_parquet(d)
        if n == 0:
            print(f"  SKIP (no samples): {d.name}", file=sys.stderr)
            continue
        print(f"  built {parquet_path}  ({n} rows)")
        built.append((parquet_path, model_label(d), n))

    if not built:
        print("Nothing to push.", file=sys.stderr)
        sys.exit(1)

    if args.no_upload:
        print(f"\n--no-upload set; built {len(built)} Parquet file(s), skipping HF upload.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    print(f"\nPushing {len(built)} model output file(s) to {DATASET_REPO_ID} …")
    for parquet_path, label, n in built:
        api.upload_file(
            path_or_fileobj=str(parquet_path),
            path_in_repo=f"outputs/{parquet_path.name}",
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            commit_message=f"Update raw outputs: {label} ({n} samples)",
        )
        print(f"  uploaded outputs/{parquet_path.name}")

    print(f"\nDone. View at: https://huggingface.co/datasets/{DATASET_REPO_ID}/tree/main/outputs")


if __name__ == "__main__":
    main()
