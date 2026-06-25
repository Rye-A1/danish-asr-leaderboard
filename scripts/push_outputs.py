#!/usr/bin/env python3
"""Convert saved raw model outputs to per-model Parquet and push to HuggingFace.

Supports two input layouts:

1. Eval-harness layout -- ``outputs/<model-slug>/`` directory with one JSONL per
   dataset (``{id, reference, hypothesis}`` per line) plus an optional ``meta.json``.

2. PR-submission layout -- ``outputs/<model-slug>.jsonl`` single combined file with
   ``{dataset, id, reference, hypothesis}`` per line (submitted alongside the result
   JSON when opening a pull request).

Both are converted to a single Parquet -- ``outputs/<model-slug>.parquet`` with
columns ``dataset, id, reference, hypothesis`` -- and uploaded to the HF dataset
repo under the ``outputs/`` prefix.

One Parquet per model keeps pushes incremental. The ``dataset`` column makes each
file self-describing so the HF viewer can filter by test set.

Run after one or more evals / PR merges:
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


def _iter_combined_jsonl(jsonl_path: Path):
    """Yield rows from a combined submission JSONL (has 'dataset' field)."""
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        yield {
            "dataset": rec.get("dataset") or rec.get("set"),
            "id": rec.get("id") or rec.get("audio"),
            "reference": rec.get("reference"),
            "hypothesis": rec.get("hypothesis"),
        }


def build_parquet(model_dir: Path) -> tuple[Path, int]:
    """Roll one model's per-dataset JSONL directory into a single Parquet."""
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


def build_parquet_from_combined(jsonl_path: Path) -> tuple[Path, int]:
    """Build a Parquet from a single combined submission JSONL file."""
    import pandas as pd

    rows = list(_iter_combined_jsonl(jsonl_path))
    if not rows:
        return Path(), 0
    parquet_path = jsonl_path.parent / f"{jsonl_path.stem}.parquet"
    pd.DataFrame(rows, columns=COLUMNS).to_parquet(str(parquet_path), index=False)
    return parquet_path, len(rows)


def model_label(model_dir: Path) -> str:
    meta = model_dir / "meta.json"
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8")).get("model", model_dir.name)
    return model_dir.name


def collect_models(outputs_dir: Path, model: str) -> list[tuple[Path, str]]:
    """Return (source_path, label) pairs -- source is either a dir or a .jsonl file."""
    if model:
        slug = slugify(model)
        combined = outputs_dir / f"{slug}.jsonl"
        d = outputs_dir / slug
        if combined.is_file():
            return [(combined, slug)]
        if d.is_dir():
            return [(d, model_label(d))]
        print(f"ERROR: no raw outputs for {model!r} at {combined} or {d}", file=sys.stderr)
        sys.exit(1)

    sources: list[tuple[Path, str]] = []
    for p in sorted(outputs_dir.glob("*.jsonl")):
        sources.append((p, p.stem))
    for d in sorted(outputs_dir.iterdir()):
        if d.is_dir():
            sources.append((d, model_label(d)))
    if not sources:
        print(f"ERROR: no model outputs under {outputs_dir}", file=sys.stderr)
        sys.exit(1)
    return sources


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

    sources = collect_models(outputs_dir, args.model)
    built: list[tuple[Path, str, int]] = []
    for source, label in sources:
        if source.is_file():
            parquet_path, n = build_parquet_from_combined(source)
        else:
            parquet_path, n = build_parquet(source)
        if n == 0:
            print(f"  SKIP (no samples): {label}", file=sys.stderr)
            continue
        print(f"  built {parquet_path}  ({n} rows)")
        built.append((parquet_path, label, n))

    if not built:
        print("Nothing to push.", file=sys.stderr)
        sys.exit(1)

    if args.no_upload:
        print(f"\n--no-upload set; built {len(built)} Parquet file(s), skipping HF upload.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    print(f"\nPushing {len(built)} model output file(s) to {DATASET_REPO_ID} ...")
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
