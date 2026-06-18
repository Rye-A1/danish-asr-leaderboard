#!/usr/bin/env python3
"""Rebuild data/results.parquet from all JSONs in results/ and push to HuggingFace.

Pulls existing result JSONs from the HF dataset repo first, merges with any
local JSONs in results/ (local takes precedence), then rebuilds the parquet
and uploads.  Safe to run on a fresh clone — no previous results are lost.

Run after run_eval.py has written one or more result JSONs:
  python scripts/push_results.py

Requires:
  uv run --with pandas --with pyarrow --with huggingface_hub python scripts/push_results.py
  # or: pip install pandas pyarrow huggingface_hub && python scripts/push_results.py
  huggingface-cli login  (with RyeAI org write token)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DATASET_REPO_ID = "RyeAI/danish-asr-leaderboard"
RESULTS_DIR = Path("results")
PARQUET_PATH = Path("data/results.parquet")

# Column order for the leaderboard table
COLUMNS = [
    "model",
    "access",
    "params_b",
    "mean_wer",
    "mean_cer",
    "speed_x",
    "coral_conversation_wer",
    "coral_read_aloud_wer",
    "cv17_da_wer",
    "fleurs_da_wer",
    "ftspeech_wer",
    "coral_conversation_cer",
    "coral_read_aloud_cer",
    "cv17_da_cer",
    "fleurs_da_cer",
    "ftspeech_cer",
    "submitted",
]


def pull_remote_results() -> dict[str, dict]:
    """Download all result JSONs from the HF dataset repo. Returns {slug: data}."""
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    remote: dict[str, dict] = {}
    try:
        api = HfApi()
        files = api.list_repo_files(DATASET_REPO_ID, repo_type="dataset")
        json_files = [f for f in files if f.startswith("results/") and f.endswith(".json")]
        if json_files:
            print(f"Pulling {len(json_files)} existing result(s) from {DATASET_REPO_ID}...")
        for repo_path in json_files:
            try:
                local = hf_hub_download(
                    repo_id=DATASET_REPO_ID,
                    filename=repo_path,
                    repo_type="dataset",
                )
                data = json.loads(Path(local).read_text(encoding="utf-8"))
                slug = Path(repo_path).name
                remote[slug] = data
                print(f"  pulled: {slug}")
            except Exception as exc:
                print(f"  WARNING: could not pull {repo_path}: {exc}", file=sys.stderr)
    except (EntryNotFoundError, RepositoryNotFoundError):
        pass  # dataset repo doesn't exist yet — first push
    except Exception as exc:
        print(f"WARNING: could not list remote results: {exc}", file=sys.stderr)
    return remote


def load_results() -> list[dict]:
    # Pull remote first, then overlay local (local takes precedence for same slug)
    remote = pull_remote_results()

    local_files = {p.name: p for p in sorted(RESULTS_DIR.glob("*.json"))}

    merged: dict[str, dict] = {**remote, **{name: json.loads(p.read_text(encoding="utf-8"))
                                             for name, p in local_files.items()}}

    if not merged:
        print("No results found locally or remotely.", file=sys.stderr)
        sys.exit(1)

    rows: list[dict] = []
    print(f"\nBuilding parquet from {len(merged)} result(s) ({len(local_files)} local, "
          f"{len(remote) - len(set(remote) & set(local_files))} remote-only)...")
    for name, data in sorted(merged.items()):
        try:
            row = {col: data.get(col) for col in COLUMNS}
            if not row.get("access"):
                row["access"] = "open"  # backwards compat for results without access field
            rows.append(row)
            src = "local" if name in local_files else "remote"
            print(f"  OK [{src}]: {name}")
        except Exception as exc:
            print(f"  WARNING: skipping {name}: {exc}", file=sys.stderr)

    return rows


def build_parquet(rows: list[dict]) -> None:
    import pandas as pd

    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=COLUMNS)

    # Sort by mean_wer ascending (best model first), nulls last
    df = df.sort_values("mean_wer", ascending=True, na_position="last")
    df.to_parquet(str(PARQUET_PATH), index=False)
    print(f"\nWrote {PARQUET_PATH} ({len(df)} rows)")


def push_to_hub(rows: list[dict]) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    n = len(rows)
    print(f"\nPushing to {DATASET_REPO_ID} ({n} model(s))...")

    # Upload the parquet
    api.upload_file(
        path_or_fileobj=str(PARQUET_PATH),
        path_in_repo="data/results.parquet",
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        commit_message=f"Update leaderboard results ({n} models)",
    )

    # Upload any new/updated local JSONs so they're retrievable by future runs
    local_files = sorted(RESULTS_DIR.glob("*.json"))
    for path in local_files:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"results/{path.name}",
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            commit_message=f"Update result: {path.stem}",
        )

    print(f"Done. View at: https://huggingface.co/datasets/{DATASET_REPO_ID}")


def main() -> None:
    rows = load_results()
    build_parquet(rows)
    push_to_hub(rows)


if __name__ == "__main__":
    main()
