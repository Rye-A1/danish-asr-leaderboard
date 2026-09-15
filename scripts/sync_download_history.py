#!/usr/bin/env python3
"""Pull or publish the daily Hugging Face download history in the results dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASET_REPO_ID = "RyeAI/danish-asr-leaderboard"
REMOTE_PATH = "data/hf_downloads.json"
DEFAULT_HISTORY_PATH = Path(__file__).resolve().parent.parent / "history" / "hf_downloads.json"


def snapshot_date(path: Path) -> str:
    """Return the newest snapshot date for an upload commit message."""
    history = json.loads(path.read_text(encoding="utf-8"))
    snapshots = history.get("snapshots", [])
    return snapshots[-1].get("date", "unknown") if snapshots else "unknown"


def pull_history(path: Path) -> None:
    """Replace the local seed with the published dataset history when present."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    try:
        downloaded = Path(
            hf_hub_download(
                repo_id=DATASET_REPO_ID,
                filename=REMOTE_PATH,
                repo_type="dataset",
            )
        )
    except EntryNotFoundError:
        print("No published download history yet; using the repository seed.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(downloaded.read_bytes())
    print(f"Restored download history through {snapshot_date(path)}.")


def push_history(path: Path) -> None:
    """Publish the local history without writing to protected GitHub main."""
    from huggingface_hub import HfApi

    HfApi().upload_file(
        path_or_fileobj=str(path),
        path_in_repo=REMOTE_PATH,
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        commit_message=f"Snapshot Hugging Face downloads ({snapshot_date(path)})",
    )
    print(f"Published download history through {snapshot_date(path)}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("pull", "push"))
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY_PATH)
    args = parser.parse_args()
    if args.operation == "pull":
        pull_history(args.history)
    else:
        push_history(args.history)


if __name__ == "__main__":
    main()