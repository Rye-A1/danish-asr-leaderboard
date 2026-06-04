#!/usr/bin/env python3
"""Deploy the Gradio app in space/ to the HF Space.

Uploads space/app.py, space/README.md and space/requirements.txt to the Space
repo. The app.py file under space/ is the single source of truth — edit it
there, then run this script to publish.

  python scripts/update_space.py

Requires HF_TOKEN with write access to the RyeAI org:
  export HF_TOKEN=hf_...
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

SPACE_REPO_ID = "RyeAI/danish-asr-leaderboard"
SPACE_DIR = Path(__file__).resolve().parent.parent / "space"

FILES = ["app.py", "README.md", "requirements.txt", "cover.jpeg"]


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: set HF_TOKEN with write access to the Space.", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)
    for name in FILES:
        path = SPACE_DIR / name
        if not path.exists():
            print(f"  skip (missing): {name}")
            continue
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=name,
            repo_id=SPACE_REPO_ID,
            repo_type="space",
            commit_message=f"Update {name}",
        )
        print(f"  uploaded: {name}")

    print(f"Done — Space will restart automatically: https://huggingface.co/spaces/{SPACE_REPO_ID}")


if __name__ == "__main__":
    main()
