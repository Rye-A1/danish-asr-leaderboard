#!/usr/bin/env python3
"""One-off repair for raw-output parquets corrupted by the NeMo `_text_of` bug.

Before the fix in `nemo_backend._text_of`, a NeMo `Hypothesis` with an empty
`.text` (silence / nothing decoded) was serialised as its full
`Hypothesis(score=…, text='', …)` repr instead of an empty string — both in the
saved raw outputs and in the value that was scored. This script repairs the
already-published parquets in place: every hypothesis that is a `Hypothesis(...)`
repr is replaced with the `text=` it carries (empty for every observed case),
which is exactly what the fixed backend would have produced.

After repair, re-score from the corrected outputs (or just push the corrected
result JSONs) so the leaderboard and the raw outputs agree.

  uv run --no-project --with pandas --with pyarrow --with huggingface_hub \
      python scripts/repair_nemo_outputs.py                 # scan all models
  uv run ... python scripts/repair_nemo_outputs.py --model nvidia/canary-1b-v2
  uv run ... python scripts/repair_nemo_outputs.py --dry-run # report only

Requires HF_TOKEN with write access to the RyeAI org (omit for --dry-run).
"""
from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from danish_asr_leaderboard.results import slugify

DATASET_REPO_ID = "RyeAI/danish-asr-leaderboard"
# Capture the text= field from a NeMo Hypothesis repr, single- or double-quoted.
_HYP_TEXT_RE = re.compile(r"""^Hypothesis\(.*\btext=(?:'([^']*)'|"([^"]*)").*\)$""", re.DOTALL)


def _repair_one(hyp: str) -> str | None:
    """Return the repaired hypothesis, or None if the value wasn't corrupted."""
    if not hyp.startswith("Hypothesis("):
        return None
    m = _HYP_TEXT_RE.match(hyp)
    return (m.group(1) or m.group(2) or "") if m else ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Repair Hypothesis-repr rows in outputs parquets")
    ap.add_argument("--model", default="", help="Only this model id (default: scan all outputs parquets)")
    ap.add_argument("--dry-run", action="store_true", help="Report corrupted rows without uploading")
    args = ap.parse_args()

    import pandas as pd
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(DATASET_REPO_ID, repo_type="dataset")
    parquets = sorted(f for f in files if f.startswith("outputs/") and f.endswith(".parquet"))
    if args.model:
        target = f"outputs/{slugify(args.model)}.parquet"
        parquets = [p for p in parquets if p == target]
        if not parquets:
            print(f"ERROR: {target} not found in dataset repo", file=sys.stderr)
            sys.exit(1)

    total_fixed = 0
    for repo_path in parquets:
        url = f"https://huggingface.co/datasets/{DATASET_REPO_ID}/resolve/main/{repo_path}"
        df = pd.read_parquet(url)
        repaired = df["hypothesis"].astype(str).map(_repair_one)
        n = int(repaired.notna().sum())
        if n == 0:
            continue
        total_fixed += n
        print(f"{repo_path}: {n} corrupted row(s)" + ("  [dry-run]" if args.dry_run else ""))
        if args.dry_run:
            continue
        df.loc[repaired.notna(), "hypothesis"] = repaired[repaired.notna()]
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            df.to_parquet(tmp.name, index=False)
            api.upload_file(
                path_or_fileobj=tmp.name,
                path_in_repo=repo_path,
                repo_id=DATASET_REPO_ID,
                repo_type="dataset",
                commit_message=f"Repair {n} Hypothesis-repr rows in {Path(repo_path).stem}",
            )
        print(f"  uploaded {repo_path}")

    if total_fixed == 0:
        print("No corrupted rows found — nothing to repair.")
    else:
        verb = "would repair" if args.dry_run else "repaired"
        print(f"\nDone: {verb} {total_fixed} row(s).")


if __name__ == "__main__":
    main()
