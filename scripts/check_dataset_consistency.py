#!/usr/bin/env python3
"""Verify every published result has raw outputs backing it on the dataset.

Two paths put models on the board: contributors commit ``results/`` +
``outputs/`` and CI publishes them (see CONTRIBUTING.md), while our own eval
runs push outputs to the dataset directly from the machine that produced them.
Both are fine, but nothing compared the two, so a model could carry a published
score with no raw outputs at all.

That is not hypothetical: ``ordbogen/whisper`` was published in August 2026 and
sat for months with a score, no ``outputs/ordbogen__whisper.parquet``, and
therefore no confidence interval — because ``compute_ci.py`` derives intervals
by enumerating exactly those parquets. Nothing failed; it was simply absent.

This turns that silence into a build failure.

Usage:
    python scripts/check_dataset_consistency.py            # fail on missing outputs
    python scripts/check_dataset_consistency.py --warn-only
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

DATASET_REPO_ID = "RyeAI/danish-asr-leaderboard"
RESULTS_DIR = Path("results")


def result_slugs() -> set[str]:
    # Skip macOS AppleDouble sidecars ("._name.json"), which scp from a Mac
    # leaves behind and which are not results.
    return {p.stem for p in RESULTS_DIR.glob("*.json")
            if not p.name.startswith("._")}


def dataset_output_slugs() -> set[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(DATASET_REPO_ID, repo_type="dataset")
    return {f[len("outputs/"):-len(".parquet")]
            for f in files
            if f.startswith("outputs/") and f.endswith(".parquet")}


def published_ci_slugs() -> set[str]:
    url = f"https://huggingface.co/datasets/{DATASET_REPO_ID}/resolve/main/data/ci.json"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
        return set(data) if isinstance(data, dict) else set()
    except Exception:
        return set()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--warn-only", action="store_true",
                    help="Report problems but exit 0")
    args = ap.parse_args()

    results = result_slugs()
    outputs = dataset_output_slugs()
    cis = published_ci_slugs()
    if not results:
        print("ERROR: no results/*.json found — wrong working directory?", file=sys.stderr)
        sys.exit(1)

    missing_outputs = sorted(results - outputs)
    orphan_outputs = sorted(outputs - results)
    # CIs are regenerated on every deploy, so a gap here is informational: it
    # means the next deploy still has work to do, not that anything is broken.
    missing_ci = sorted(results & outputs - cis)

    print(f"results: {len(results)} | dataset outputs: {len(outputs)} | "
          f"published CIs: {len(cis)}")

    if orphan_outputs:
        print(f"\nnote: {len(orphan_outputs)} outputs with no result "
              f"(superseded runs are fine): {', '.join(orphan_outputs)}")
    if missing_ci:
        print(f"\nnote: {len(missing_ci)} model(s) awaiting a confidence interval "
              f"(generated on deploy): {', '.join(missing_ci)}")

    if missing_outputs:
        print(f"\nFAIL: {len(missing_outputs)} published result(s) have no raw "
              f"outputs on the dataset:", file=sys.stderr)
        for slug in missing_outputs:
            print(f"  - {slug}", file=sys.stderr)
        print("\nPush them from the machine that produced them:\n"
              "  python scripts/push_outputs.py --model <slug>\n"
              "or commit outputs/<slug>/ so CI publishes them (CONTRIBUTING.md).",
              file=sys.stderr)
        sys.exit(0 if args.warn_only else 1)

    print("\nOK: every published result has raw outputs on the dataset.")


if __name__ == "__main__":
    main()
