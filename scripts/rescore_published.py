#!/usr/bin/env python3
"""Re-score every published result from the saved Hugging Face output Parquets.

Download ``outputs/*.parquet`` from a single revision of the leaderboard dataset,
then run, for example::

    python scripts/rescore_published.py \
      --outputs-dir eval_audio_cache/compare-data/hf/outputs \
      --out-dir results_rescored

Metadata such as submission date and measured speed is copied from ``results/``.
The output directory defaults to ``results_rescored`` so a methodology experiment
never overwrites published scores accidentally.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from danish_asr_leaderboard.datasets import CORE_COLUMNS
from danish_asr_leaderboard.metrics import compute_cer, compute_wer
from danish_asr_leaderboard.normalizer import normalise
from danish_asr_leaderboard.results import mean

NORMALISE = dict(unicode_form="NFKC", number_words=True,
                 spoken_numbers=True, filler_words=True)


def rescore_result(result: dict, frame: pd.DataFrame) -> dict:
    """Keep published metadata and replace all five corpus scores and means."""
    result = dict(result)
    for column in CORE_COLUMNS:
        rows = frame[frame["dataset"] == column]
        if rows.empty:
            raise ValueError(f"missing core dataset {column}")
        refs = [normalise("" if pd.isna(value) else str(value), **NORMALISE)
                for value in rows["reference"]]
        hyps = [normalise("" if pd.isna(value) else str(value), **NORMALISE)
                for value in rows["hypothesis"]]
        result[f"{column}_wer"] = round(compute_wer(refs, hyps), 2)
        result[f"{column}_cer"] = round(compute_cer(refs, hyps), 2)
    result["mean_wer"] = mean([result[f"{column}_wer"] for column in CORE_COLUMNS])
    result["mean_cer"] = mean([result[f"{column}_cer"] for column in CORE_COLUMNS])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, required=True,
                        help="Local snapshot directory containing outputs/*.parquet")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("results_rescored"))
    args = parser.parse_args()
    files = sorted(args.results_dir.glob("*.json"))
    if not files:
        parser.error(f"no published results in {args.results_dir}")
    missing = [path.stem for path in files
               if not (args.outputs_dir / f"{path.stem}.parquet").is_file()]
    if missing:
        parser.error(f"missing output Parquets for: {', '.join(missing)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        published = json.loads(path.read_text(encoding="utf-8"))
        frame = pd.read_parquet(args.outputs_dir / f"{path.stem}.parquet",
                                columns=["dataset", "reference", "hypothesis"])
        updated = rescore_result(published, frame)
        target = args.out_dir / path.name
        target.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
        print(f"{path.stem}: {published['mean_wer']:.2f} → {updated['mean_wer']:.2f}")
    print(f"Wrote {len(files)} results to {args.out_dir}")


if __name__ == "__main__":
    main()
