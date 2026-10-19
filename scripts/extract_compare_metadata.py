#!/usr/bin/env python3
"""Extract local test-set speaker labels for scripts/build_compare_data.py.

Requires access to the gated CoRal v3 test set. Keep the output under
eval_audio_cache/ or another private location; do not commit it. The published
Compare asset contains only aggregate group counts and scores.

Example:
    python scripts/extract_compare_metadata.py \
      --out eval_audio_cache/compare-data/metadata.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

SOURCES = (
    ("CoRal-project/coral-v3", "conversation/test-", "coral_conversation", "text", "gender", "id_speaker"),
    ("CoRal-project/coral-v3", "read_aloud/test-", "coral_read_aloud", "text", "gender", "id_speaker"),
    ("alexandrainst/ftspeech", "data/test_balanced-", "ftspeech", "sentence", "speaker_gender", "speaker_id"),
)
EXPECTED = {"coral_conversation": 8438, "coral_read_aloud": 9122, "ftspeech": 5534}


def extract(out: Path) -> None:
    api = HfApi()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as file:
        for repo, prefix, dataset, text_col, gender_col, speaker_col in SOURCES:
            info = api.repo_info(repo, repo_type="dataset")
            shards = sorted(s.rfilename for s in info.siblings
                            if s.rfilename.startswith(prefix) and s.rfilename.endswith(".parquet"))
            if not shards:
                raise ValueError(f"No test Parquet shards found for {dataset}")
            count = 0
            columns = [text_col, gender_col, speaker_col]
            if dataset.startswith("coral_"):
                columns.append("dialect")
            for shard in shards:
                frame = pd.read_parquet(f"hf://datasets/{repo}@{info.sha}/{shard}", columns=columns)
                for row in frame.to_dict("records"):
                    reference = (row[text_col] or "").strip()
                    if reference:
                        record = {
                            "dataset": dataset,
                            "id": f"{dataset}_{count:05d}.wav",
                            "reference": reference,
                            "gender": row[gender_col],
                            "dialect": row.get("dialect"),
                            "speaker": row[speaker_col],
                            "source_sha": info.sha,
                        }
                        file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
                file.flush()
                print(f"{dataset}: {count}/{EXPECTED[dataset]}", flush=True)
            file.flush()
            if count != EXPECTED[dataset]:
                raise ValueError(f"{dataset}: expected {EXPECTED[dataset]} rows, got {count}")
            print(f"{dataset}: {count} rows at {info.sha}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    extract(parser.parse_args().out)
