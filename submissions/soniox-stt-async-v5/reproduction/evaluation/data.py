"""Pinned test splits; fail on missing audio rather than shrink the benchmark."""

import argparse
import json
import os
from itertools import islice
from pathlib import Path

from danish_asr_leaderboard.audio import audio_bytes_to_wav, wav_duration
from danish_asr_leaderboard.datasets import load_common_voice

from evaluation.checkpoints import PINS, sha256


def load_rows(name, cache, max_samples=0, cv_dir=None):
    spec = PINS["datasets"][name]
    if name == "cv17":
        os.environ["CV_DATA_DIR"] = str(cv_dir or cache / "common_voice")
        rows = load_common_voice(cache, max_samples=0)  # Always validate exact split.
        rows = rows[:max_samples] if max_samples else rows
    else:
        root = cache / "audio" / name / spec["revision"]
        root.mkdir(parents=True, exist_ok=True)
        manifest = root / f"manifest-{max_samples or 'full'}.jsonl"
        if manifest.exists():
            rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
        else:
            rows = materialise(name, spec, root, max_samples)
            manifest.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
            )
    if not rows or (not max_samples and len(rows) != spec["expected_rows"]):
        raise ValueError(f"{name}: got {len(rows)} rows; expected {spec['expected_rows']}")
    for i, row in enumerate(rows):
        if not Path(row["audio_path"]).is_file():
            raise FileNotFoundError(row["audio_path"])
        row["id"] = f"{name}:{i}"
        row["audio_sha256"] = sha256(row["audio_path"])
        row["duration"] = wav_duration(row["audio_path"])
        if row["duration"] <= 0:
            raise ValueError(f"Invalid audio: {row['audio_path']}")
    return rows


def materialise(name, spec, root, max_samples):
    from datasets import Audio, load_dataset
    from huggingface_hub import snapshot_download

    if name == "fleurs":
        pattern = "da_dk/test/*.parquet"
        keys = ("transcription", "raw_transcription")
    else:
        pattern = (
            f"{spec['config']}/test-*.parquet" if spec["config"] else "data/test_balanced-*.parquet"
        )
        keys = ("text", "sentence")
    # Cache only the requested test shards; never pull a training split.
    snapshot = snapshot_download(
        spec["repo"],
        repo_type="dataset",
        revision=spec["revision"],
        allow_patterns=[pattern],
        cache_dir=str(root.parents[2] / "hub"),
        max_workers=4,
    )
    files = sorted(str(p) for p in Path(snapshot).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No test shards matched {pattern}")
    ds = load_dataset(
        "parquet", data_files={spec["split"]: files}, split=spec["split"], streaming=True
    )
    ds = ds.cast_column("audio", Audio(decode=False))
    rows = []
    source = islice(ds, max_samples) if max_samples else ds
    for i, entry in enumerate(source):
        text = next((entry[k].strip() for k in keys if entry.get(k)), "")
        if not text:
            raise ValueError(f"{name}:{i}: missing reference")
        path = root / f"{i:06d}.wav"
        if not path.exists() and not audio_bytes_to_wav(entry["audio"], path):
            path.unlink(missing_ok=True)
            raise ValueError(f"{name}:{i}: audio decoding failed")
        rows.append({"audio_path": str(path.resolve()), "reference_text": text})
        if (i + 1) % 100 == 0:
            print(f"{name}: prepared {i + 1} clips", flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=PINS["datasets"], default=list(PINS["datasets"])
    )
    parser.add_argument("--cache", type=Path, default=Path(".cache"))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--cv-dir", type=Path)
    args = parser.parse_args()
    for name in args.datasets:
        rows = load_rows(name, args.cache.resolve(), args.max_samples, args.cv_dir)
        print(f"{name}: {len(rows)} clips, {sum(r['duration'] for r in rows) / 3600:.2f} hours")
