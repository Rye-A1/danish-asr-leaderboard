"""Evaluate Soniox on the same pinned Danish splits; defaults to a four-clip smoke test."""

import argparse
import importlib.metadata
import json
import platform
import subprocess
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from evaluation.checkpoints import PINS, sha256
from evaluation.data import load_rows
from evaluation.results import (
    NORMALIZATION,
    core_mean,
    fingerprint,
    resume,
    score,
    write_json,
    write_overview,
)
from evaluation.soniox import Soniox


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="stt-async-v5", choices=["stt-async-v5"])
    parser.add_argument("--region", choices=["us", "eu"], default="us")
    parser.add_argument("--datasets", nargs="+", choices=PINS["datasets"], default=["fleurs"])
    parser.add_argument("--max-samples", type=int, default=4, help="Clips per set; 0 = full")
    parser.add_argument("--timeout", type=float, default=600, help="Polling timeout per clip")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent independent clips")
    parser.add_argument("--cache", type=Path, default=Path(".cache"))
    parser.add_argument("--cv-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("evaluation/results"))
    args = parser.parse_args()
    if args.max_samples < 0 or args.timeout <= 0 or args.workers < 1:
        parser.error("Sample limit must be nonnegative; timeout and workers must be positive")
    return args


def unchanged(path, value):
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
        raise ValueError(f"Run settings or data changed: use a different --output ({path})")
    write_json(path, value)


def main(args=None):
    args = args or arguments()
    backend = Soniox(args.model, args.region, args.timeout)  # Fail before data prep without key.
    slug = f"soniox-{args.model}-{args.region}"
    folder = args.output / slug / (f"smoke-{args.max_samples}" if args.max_samples else "full")
    folder.mkdir(parents=True, exist_ok=True)
    workers = getattr(args, "workers", 1)
    baseline_path = Path(__file__).with_name("benchmark.json")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    config = {
        "model": {
            "provider": "soniox",
            "id": args.model,
            "revision": None,
            "revision_note": "Hosted API; immutable weight revision unavailable",
        },
        "base_url": backend.base_url,
        "request_settings": backend.settings,
        "translation": None,
        "context": None,
        "harness_revision": PINS["harness"],
        "normalization": NORMALIZATION,
        "max_samples": args.max_samples,
        "workers": workers,
        "baseline_sha256": sha256(baseline_path),
        "timing": "API upload/create/poll/download wall time; excludes local data preparation",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                p: importlib.metadata.version(p) for p in ["danish-asr-leaderboard", "datasets"]
            },
            "ffmpeg": subprocess.check_output(["ffmpeg", "-version"]).decode().splitlines()[0],
        },
        "code_sha256": {
            name: sha256(Path(__file__).with_name(name))
            for name in ["soniox.py", "soniox_evaluation.py", "data.py", "results.py"]
        },
    }
    unchanged(folder / "config.json", config)
    summary_path = folder / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    scores = summary.get("datasets", {})
    for name in dict.fromkeys(args.datasets):
        rows = load_rows(name, args.cache.resolve(), args.max_samples, args.cv_dir)
        dataset_info = {
            **PINS["datasets"][name],
            "samples": len(rows),
            "manifest_sha256": fingerprint(rows),
        }
        if not args.max_samples and dataset_info != baseline["datasets"][name]:
            raise ValueError(f"{name}: full test data differs from audited baseline")
        unchanged(folder / f"{name}.data.json", dataset_info)
        path = folder / f"{name}.jsonl"
        records = resume(path, rows)
        print(f"{name}: {len(rows)} clips, {len(records)} already saved", flush=True)
        with path.open("a", encoding="utf-8") as stream, ThreadPoolExecutor(workers) as pool:

            def submit(index):
                row = rows[index]
                return pool.submit(
                    backend.transcribe, row["audio_path"], folder / "jobs" / name / f"{index}.json"
                )

            start = len(records)
            next_index = min(start + workers, len(rows))
            pending = deque(submit(index) for index in range(start, next_index))
            while pending:
                # Refill immediately instead of waiting for a whole group to finish.
                # Keep a bounded window and write in original dataset order.
                row = rows[len(records)]
                job = pending.popleft().result()
                record = {
                    "id": row["id"],
                    "reference": row["reference_text"],
                    "hypothesis": job["transcript"]["text"],
                    "audio_seconds": row["duration"],
                    "batch_seconds_per_clip": job["elapsed_seconds"],
                    "transcription_id": job["transcription_id"],
                    "requested_at": job["started_at"],
                    "completed_at": job["completed_at"],
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                records.append(record)
                if next_index < len(rows):
                    pending.append(submit(next_index))
                    next_index += 1
                if len(records) % 32 == 0 or len(records) == len(rows):
                    print(f"{name}: {len(records)}/{len(rows)}", flush=True)
        metrics = score(records, len(rows))
        metrics["audio_seconds"] = sum(r["audio_seconds"] for r in records)
        metrics["inference_seconds"] = sum(r["batch_seconds_per_clip"] for r in records)
        metrics["rtf"] = metrics["inference_seconds"] / metrics["audio_seconds"]
        scores[name] = metrics
        write_json(
            summary_path,
            {
                "model": config["model"],
                "datasets": scores,
                "core5_mean": core_mean(scores, PINS["datasets"], args.max_samples > 0),
                "smoke_test": args.max_samples > 0,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        if not args.max_samples:
            write_overview(args.output)
        print(f"{name}: WER {metrics['wer']:.2f}% / CER {metrics['cer']:.2f}%", flush=True)
    backend.close()
    return folder


if __name__ == "__main__":
    main()
