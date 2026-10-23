"""Re-audit Soniox transcripts against the baseline before exporting benchmark scores."""

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from evaluation.checkpoints import PINS, sha256
from evaluation.data import load_rows
from evaluation.results import NORMALIZATION, core_mean, fingerprint, resume, score, write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(folder, cache, cv_dir=None, allow_partial=False):
    config = read(folder / "config.json")
    summary = read(folder / "summary.json")
    baseline_path = Path(__file__).with_name("benchmark.json")
    baseline = read(baseline_path)
    require(config["max_samples"] == 0 and not summary["smoke_test"], "Cannot export smoke tests")
    require(summary["model"] == config["model"], "Model metadata differs")
    require(config["model"]["id"] == "stt-async-v5", "Unexpected model")
    require(config["harness_revision"] == PINS["harness"], "Wrong Rye revision")
    require(config["normalization"] == NORMALIZATION, "Wrong normalization")
    require(config["baseline_sha256"] == sha256(baseline_path), "Audited baseline changed")
    require(config["translation"] is None and config["context"] is None, "Unexpected prompting")
    require(
        config["request_settings"]
        == {
            "model": "stt-async-v5",
            "language_hints": ["da"],
            "language_hints_strict": False,
            "enable_speaker_diarization": False,
            "enable_language_identification": False,
        },
        "Unexpected API settings",
    )
    for name, digest in config["code_sha256"].items():
        require(sha256(folder / "source" / name) == digest, f"Source snapshot changed: {name}")
    history_path = folder / "execution-history.json"
    history = read(history_path) if history_path.exists() else []
    for event in history:
        archive = folder / event["previous_config_file"]
        require(sha256(archive) == event["previous_config_sha256"], "Execution archive changed")
        previous = read(archive)
        require(previous["workers"] == event["previous_workers"], "Worker history differs")
        for name, digest in previous["code_sha256"].items():
            source_dir = folder / event.get("previous_source_dir", "source")
            require(sha256(source_dir / name) == digest, f"Archived source changed: {name}")
        for name in ("data.py", "results.py"):
            require(
                previous["code_sha256"][name] == config["code_sha256"][name],
                f"Data/scoring implementation changed: {name}",
            )
        require(
            {k: v for k, v in previous.items() if k not in ("workers", "code_sha256")}
            == {k: v for k, v in config.items() if k not in ("workers", "code_sha256")},
            "Recognition settings changed across execution phases",
        )
    available = summary["datasets"]
    complete = set(available) == set(PINS["datasets"])
    require(available and set(available) <= set(PINS["datasets"]), "Unexpected test sets")
    require(complete or allow_partial, "Incomplete benchmark: all five complete sets required")
    scores, datasets, transcript_hashes, job_hashes, dates = {}, {}, {}, {}, []
    for name in PINS["datasets"]:
        if name not in available:
            continue
        rows = load_rows(name, cache, cv_dir=cv_dir)
        data_info = {
            **PINS["datasets"][name],
            "samples": len(rows),
            "manifest_sha256": fingerprint(rows),
        }
        require(data_info == baseline["datasets"][name], f"{name}: baseline data differs")
        require(read(folder / f"{name}.data.json") == data_info, f"{name}: run data differs")
        path = folder / f"{name}.jsonl"
        records = resume(path, rows)
        metrics = score(records, PINS["datasets"][name]["expected_rows"])
        job_digest = hashlib.sha256()
        for index, (record, row) in enumerate(zip(records, rows)):
            label = f"{name}:{index}"
            job_path = folder / "jobs" / name / f"{index}.json"
            job = read(job_path)
            require(job["audio_sha256"] == row["audio_sha256"], f"{label}: wrong audio")
            require(job["settings"] == config["request_settings"], f"{label}: wrong settings")
            require(job["base_url"] == config["base_url"], f"{label}: wrong endpoint")
            require(
                not job.get("pending_request") and job.get("remote_deleted"),
                f"{label}: unfinished request or cleanup",
            )
            status = job["status_response"]
            require(
                status["status"] == "completed" and status["model"] == config["model"]["id"],
                f"{label}: job not completed by expected model",
            )
            require(status.get("translation") is None, f"{label}: unexpected translation")
            for field in (
                "language_hints",
                "enable_speaker_diarization",
                "enable_language_identification",
            ):
                require(
                    status[field] == config["request_settings"][field],
                    f"{label}: server settings differ for {field}",
                )
            require(
                job["transcription_id"] == record["transcription_id"] == status["id"],
                f"{label}: job ID mismatch",
            )
            require(job["transcript"]["text"] == record["hypothesis"], f"{label}: text mismatch")
            require(
                math.isclose(record["audio_seconds"], row["duration"], abs_tol=1e-6),
                f"{label}: duration mismatch",
            )
            seconds = record["batch_seconds_per_clip"]
            require(
                math.isfinite(seconds) and seconds > 0 and seconds == job["elapsed_seconds"],
                f"{label}: invalid timing",
            )
            for field in ("requested_at", "completed_at"):
                require(
                    datetime.fromisoformat(record[field]).tzinfo is not None,
                    f"{label}: missing timestamp zone",
                )
            require(
                record["requested_at"] == job["started_at"]
                and record["completed_at"] == job["completed_at"],
                f"{label}: timestamp mismatch",
            )
            dates.append((record["requested_at"], record["completed_at"]))
            job_digest.update(f"{index}:{sha256(job_path)}\n".encode())
        metrics["audio_seconds"] = sum(r["audio_seconds"] for r in records)
        metrics["inference_seconds"] = sum(r["batch_seconds_per_clip"] for r in records)
        metrics["rtf"] = metrics["inference_seconds"] / metrics["audio_seconds"]
        require(metrics == available[name], f"{name}: summary differs from re-scored transcripts")
        scores[name], datasets[name] = metrics, data_info
        transcript_hashes[name], job_hashes[name] = sha256(path), job_digest.hexdigest()
        print(f"Audited {name}: {len(records)} clips, WER {metrics['wer']:.4f}%", flush=True)
    average = core_mean(scores, PINS["datasets"], False)
    saved_average = summary["core5_mean"]
    require(
        (average is None and saved_average is None)
        or (
            average is not None
            and saved_average is not None
            and all(
                math.isclose(average[k], saved_average[k], rel_tol=0, abs_tol=1e-12)
                for k in average
            )
        ),
        "Incorrect five-set mean",
    )
    return {
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "complete": complete,
        "missing_datasets": sorted(set(PINS["datasets"]) - set(scores)),
        "evaluation_started_at": min(d[0] for d in dates),
        "evaluation_completed_at": max(d[1] for d in dates),
        "settings": config,
        "datasets": datasets,
        "models": {
            "soniox-stt-async-v5": {
                **config["model"],
                "datasets": scores,
                "core5_mean": average,
                "transcript_sha256": transcript_hashes,
                "raw_jobs_sha256": job_hashes,
            }
        },
        "config_sha256": sha256(folder / "config.json"),
        "summary_sha256": sha256(folder / "summary.json"),
        "api_preflight_sha256": sha256(folder / "api-preflight.json"),
        "audit_source_sha256": sha256(Path(__file__)),
        "lockfile_sha256": sha256(Path(__file__).resolve().parents[1] / "uv.lock"),
        "execution_history": history,
    }


def render(result):
    model = result["models"]["soniox-stt-async-v5"]
    lines = [
        "# Soniox Danish ASR evaluation",
        "",
        f"Audited {result['audited_at']}.",
        "",
        "Full five-set benchmark."
        if result["complete"]
        else "Incomplete benchmark; only complete, audited test sets are shown. No five-set mean.",
        "",
        "| Test set | Clips | WER (%) | CER (%) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, values in model["datasets"].items():
        lines.append(
            f"| {name} | {values['samples']} | {values['wer']:.2f} | {values['cer']:.2f} |"
        )
    if result["complete"]:
        values = model["core5_mean"]
        lines.append(f"| Five-set mean | | {values['wer']:.2f} | {values['cer']:.2f} |")
    lines += [
        "",
        "Model: `stt-async-v5`, hosted Soniox API; weight revision unavailable.",
        "Danish language hint, no translation, context or speaker diarization.",
        "Pinned RyeAI normalization and corpus WER/CER; identical baseline data fingerprints.",
        "API timing includes network and queue waits and sums per-clip times across workers.",
        "It is not GPU inference speed or the elapsed duration of this parallel run.",
        "Raw audio, reference transcripts, API responses and credentials are not in this export.",
        "Raw responses and predictions remain in the ignored local results folder.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, default=Path("evaluation/results/soniox-stt-async-v5-us/full")
    )
    parser.add_argument("--cache", type=Path, default=Path(".cache"))
    parser.add_argument("--cv-dir", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = audit(args.run, args.cache.resolve(), args.cv_dir, args.allow_partial)
    write_json(args.run / "audit.json", result)
    (args.run / "audit.md").write_text(render(result), encoding="utf-8")
    if result["complete"]:
        write_json(Path(__file__).with_name("soniox-benchmark.json"), result)
        Path(__file__).with_name("SONIOX_RESULTS.md").write_text(render(result), encoding="utf-8")
    else:
        print("Partial audit saved locally; no full benchmark export written.")


if __name__ == "__main__":
    main()
