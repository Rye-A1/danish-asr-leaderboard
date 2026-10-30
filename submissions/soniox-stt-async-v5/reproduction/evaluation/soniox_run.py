"""Finish an authorized Soniox run and audit it; credentials stay in the environment."""

import argparse
import importlib.metadata
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from evaluation.checkpoints import PINS, sha256
from evaluation.results import write_json
from evaluation.soniox import Soniox
from evaluation.soniox_evaluation import main as evaluate
from evaluation.soniox_report import audit, render
from evaluation.soniox_upgrade import upgrade


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-fleurs", action="store_true")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--upgrade-client", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 64:
        parser.error("workers must be between 1 and 64")
    folder = Path("evaluation/results/soniox-stt-async-v5-us/full")
    folder.mkdir(parents=True, exist_ok=True)
    receipt = {"started_at": now(), "phase": "preflight", "complete": False}
    status_path = folder / "run-status.json"
    write_json(status_path, receipt)
    try:
        distribution = importlib.metadata.distribution("danish-asr-leaderboard")
        installed = json.loads(distribution.read_text("direct_url.json"))
        if installed["vcs_info"]["commit_id"] != PINS["harness"]:
            raise ValueError("Installed Rye harness differs from the pinned commit")
        if args.wait_for_fleurs:
            receipt["phase"] = "waiting_for_initial_fleurs_run"
            write_json(status_path, receipt)
            deadline = time.monotonic() + 3600
            while True:
                summary_path = folder / "summary.json"
                if (
                    summary_path.exists()
                    and read(summary_path).get("datasets", {}).get("fleurs", {}).get("samples")
                    == PINS["datasets"]["fleurs"]["expected_rows"]
                ):
                    break
                initial_log = Path(".cache/soniox-full-fleurs.log")
                if (
                    initial_log.exists()
                    and "Traceback (most recent call last)"
                    in initial_log.read_text(encoding="utf-8", errors="replace")
                ):
                    raise RuntimeError("Initial FLEURS run failed; inspect its log before resuming")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Initial FLEURS run did not complete within one hour")
                time.sleep(5)
        if args.upgrade_client:
            upgrade(folder, args.workers)
        source = folder / "source"
        source.mkdir(exist_ok=True)
        for name in ("soniox.py", "soniox_evaluation.py", "data.py", "results.py"):
            path = source / name
            if not path.exists():
                shutil.copy2(Path(__file__).with_name(name), path)
        for name in ("soniox_run.py", "soniox_report.py", "soniox_upgrade.py"):
            shutil.copy2(Path(__file__).with_name(name), source / name)
        write_json(
            folder / "run-provenance.json",
            {
                "recorded_at": now(),
                "installed_harness": installed,
                "driver_sha256": sha256(Path(__file__)),
                "client_transport": {
                    "http": "requests.Session",
                    "initial_requests_per_second": 12,
                    "rate_limit_cooldown_seconds": 60,
                    "adaptive_backoff": True,
                },
                "lockfile_sha256": sha256(Path(__file__).resolve().parents[1] / "uv.lock"),
                "common_voice_preparation": read(Path(".cache/rye/source.json"))
                if Path(".cache/rye/source.json").exists()
                else None,
            },
        )
        config_path = folder / "config.json"
        if config_path.exists():
            config = read(config_path)
            if config["workers"] != args.workers:
                # Recognition settings are unchanged. Archive the earlier execution config.
                for path in folder.glob("jobs/*/*.json"):
                    if not read(path).get("remote_deleted"):
                        raise ValueError("Cannot change concurrency with unfinished saved jobs")
                history_path = folder / "execution-history.json"
                history = read(history_path) if history_path.exists() else []
                archive = folder / f"execution-config-{len(history):03d}.json"
                shutil.copy2(config_path, archive)
                history.append(
                    {
                        "changed_at": now(),
                        "reason": "Increase independent clip concurrency",
                        "previous_workers": config["workers"],
                        "new_workers": args.workers,
                        "previous_config_file": archive.name,
                        "previous_config_sha256": sha256(archive),
                        "completed_sets_before_change": read(folder / "summary.json")["datasets"],
                    }
                )
                write_json(history_path, history)
                config["workers"] = args.workers
                write_json(config_path, config)
        client = Soniox()
        if not (folder / "api-preflight.json").exists():
            write_json(
                folder / "api-preflight.json",
                {
                    "checked_at": now(),
                    "models": client.request("GET", "/models"),
                    "concurrency_limits": client.request("GET", "/concurrency-limits"),
                },
            )
        receipt["phase"] = "full_transcription"
        write_json(status_path, receipt)
        evaluate(
            SimpleNamespace(
                model="stt-async-v5",
                region="us",
                timeout=600,
                workers=args.workers,
                datasets=list(PINS["datasets"]),
                max_samples=0,
                cache=Path(".cache"),
                cv_dir=Path(".cache/common_voice"),
                output=Path("evaluation/results"),
            )
        )
        write_json(
            folder / "api-postflight.json",
            {
                "checked_at": now(),
                "models": client.request("GET", "/models"),
                "concurrency_limits": client.request("GET", "/concurrency-limits"),
            },
        )
        receipt["phase"] = "independent_audit"
        write_json(status_path, receipt)
        result = audit(folder, Path(".cache").resolve(), Path(".cache/common_voice"))
        write_json(folder / "audit.json", result)
        (folder / "audit.md").write_text(render(result), encoding="utf-8")
        write_json(Path("evaluation/soniox-benchmark.json"), result)
        Path("evaluation/SONIOX_RESULTS.md").write_text(render(result), encoding="utf-8")
        receipt.update(phase="complete", complete=True, finished_at=now())
        write_json(status_path, receipt)
    except Exception as exc:
        receipt.update(
            phase="failed", error_type=type(exc).__name__, error=str(exc), failed_at=now()
        )
        write_json(status_path, receipt)
        raise


if __name__ == "__main__":
    main()
