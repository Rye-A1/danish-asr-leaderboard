"""Archive a stopped run's client before changing transport or scheduling only."""

import importlib.metadata
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from evaluation.checkpoints import sha256
from evaluation.results import write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def upgrade(folder, workers):
    config_path = folder / "config.json"
    config = read(config_path)
    current = {name: sha256(Path(__file__).with_name(name)) for name in config["code_sha256"]}
    if current == config["code_sha256"] and config["workers"] == workers:
        return
    for name in ("data.py", "results.py"):
        if current[name] != config["code_sha256"][name]:
            raise ValueError("Client upgrade cannot change data preparation or scoring")
    jobs = sorted(folder.glob("jobs/*/*.json"))
    if any(read(path).get("pending_request") for path in jobs):
        raise ValueError("Resolve uncertain POST outcomes before upgrading the client")
    history_path = folder / "execution-history.json"
    history = read(history_path) if history_path.exists() else []
    phase = folder / f"client-phase-{len(history):03d}"
    if phase.exists():
        raise ValueError(f"Client phase already exists: inspect {phase}")
    phase.mkdir()
    shutil.copytree(folder / "source", phase / "source")
    shutil.copy2(config_path, phase / "config.json")
    if (folder / "run-provenance.json").exists():
        shutil.copy2(folder / "run-provenance.json", phase / "run-provenance.json")
    for name in ("client-restart.json", "client-failure.json", "request-profile.json"):
        if (folder / name).exists():
            shutil.copy2(folder / name, phase / name)
    prefixes = {}
    for path in folder.glob("*.jsonl"):
        shutil.copy2(path, phase / path.name)
        prefixes[path.name] = {
            "sha256": sha256(path),
            "rows": len(path.read_text(encoding="utf-8").splitlines()),
        }
    receipt = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "previous_transcript_prefixes": prefixes,
        "saved_jobs_sha256": {path.relative_to(folder).as_posix(): sha256(path) for path in jobs},
    }
    write_json(phase / "receipt.json", receipt)
    # Earlier worker-count changes used this same source revision.
    for event in history:
        event.setdefault("previous_source_dir", (phase / "source").relative_to(folder).as_posix())
    history.append(
        {
            "changed_at": receipt["recorded_at"],
            "reason": "Client transport/scheduling update; recognition and data/scoring unchanged",
            "previous_workers": config["workers"],
            "new_workers": workers,
            "previous_config_file": (phase / "config.json").relative_to(folder).as_posix(),
            "previous_config_sha256": sha256(phase / "config.json"),
            "previous_source_dir": (phase / "source").relative_to(folder).as_posix(),
            "receipt_sha256": sha256(phase / "receipt.json"),
            "transport_packages": {"requests": importlib.metadata.version("requests")},
            "initial_max_requests_per_second": 12,
        }
    )
    write_json(history_path, history)
    config.update(workers=workers, code_sha256=current)
    for name in current:
        shutil.copy2(Path(__file__).with_name(name), folder / "source" / name)
    write_json(config_path, config)
