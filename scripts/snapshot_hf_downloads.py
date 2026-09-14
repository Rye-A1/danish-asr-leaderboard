#!/usr/bin/env python3
"""Record daily rolling-30-day Hugging Face download counts for leaderboard models."""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HISTORY_PATH = ROOT / "history" / "hf_downloads.json"
DEFAULT_RESULTS_DIR = ROOT / "results"
_MODEL_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")


def model_repositories(results_dir: Path) -> dict[str, str]:
    """Return leaderboard display names mapped to their Hugging Face repo ids."""
    repositories: dict[str, str] = {}
    for path in sorted(results_dir.glob("*.json")):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if result.get("access") == "proprietary":
            continue
        model = result.get("model", "")
        match = _MODEL_LINK.fullmatch(str(model).strip())
        if not match:
            continue
        name, url = match.groups()
        parsed = urlparse(url)
        if parsed.netloc != "huggingface.co":
            continue
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 2:
            repositories[name] = "/".join(parts[:2])
    return repositories


def fetch_downloads(model_id: str) -> int:
    """Fetch Hugging Face's rolling 30-day download count for one model."""
    request = Request(
        f"https://huggingface.co/api/models/{model_id}",
        headers={"User-Agent": "danish-asr-leaderboard-download-snapshot"},
    )
    with urlopen(request, timeout=15) as response:  # noqa: S310 - fixed HTTPS host
        payload = json.load(response)
    downloads = int(payload["downloads"])
    if downloads < 0:
        raise ValueError(f"negative download count for {model_id}")
    return downloads


def load_history(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"version": 1, "snapshots": []}


def upsert_snapshot(history: dict, snapshot_date: str, downloads: dict[str, int]) -> dict:
    """Add or replace one day's values while retaining prior values on retries."""
    snapshots = [item for item in history.get("snapshots", []) if item.get("date") != snapshot_date]
    existing = next((item for item in history.get("snapshots", []) if item.get("date") == snapshot_date), {})
    values = dict(existing.get("downloads", {}))
    values.update(downloads)
    snapshots.append({"date": snapshot_date, "downloads": dict(sorted(values.items()))})
    snapshots.sort(key=lambda item: item["date"])
    return {"version": 1, "snapshots": snapshots}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--date", default=date.today().isoformat(), help="ISO snapshot date")
    args = parser.parse_args()

    sources = model_repositories(args.results_dir)
    if not sources:
        print("ERROR: no Hugging Face model repositories found", file=sys.stderr)
        raise SystemExit(1)

    downloaded: dict[str, int] = {}
    for name, model_id in sources.items():
        try:
            downloaded[name] = fetch_downloads(model_id)
        except Exception as exc:  # noqa: BLE001 - retain prior data on a transient API failure
            print(f"WARNING: could not fetch {model_id}: {exc}", file=sys.stderr)
    if not downloaded:
        print("ERROR: no download counts fetched", file=sys.stderr)
        raise SystemExit(1)

    history = load_history(args.history)
    updated = upsert_snapshot(history, args.date, downloaded)
    if updated == history:
        print(f"No download changes for {args.date}.")
        return
    args.history.parent.mkdir(parents=True, exist_ok=True)
    args.history.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded {len(downloaded)} Hugging Face download counts for {args.date}.")


if __name__ == "__main__":
    main()