"""Fire-and-forget webhook notifier (stdlib only, never raises).

Used by ``scripts/run_benchmark.sh`` to post sweep progress. The webhook URL is
read from the ``MATTERMOST_WEBHOOK_URL`` environment variable, or a ``.env`` file
in the repo root. If no URL is configured, every call is a silent no-op — so the
sweep runs identically with or without notifications.

Works with **Slack and Mattermost out of the box** — both accept the same
``{"text": ...}`` incoming-webhook payload, so just point the env var at either.
To support Discord (``{"content": ...}``), Teams, or another service, change the
single payload dict in ``send_mattermost`` — the rest is service-agnostic.

CLI:
    python notify.py "message text"
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path


def get_webhook_url(explicit: str = "") -> str:
    """Webhook URL from explicit arg, ``MATTERMOST_WEBHOOK_URL`` env, or .env file."""
    if explicit:
        return explicit
    url = os.environ.get("MATTERMOST_WEBHOOK_URL", "").strip()
    if url:
        return url
    for env_path in (Path(".env"), Path(__file__).resolve().parent / ".env"):
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("MATTERMOST_WEBHOOK_URL="):
                    val = line.split("=", 1)[1].strip().strip("\"'")
                    if val:
                        return val
    return ""


def send_mattermost(webhook_url: str, text: str) -> None:
    """POST ``text`` to the Mattermost webhook. Never raises; no-op if URL empty."""
    if not webhook_url:
        return
    try:
        # Slack & Mattermost both use {"text": ...}; change this dict for others.
        data = json.dumps({"text": text}).encode("utf-8")
        req = urllib.request.Request(
            webhook_url, data=data,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as exc:  # noqa: BLE001 - notifications must never break a run
        print(f"[Mattermost] notification failed: {exc}", flush=True)


if __name__ == "__main__":
    msg = sys.argv[1] if len(sys.argv) > 1 else ""
    send_mattermost(get_webhook_url(), msg)
