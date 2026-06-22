"""Fire-and-forget webhook notifier (stdlib only, never raises).

Used by ``scripts/run_benchmark.sh`` and the eval CLI to post progress. The
webhook URL is read from the ``MATTERMOST_WEBHOOK_URL`` environment variable, or a
``.env`` file (cwd or repo root). If no URL is configured, every call is a silent
no-op — so runs behave identically with or without notifications.

Works with **Slack and Mattermost out of the box** — both accept the same
``{"text": ...}`` incoming-webhook payload, so just point the env var at either.
To support Discord (``{"content": ...}``), Teams, or another service, change the
single payload dict in ``send_mattermost`` — the rest is service-agnostic.

This lives inside the package so it imports cleanly under the installed
``danish-asr-eval`` console script regardless of cwd; ``notify.py`` at the repo
root is a thin CLI shim over this module.

CLI:
    python -m danish_asr_leaderboard.notify "message text"
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

_ENV_KEY = "MATTERMOST_WEBHOOK_URL"
# Repo root = two levels up from this file (danish_asr_leaderboard/notify.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent


def get_webhook_url(explicit: str = "") -> str:
    """Webhook URL from explicit arg, env var, or a ``.env`` (cwd then repo root)."""
    if explicit:
        return explicit
    url = os.environ.get(_ENV_KEY, "").strip()
    if url:
        return url
    for env_path in (Path(".env"), _REPO_ROOT / ".env"):
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith(f"{_ENV_KEY}="):
                    val = line.split("=", 1)[1].strip().strip("\"'")
                    if val:
                        return val
    return ""


def send_mattermost(webhook_url: str, text: str) -> None:
    """POST ``text`` to the webhook. Never raises; no-op if the URL is empty."""
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
        print(f"[notify] webhook POST failed: {exc}", flush=True)


def notify(text: str) -> None:
    """Resolve the configured webhook and post ``text`` (silent no-op if unset)."""
    send_mattermost(get_webhook_url(), text)


if __name__ == "__main__":
    notify(sys.argv[1] if len(sys.argv) > 1 else "")
