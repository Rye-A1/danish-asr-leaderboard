#!/usr/bin/env python3
"""Repo-root CLI shim for the webhook notifier.

The implementation lives in ``danish_asr_leaderboard/notify.py`` (so it imports
cleanly under the installed console script). This shim keeps
``python notify.py "msg"`` working for ``scripts/run_benchmark.sh``.
"""
import sys

from danish_asr_leaderboard.notify import get_webhook_url, notify, send_mattermost  # noqa: F401

if __name__ == "__main__":
    notify(sys.argv[1] if len(sys.argv) > 1 else "")
