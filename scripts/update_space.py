#!/usr/bin/env python3
"""Build and deploy the static HTML Space.

Bakes leaderboard.json from the results parquet, resolving provider logos and
formatting sizes server-side, then uploads static files and removes obsolete
gradio files from the Space repo.

Usage:
  python scripts/update_space.py

Requires HF_TOKEN with write access to the RyeAI org:
  export HF_TOKEN=hf_...
"""
from __future__ import annotations

import functools
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from huggingface_hub import HfApi, get_token

SPACE_REPO_ID = "RyeAI/danish-asr-leaderboard"
DATASET_PARQUET = "hf://datasets/RyeAI/danish-asr-leaderboard/data/results.parquet"
SPACE_DIR = Path(__file__).resolve().parent.parent / "space"

UPLOAD = ["index.html", "leaderboard.json", "README.md", "cover.jpeg"]
OBSOLETE = ["app.py", "requirements.txt"]

_MD_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")


@functools.lru_cache(maxsize=256)
def _provider_logo(org: str) -> str:
    """Best-effort HF avatar URL for an org/user handle (cached per run)."""
    for kind in ("organizations", "users"):
        try:
            r = requests.get(
                f"https://huggingface.co/api/{kind}/{org}/avatar", timeout=4
            )
            if r.ok:
                url = r.json().get("avatarUrl")
                if url:
                    return url
        except Exception:
            continue
    return ""


def _fmt_size(x) -> str:
    """1 decimal (2 for sub-0.1B), em dash for 0 / NaN (API models)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if pd.isna(v) or v <= 0:
        return "—"
    return f"{v:.2f}" if v < 0.1 else f"{v:.1f}"


def _num(x) -> float | None:
    """JSON-safe float rounded to 2 decimal places, or None."""
    try:
        v = float(x)
        if pd.isna(v):
            return None
        return round(v, 2)
    except (TypeError, ValueError):
        return None


def _parse_model(cell: str) -> tuple[str, str]:
    """Return (display_name, url) from a markdown link cell."""
    if not isinstance(cell, str):
        return str(cell), ""
    m = _MD_LINK.fullmatch(cell.strip())
    if m:
        return m.group(1), m.group(2)
    return cell, ""


def build_leaderboard_json() -> dict:
    df = pd.read_parquet(DATASET_PARQUET)

    if "rtf" in df.columns and "speed_x" not in df.columns:
        df["speed_x"] = (1.0 / df["rtf"]).round(1)
        df = df.drop(columns=["rtf"])
    if "access" not in df.columns:
        df["access"] = "open"

    def build_rows(df_sorted: pd.DataFrame, metric_cols: list[str]) -> list[dict]:
        rows = []
        for rank, (_, row) in enumerate(df_sorted.iterrows(), 1):
            name, url = _parse_model(row.get("model", ""))
            org = name.split("/", 1)[0] if "/" in name else ""
            logo = _provider_logo(org) if org else ""
            submitted = row.get("submitted")
            entry: dict = {
                "rank": rank,
                "name": name,
                "url": url,
                "logo": logo,
                "access": str(row.get("access", "open")),
                "size": _fmt_size(row.get("params_b")),
                "submitted": str(submitted)[:10] if pd.notna(submitted) else "",
            }
            for col in metric_cols:
                entry[col] = _num(row.get(col))
            rows.append(entry)
        return rows

    wer_metrics = [
        "mean_wer", "mean_cer", "speed_x",
        "coral_conversation_wer", "coral_read_aloud_wer",
        "ftspeech_wer", "cv17_da_wer", "fleurs_da_wer",
    ]
    cer_metrics = [
        "mean_cer",
        "coral_conversation_cer", "coral_read_aloud_cer",
        "ftspeech_cer", "cv17_da_cer", "fleurs_da_cer",
    ]

    wer_df = (df.dropna(subset=["mean_wer"])
                .sort_values("mean_wer", ascending=True)
                .reset_index(drop=True))
    cer_df = (df.dropna(subset=["mean_cer"])
                .sort_values("mean_cer", ascending=True)
                .reset_index(drop=True))

    return {
        "updated": date.today().isoformat(),
        "org_logo": _provider_logo("RyeAI"),
        "wer": build_rows(wer_df, wer_metrics),
        "cer": build_rows(cer_df, cer_metrics),
    }


def main() -> None:
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        print(
            "ERROR: no HF credentials found. "
            "Set HF_TOKEN or run `huggingface-cli login`.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Building leaderboard.json …")
    data = build_leaderboard_json()
    out = SPACE_DIR / "leaderboard.json"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"  {out}  ({len(data['wer'])} WER rows, {len(data['cer'])} CER rows)")

    api = HfApi(token=token)

    print("\nUploading static files …")
    for name in UPLOAD:
        path = SPACE_DIR / name
        if not path.exists():
            print(f"  skip (missing): {name}")
            continue
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=name,
            repo_id=SPACE_REPO_ID,
            repo_type="space",
            commit_message=f"Update {name}",
        )
        print(f"  ✓ {name}")

    print("\nRemoving obsolete gradio files …")
    space_files = set(api.list_repo_files(repo_id=SPACE_REPO_ID, repo_type="space"))
    for name in OBSOLETE:
        if name in space_files:
            api.delete_file(
                path_in_repo=name,
                repo_id=SPACE_REPO_ID,
                repo_type="space",
                commit_message=f"Remove {name} (static rewrite)",
            )
            print(f"  ✓ deleted {name}")
        else:
            print(f"  (not present) {name}")

    print(f"\nDone → https://huggingface.co/spaces/{SPACE_REPO_ID}")


if __name__ == "__main__":
    main()
