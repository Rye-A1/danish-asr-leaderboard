#!/usr/bin/env python3
"""Build a self-contained, offline preview from the checked-in result JSONs.

Usage from the repo root:
    python scripts/build_local_preview.py

Open space/local-preview.html directly. This file is ignored by Git and is not
uploaded to the Space. Hub metadata is unavailable offline, so some profile
fields remain unknown in the preview.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import update_space
from push_results import COLUMNS

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "space" / "local-preview.html"


def main() -> None:
    rows = []
    for path in sorted((ROOT / "results").glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        row = {col: result.get(col) for col in COLUMNS}
        row["access"] = row["access"] or "open"
        name, _ = update_space._parse_model(row["model"])
        if name.lower() in update_space.EXCLUDE_MODELS:
            continue
        if name.split("/", 1)[0].lower() in update_space.EXCLUDE_ORGS:
            continue
        rows.append(row)
    if not rows:
        raise SystemExit("No results/*.json files found for the local preview")

    # Reuse the deployment builder, while keeping preview generation offline.
    update_space._model_metadata = lambda _name: {}
    update_space._provider_logo = lambda _org: ""
    update_space._bootstrap_cis = lambda: {}
    update_space._model_downloads = lambda _name: None
    update_space._model_download_history = lambda _name: ()
    data = update_space.build_leaderboard_json(pd.DataFrame(rows, columns=COLUMNS))

    source = (ROOT / "space" / "index.html").read_text(encoding="utf-8")
    marker = "\n<script>\n"
    if source.count(marker) != 1:
        raise SystemExit("Could not find the leaderboard script in space/index.html")
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    embedded = f'\n<script id="embedded-leaderboard" type="application/json">{payload}</script>'
    note = ('  <p role="status" style="margin:12px 0;color:#fbbf24;font-size:13px">'
            'Local preview from checked-in results. Hub metadata and download counts are unavailable offline.'
            '</p>\n')
    preview = source.replace("  <!-- Tabs -->\n", note + "  <!-- Tabs -->\n", 1)
    OUTPUT.write_text(preview.replace(marker, embedded + marker), encoding="utf-8")
    print(f"Wrote {OUTPUT} ({len(data['wer'])} models; open it directly in a browser)")


if __name__ == "__main__":
    main()
