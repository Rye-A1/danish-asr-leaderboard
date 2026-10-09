#!/usr/bin/env python3
"""Compare source-quoted agent drafts with the human-reviewed leaderboard profiles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_profile_agent import FIELDS
from model_profiles import PROFILE_FILE, build_profile, load_reviews
from refresh_model_profile_candidates import model_records

SUGGESTIONS = Path(__file__).with_name("model_profile_agent_suggestions.json")


def compare(suggestions: dict, reviews: dict, records: dict) -> dict:
    groups = {field: "openness" if field in FIELDS[:4] else "features"
              for field in FIELDS}
    rows = []
    for name, draft in sorted(suggestions.items()):
        if name not in records:
            continue
        record = records[name]
        profile = build_profile(name, record["source"], {}, reviews,
                                access=record["access"])
        fields = draft.get("fields", {})
        for field in FIELDS:
            human = profile[groups[field]][field]["state"]
            agent = fields.get(field, {}).get("state", "unknown")
            rows.append({"model": name, "field": field, "human": human,
                         "agent": agent, "match": human == agent})
    return {"models_compared": len({row["model"] for row in rows}),
            "models_without_draft": sorted(set(records) - set(suggestions)),
            "fields_compared": len(rows),
            "exact_matches": sum(row["match"] for row in rows),
            "rows": rows}


def markdown(report: dict) -> str:
    lines = ["# Agent versus reviewed model profiles", "",
             f"Models compared: **{report['models_compared']}**; fields: "
             f"**{report['fields_compared']}**; exact matches: "
             f"**{report['exact_matches']}**.", "",
             "Partial and unknown count as distinct states. This is an agreement "
             "check, not an accuracy estimate or an automatic score update.", "",
             "## Differences", ""]
    mismatches = [row for row in report["rows"] if not row["match"]]
    if mismatches:
        lines += ["| Model | Field | Human | Agent |", "|---|---|---|---|"]
        lines += [f"| `{row['model']}` | {row['field']} | {row['human']} | {row['agent']} |"
                  for row in mismatches]
    else:
        lines.append("None in the available drafts.")
    lines += ["", f"Models without a draft: {len(report['models_without_draft'])}.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suggestions", type=Path, default=SUGGESTIONS)
    args = parser.parse_args()
    if not args.suggestions.exists():
        raise SystemExit(f"No agent drafts found at {args.suggestions}")
    report = compare(json.loads(args.suggestions.read_text(encoding="utf-8")),
                     load_reviews(PROFILE_FILE), model_records())
    print(markdown(report))


if __name__ == "__main__":
    main()
