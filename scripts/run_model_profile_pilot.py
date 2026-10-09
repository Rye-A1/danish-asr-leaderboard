#!/usr/bin/env python3
"""Run a local, non-scoring agent pilot and compare it with reviewed profiles."""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

from compare_model_profile_agent import compare, markdown
from model_profile_agent import (MODEL, _add_saved_output_source, ask_agent,
                                 collect_sources, validate_suggestions)
from model_profiles import load_reviews
from refresh_model_profile_candidates import collect, model_records

DEFAULT_MODELS = ("danish-foundation-models/edda-v0.2", "RyeAI/ekko-v1-tiny")
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "eval_audio_cache" / "model-profile-pilot"


def run(models: set[str], key: str, *, output_dir: Path = DEFAULT_OUTPUT) -> dict:
    records = model_records()
    missing = models - records.keys()
    if missing:
        raise ValueError(f"Unknown result model(s): {', '.join(sorted(missing))}")
    output_dir.mkdir(parents=True, exist_ok=True)

    def status(message: str) -> None:
        print(message, file=sys.stderr, flush=True)
        (output_dir / "status.txt").write_text(message + "\n", encoding="utf-8")

    def safe_error(exc: Exception) -> str:
        return f"{type(exc).__name__}: {str(exc).replace(key, '[redacted]')[:500]}"

    drafts = {}
    with requests.Session() as session:
        status(f"Collecting public sources for {len(models)} model(s)...")
        try:
            candidates = collect(session, only_models=models)
        except Exception as exc:
            status(f"Source collection failed: {safe_error(exc)}")
            raise
        for name in sorted(models):
            try:
                status(f"Collecting evidence for {name}...")
                candidate = candidates[name]
                sources = collect_sources(session, candidate)
                _add_saved_output_source(sources, name)
                if not any(s["kind"] in {"model_card", "provider_api"} for s in sources):
                    status(f"{name}: no public model card or official API source; manual review only")
                    continue
                status(f"Requesting {MODEL} review for {name}...")
                fields = validate_suggestions(
                    ask_agent(session, key, name, candidate, sources), sources)
                if candidate.get("source_kind") == "provider_api":
                    for field in ("license", "data", "code", "model_card"):
                        fields[field] = {"state": "no", "evidence": [],
                                         "explanation": "Hosted/API openness policy."}
                drafts[name] = {
                    "agent_model": MODEL,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "sources": [{"id": s["id"], "kind": s["kind"], "url": s["url"]}
                                for s in sources],
                    "fields": fields,
                }
                status(f"Reviewed {name}")
            except Exception as exc:
                status(f"{name} failed: {safe_error(exc)}")
                raise
    if not drafts:
        raise ValueError("No model received an agent draft")
    report = compare(drafts, load_reviews(), records)
    (output_dir / "suggestions.json").write_text(
        json.dumps(drafts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "comparison.md").write_text(markdown(report), encoding="utf-8")
    status(f"Finished {report['models_compared']} model(s); "
           f"{report['exact_matches']}/{report['fields_compared']} fields match")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", dest="models",
                        help="Exact leaderboard model ID; repeat to compare several")
    parser.add_argument("--all", action="store_true", help="Try all current models")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    records = model_records()
    models = set(records) if args.all else set(args.models or DEFAULT_MODELS)
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        if not sys.stdin.isatty():
            raise SystemExit("Set OPENROUTER_API_KEY locally or run in a terminal for a hidden prompt")
        try:
            key = getpass.getpass("OpenRouter API key (hidden; not saved): ")
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\nNo key entered; pilot cancelled") from None
    if not key:
        raise SystemExit("An OpenRouter API key is required")
    report = run(models, key, output_dir=args.output_dir)
    print(markdown(report))
    print(f"Saved public-source drafts and comparison under {args.output_dir}")


if __name__ == "__main__":
    main()
