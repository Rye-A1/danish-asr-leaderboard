#!/usr/bin/env python3
"""Collect public evidence leads for new or changed leaderboard models.

The generated file is a review queue, not a scoring input. The Space can show
safe facts immediately; a human confirms claims that metadata cannot prove.
Run from any directory: python scripts/refresh_model_profile_candidates.py
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import quote

import requests

from model_profiles import PROFILE_FILE, license_tag_class, load_reviews

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUTPUT = Path(__file__).with_name("model_profile_candidates.json")
PROVIDER_SOURCES = Path(__file__).with_name("provider_profile_sources.json")
MODEL_LINK = re.compile(r"^\[([^]]+)\]\((https://[^)]+)\)$")
HF_MODEL = re.compile(r"^https://huggingface\.co/([^/]+/[^/?#]+)(?:[/?#].*)?$")
ARXIV = re.compile(r"^\d{4}\.\d{4,5}$")
GITHUB_LINK = re.compile(r"https://github\.com/[\w.-]+/[\w.-]+(?:/[^\s)<>]*)?")
HF_DATASET_LINK = re.compile(r"https://huggingface\.co/datasets/([\w.-]+/[\w.-]+)")
TRAINING_NAME = re.compile(r"train|finetun|preprocess|recipe", re.I)
HEADING = re.compile(r"^#{1,4}\s+(.+?)\s*$", re.M)


def model_records(results: Path = RESULTS) -> dict[str, dict]:
    """Read profile identity, access, and file path from committed results."""
    rows = {}
    for path in sorted(results.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        match = MODEL_LINK.fullmatch(result.get("model", "").strip())
        if match:
            rows[match.group(1)] = {"source": match.group(2),
                                    "access": result.get("access", "unknown"),
                                    "result_file": path.name}
    return rows


def model_rows(results: Path = RESULTS) -> dict[str, str]:
    """Compatibility view of the model IDs and source URLs."""
    return {name: row["source"] for name, row in model_records(results).items()}


def evidence_leads(name: str, source: str, info: dict | None, readme: str = "") -> dict:
    """Extract candidate evidence; none of these leads earns a positive score."""
    if not info:
        return {"source": source, "metadata": "unavailable"}
    tags = [t for t in info.get("tags", []) if isinstance(t, str)]
    card = info.get("cardData") if isinstance(info.get("cardData"), dict) else {}
    license_tag = str(card.get("license") or next(
        (t.partition(":")[2] for t in tags if t.startswith("license:")), ""))
    datasets = card.get("datasets") or []
    if isinstance(datasets, str):
        datasets = [datasets]
    if not isinstance(datasets, list):
        datasets = []
    datasets += [t.partition(":")[2] for t in tags if t.startswith("dataset:")]
    dataset_ids = {d for d in datasets
                   if isinstance(d, str) and re.fullmatch(r"[\w.-]+/[\w.-]+", d)}
    dataset_ids.update(HF_DATASET_LINK.findall(readme))
    dataset_urls = sorted(f"https://huggingface.co/datasets/{d}" for d in dataset_ids)
    arxiv_ids = [t.partition(":")[2] for t in tags if t.startswith("arxiv:")]
    paper_urls = sorted({f"https://arxiv.org/abs/{a}" for a in arxiv_ids if ARXIV.fullmatch(a)})
    siblings = [s.get("rfilename", "") for s in info.get("siblings", [])
                if isinstance(s, dict)]
    weight_files = sorted(f for f in siblings if f.endswith((
        ".safetensors", ".bin", ".gguf", ".onnx", ".ckpt")))
    training_files = sorted(f for f in siblings if (
        (f.endswith((".py", ".sh")) and TRAINING_NAME.search(f))
        or (f.endswith((".yaml", ".yml", ".json"))
            and re.search(r"train|finetun|recipe", f, re.I))))
    code_links = sorted({m.group(0).rstrip(".,") for m in GITHUB_LINK.finditer(readme)
                         if TRAINING_NAME.search(m.group(0))})
    prose = re.sub(r"(?ms)^```.*?^```\s*$", "", readme)
    headings = [h.strip().rstrip("# ") for h in HEADING.findall(prose)]
    return {
        "source": source,
        "last_modified": str(info.get("lastModified") or "")[:10],
        "revision": info.get("sha") or "",
        "gated": info.get("gated", False),
        "weight_files": weight_files[:12],
        "license_tag": license_tag,
        "license_tag_class": license_tag_class(license_tag),
        "license_link": card.get("license_link") if isinstance(card.get("license_link"), str) else "",
        "dataset_links": dataset_urls,
        "unverified_paper_links": paper_urls,
        "training_code_leads": code_links[:12] + [f"{source}/blob/main/{quote(f)}" for f in training_files[:12]],
        "card_headings": headings[:24],
        "card_characters": len(readme),
        "review_note": "Links and tags are leads. A base-model paper is not a paper for its fine-tune. Check the exact checkpoint, base terms, data completeness, and code before scoring.",
    }


def needs_review(review: dict | None, info: dict | None) -> bool:
    if not review:
        return True
    modified = str((info or {}).get("lastModified") or "")[:10]
    reviewed_on = review.get("reviewed_on", "")
    return bool(modified and (not reviewed_on or modified > reviewed_on))


def collect(session: requests.Session, *, results: Path = RESULTS,
            reviews_path: Path = PROFILE_FILE,
            provider_sources_path: Path = PROVIDER_SOURCES,
            include_all: bool = False,
            only_models: set[str] | None = None) -> dict:
    reviews = load_reviews(reviews_path)
    provider_sources = (json.loads(provider_sources_path.read_text(encoding="utf-8"))
                        if provider_sources_path.exists() else {})
    output = {}
    for name, row in model_records(results).items():
        if only_models is not None and name not in only_models:
            continue
        source = row["source"]
        if row["access"] == "proprietary":
            if include_all or name not in reviews or only_models is not None:
                output[name] = {
                    "status": "new" if name not in reviews else "scheduled_review",
                    "source": source, "source_kind": "provider_api",
                    "provider_docs": provider_sources.get(name, []),
                    "review_note": "Check the exact evaluated deployment and model compatibility in official API documentation; generic provider features are insufficient.",
                }
            continue
        match = HF_MODEL.fullmatch(source)
        if not match:
            if include_all or name not in reviews or only_models is not None:
                output[name] = {"status": "new", "source": source,
                                "review_note": "Hosted model: review provider documentation manually."}
            continue
        repo = match.group(1)
        response = session.get(f"https://huggingface.co/api/models/{repo}",
                               timeout=15)
        if response.status_code not in (200, 401, 403, 404):
            response.raise_for_status()
        info = response.json() if response.ok else None
        if not (include_all or only_models is not None
                or needs_review(reviews.get(name), info)):
            continue
        readme = ""
        if info:
            card_response = session.get(f"https://huggingface.co/{repo}/resolve/main/README.md",
                                        timeout=15)
            if card_response.ok:
                readme = card_response.text
            elif card_response.status_code not in (401, 403, 404):
                card_response.raise_for_status()
        output[name] = {"status": "new" if name not in reviews else "source_changed",
                        "source_kind": "hf_model", **evidence_leads(name, source, info, readme)}
    return dict(sorted(output.items(), key=lambda x: x[0].casefold()))


def changed_result_files(base: str) -> set[str]:
    paths = subprocess.check_output(
        ["git", "diff", "--name-only", "--diff-filter=ACMR", base, "HEAD", "--", "results/"],
        cwd=ROOT, text=True).splitlines()
    return {Path(path).name for path in paths if path.startswith("results/")}


def changed_provider_models(base: str) -> set[str]:
    try:
        old_text = subprocess.check_output(
            ["git", "show", f"{base}:scripts/provider_profile_sources.json"],
            cwd=ROOT, text=True, stderr=subprocess.DEVNULL)
        old = json.loads(old_text)
    except (subprocess.CalledProcessError, ValueError):
        old = {}
    current = json.loads(PROVIDER_SOURCES.read_text(encoding="utf-8"))
    return {name for name in old.keys() | current.keys()
            if old.get(name) != current.get(name)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Check all current model sources")
    parser.add_argument("--changed-from", metavar="BASE_SHA",
                        help="Check only models with result files changed since this commit")
    parser.add_argument("--model", help="Check one exact leaderboard model ID")
    args = parser.parse_args()
    models = None
    if args.changed_from:
        files = changed_result_files(args.changed_from)
        models = {name for name, row in model_records().items()
                  if row["result_file"] in files}
        models |= changed_provider_models(args.changed_from)
    if args.model:
        models = {args.model} if models is None else models & {args.model}
    with requests.Session() as session:
        candidates = collect(session, include_all=args.all,
                             only_models=models)
    OUTPUT.write_text(json.dumps(candidates, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT} with {len(candidates)} candidate model(s)")


if __name__ == "__main__":
    main()
