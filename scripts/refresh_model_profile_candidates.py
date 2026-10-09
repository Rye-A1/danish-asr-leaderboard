#!/usr/bin/env python3
"""Collect public evidence leads for new or changed leaderboard models.

The generated file is a review queue, not a scoring input. The Space can show
safe facts immediately; a human confirms claims that metadata cannot prove.
Run from any directory: python scripts/refresh_model_profile_candidates.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import quote

import requests

from model_profiles import PROFILE_FILE, license_tag_class, load_reviews

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUTPUT = Path(__file__).with_name("model_profile_candidates.json")
MODEL_LINK = re.compile(r"^\[([^]]+)\]\((https://[^)]+)\)$")
HF_MODEL = re.compile(r"^https://huggingface\.co/([^/]+/[^/?#]+)(?:[/?#].*)?$")
ARXIV = re.compile(r"^\d{4}\.\d{4,5}$")
GITHUB_LINK = re.compile(r"https://github\.com/[\w.-]+/[\w.-]+(?:/[^\s)<>]*)?")
TRAINING_NAME = re.compile(r"train|finetun|preprocess|recipe", re.I)
HEADING = re.compile(r"^#{1,4}\s+(.+?)\s*$", re.M)


def model_rows(results: Path = RESULTS) -> dict[str, str]:
    """Read the model IDs and source URLs used in committed result files."""
    rows = {}
    for path in sorted(results.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        match = MODEL_LINK.fullmatch(result.get("model", "").strip())
        if match:
            rows[match.group(1)] = match.group(2)
    return rows


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
    dataset_urls = sorted({f"https://huggingface.co/datasets/{d}" for d in datasets
                           if isinstance(d, str) and re.fullmatch(r"[\w.-]+/[\w.-]+", d)})
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
        "dataset_links": dataset_urls,
        "paper_links": paper_urls,
        "training_code_leads": code_links[:12] + [f"{source}/blob/main/{quote(f)}" for f in training_files[:12]],
        "card_headings": headings[:24],
        "card_characters": len(readme),
        "review_note": "Links and tags are leads. Check exact checkpoint, base terms, data completeness, and code before scoring.",
    }


def needs_review(review: dict | None, info: dict | None) -> bool:
    if not review:
        return True
    modified = str((info or {}).get("lastModified") or "")[:10]
    reviewed_on = review.get("reviewed_on", "")
    return bool(modified and (not reviewed_on or modified > reviewed_on))


def collect(session: requests.Session, *, results: Path = RESULTS,
            reviews_path: Path = PROFILE_FILE) -> dict:
    reviews = load_reviews(reviews_path)
    output = {}
    for name, source in model_rows(results).items():
        match = HF_MODEL.fullmatch(source)
        if not match:
            if name not in reviews:
                output[name] = {"status": "new", "source": source,
                                "review_note": "Hosted model: review provider documentation manually."}
            continue
        repo = match.group(1)
        response = session.get(f"https://huggingface.co/api/models/{repo}",
                               timeout=15)
        if response.status_code not in (200, 401, 403, 404):
            response.raise_for_status()
        info = response.json() if response.ok else None
        if not needs_review(reviews.get(name), info):
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
                        **evidence_leads(name, source, info, readme)}
    return dict(sorted(output.items(), key=lambda x: x[0].casefold()))


def main() -> None:
    with requests.Session() as session:
        candidates = collect(session)
    OUTPUT.write_text(json.dumps(candidates, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT} with {len(candidates)} candidate model(s)")


if __name__ == "__main__":
    main()
