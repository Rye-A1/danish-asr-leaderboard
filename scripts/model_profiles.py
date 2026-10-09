"""Evidence-linked openness and capability metadata for leaderboard models."""
from __future__ import annotations

import json
import functools
import math
import re
from datetime import date
from pathlib import Path

PROFILE_FILE = Path(__file__).with_name("model_profiles.json")
OPENNESS = ("weights", "license", "data", "code", "model_card")
ADDITIONAL = ("paper",)
FEATURES = ("punctuation_case", "timestamps", "diarization", "streaming")
STATES = {"yes", "no", "partial", "unknown"}
# Hub tags are discovery hints. Even a permissive tag can be narrowed by the
# checkpoint card, attached LICENSE, or base-model terms, so it never awards a
# positive score on its own.
OPEN_LICENSE_TAGS = {
    "apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "isc",
    "cc0-1.0", "cc-by-4.0", "unlicense", "mpl-2.0",
}
UNCONFIRMED = {
    "weights": "Downloadable weights are not established by the reviewed source.",
    "data": "Exact training data and release terms are not established by the reviewed source.",
    "code": "Checkpoint-specific training and preprocessing code is not established by the reviewed source.",
    "model_card": "A substantive checkpoint card is not established by the reviewed source.",
    "license": "Effective checkpoint terms are not established by the reviewed source.",
    "paper": "A public paper or technical report for this checkpoint is not established by the reviewed source.",
    "punctuation_case": "Cased, punctuated output is not established for this checkpoint.",
    "timestamps": "Danish word or segment timestamp output is not documented for this checkpoint.",
    "diarization": "Speaker-labeled output is not documented for this checkpoint.",
    "streaming": "Incremental audio streaming is not documented for this checkpoint.",
}

def load_reviews(path: Path = PROFILE_FILE) -> dict:
    reviews = json.loads(path.read_text(encoding="utf-8"))
    for model, fields in reviews.items():
        if set(fields) - {"openness", "features", "report", "reviewed_source", "reviewed_on"}:
            raise ValueError(f"Unknown profile group for {model}")
        source = fields.get("reviewed_source")
        if source is not None and (not isinstance(source, str) or not source.startswith("https://")):
            raise ValueError(f"Invalid review source for {model}")
        if fields.get("reviewed_on"):
            if not fields.get("reviewed_source"):
                raise ValueError(f"Review date needs source for {model}")
            try:
                date.fromisoformat(fields["reviewed_on"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid review date for {model}") from exc
        for group, keys in (("openness", OPENNESS + ADDITIONAL), ("features", FEATURES)):
            for key, decision in fields.get(group, {}).items():
                if key not in keys or decision.get("state") not in STATES:
                    raise ValueError(f"Invalid {group}.{key} for {model}")
                url = decision.get("url", "")
                if url and not url.startswith("https://"):
                    raise ValueError(f"Invalid evidence URL: {model} {group}.{key}")
                if decision["state"] == "yes" and not decision.get("url"):
                    raise ValueError(f"Positive {group}.{key} needs evidence URL: {model}")
        report = fields.get("report", {})
        if report and (report.get("state") not in STATES or
                       (report.get("url") and not report["url"].startswith("https://")) or
                       (report.get("state") == "yes" and not report.get("url"))):
            raise ValueError(f"Invalid report for {model}")
    return reviews


def _candidate(state: str = "unknown", *, detail: str = "", url: str = "", suggestion: str = "") -> dict:
    return {"state": state, "detail": detail, "url": url, "suggestion": suggestion, "reviewed": False}


@functools.lru_cache(maxsize=256)
def formatting_from_outputs(model: str, outputs_dir: Path) -> dict | None:
    """Confirm cased, punctuated text on the scored path from saved raw output.

    A lack of examples is not proof that the model cannot format text, so only
    positive evidence is returned. Other feature fields need endpoint-specific
    documentation or a structured API probe.
    """
    slug = re.sub(r"[^a-zA-Z0-9_-]", "__", model.strip("/"))
    model_dir = outputs_dir / slug
    paths = sorted(model_dir.glob("*.jsonl")) if model_dir.is_dir() else []
    combined = outputs_dir / f"{slug}.jsonl"
    if not paths and combined.is_file():
        paths = [combined]
    if not paths:
        return None
    total = formatted = 0
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                hypothesis = json.loads(line).get("hypothesis") or ""
                if not isinstance(hypothesis, str):
                    continue
                total += 1
                if any(char.isupper() for char in hypothesis) and re.search(r"[.!?,;:]", hypothesis):
                    formatted += 1
    if total < 1000 or formatted < max(50, math.ceil(total * 0.01)):
        return None
    evidence = _candidate(
        "yes",
        detail=f"{formatted:,} of {total:,} saved benchmark transcriptions contain both casing and punctuation.",
        url=f"https://huggingface.co/datasets/RyeAI/danish-asr-leaderboard/blob/main/outputs/{slug}.parquet",
    )
    evidence["reviewed"] = True
    return evidence


def license_tag_class(tag: str) -> str:
    """Classify an advertised license tag as a lead, never as final terms."""
    tag = tag.strip().lower()
    if tag in OPEN_LICENSE_TAGS:
        return "open_candidate"
    if re.search(r"(^|[-_])(nc|non-commercial|noncommercial)([-_]|$)", tag):
        return "noncommercial"
    return "needs_review"


def build_profile(model: str, model_url: str, metadata: dict, reviews: dict,
                  *, access: str = "unknown", formatting_evidence: dict | None = None) -> dict:
    """Build a display profile; Hub tags are review leads, not positive proof."""
    tags = metadata.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    card = metadata.get("cardData") or {}
    if not isinstance(card, dict):
        card = {}
    repo_url = model_url if model_url.startswith("https://huggingface.co/") else ""
    weight_state = "yes" if access == "open" and repo_url else "no" if access == "proprietary" else "unknown"
    weight_candidate = _candidate(
        weight_state,
        detail=("Downloadable model repository" if weight_state == "yes" else
                "Hosted API; weights not downloadable" if weight_state == "no" else
                "Weight availability not reviewed"),
        url=repo_url,
    )
    weight_candidate["reviewed"] = weight_state != "unknown"
    license_tag = next((t.split(":", 1)[1] for t in tags if isinstance(t, str) and t.startswith("license:")), "")
    license_tag = str(card.get("license") or license_tag).lower()
    # A Hub tag is a useful lead, but a model card or a parent-model license can
    # narrow it. Never award a positive license tile without reviewing the terms.
    license_class = license_tag_class(license_tag)
    license_state = "no" if license_class == "noncommercial" else "unknown"
    license_detail = f"Hub license: {license_tag}" if license_tag else "License not reviewed"
    if license_class == "open_candidate":
        license_detail += "; appears open, but checkpoint and base terms need review"
    elif license_tag and license_state == "unknown":
        license_detail += "; effective terms need review"
    license_candidate = _candidate(license_state, detail=license_detail, url=repo_url,
                                   suggestion=license_class)
    license_candidate["reviewed"] = license_state == "no"

    dataset_tags = [t.split(":", 1)[1] for t in tags if isinstance(t, str) and t.startswith("dataset:")]
    card_datasets = card.get("datasets") or []
    if isinstance(card_datasets, str):
        card_datasets = [card_datasets]
    elif not isinstance(card_datasets, (list, tuple)):
        card_datasets = []
    datasets = list(dict.fromkeys([*dataset_tags, *[str(d) for d in card_datasets if d]]))
    first_dataset = next((d for d in datasets if re.fullmatch(r"[\w.-]+/[\w.-]+", d)), "")
    data_candidate = _candidate(
        detail=f"Hub lists {len(datasets)} dataset(s); open access needs review" if datasets else "Open training data not reviewed",
        url=f"https://huggingface.co/datasets/{first_dataset}" if first_dataset else repo_url,
        suggestion="listed" if datasets else "",
    )
    report_candidate = _candidate(
        detail="A paper or technical report for this exact checkpoint has not been verified",
        url=repo_url,
    )
    openness = {
        "weights": weight_candidate,
        "license": license_candidate,
        "data": data_candidate,
        "paper": report_candidate,
        "code": _candidate(detail="Training code not reviewed", url=repo_url),
        "model_card": _candidate(detail="Card completeness not reviewed", url=repo_url),
    }
    features = {key: _candidate(detail="Capability not reviewed") for key in FEATURES}
    if formatting_evidence:
        features["punctuation_case"] = formatting_evidence
    reviewed = reviews.get(model, {})
    if source := reviewed.get("reviewed_source", ""):
        for entries in (openness, features):
            for key, entry in entries.items():
                if entry["state"] == "unknown":
                    entry.update(detail=UNCONFIRMED[key], url=source, reviewed=True)
    for group, entries in (("openness", openness), ("features", features)):
        for key, decision in reviewed.get(group, {}).items():
            entries[key] = {"state": decision["state"],
                            "detail": decision.get("detail", ""),
                            "url": decision.get("url", ""),
                            "suggestion": "", "reviewed": True}
    if "report" in reviewed:
        decision = reviewed["report"]
        report_candidate = {"state": decision["state"],
                            "detail": decision.get("detail", ""),
                            "url": decision.get("url", ""),
                            "suggestion": "", "reviewed": True}
        if "paper" not in reviewed.get("openness", {}):
            openness["paper"] = report_candidate
    else:
        report_candidate = openness["paper"]
    return {"openness": openness, "features": features, "report": report_candidate,
            "openness_score": sum(openness[key]["state"] == "yes" for key in OPENNESS),
            "feature_count": sum(features[key]["state"] == "yes" for key in FEATURES)}
