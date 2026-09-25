"""Evidence-linked openness and capability metadata for leaderboard models."""
from __future__ import annotations

import json
import re
from pathlib import Path

PROFILE_FILE = Path(__file__).with_name("model_profiles.json")
OPENNESS = ("data", "code", "paper", "model_card", "license")
FEATURES = ("punctuation_case", "timestamps", "diarization", "streaming")
STATES = {"yes", "no", "partial", "unknown"}

# Standard licenses that allow commercial use and redistribution. Custom,
# gated and unfamiliar terms require a manual review.
OPEN_LICENSES = {
    "apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "isc", "cc0-1.0",
    "cc-by-4.0", "cc-by-sa-4.0", "unlicense", "mpl-2.0", "epl-2.0",
    "gpl-3.0", "lgpl-3.0", "agpl-3.0",
}


def load_reviews(path: Path = PROFILE_FILE) -> dict:
    reviews = json.loads(path.read_text(encoding="utf-8"))
    for model, fields in reviews.items():
        if set(fields) - {"openness", "features"}:
            raise ValueError(f"Unknown profile group for {model}")
        for group, keys in (("openness", OPENNESS), ("features", FEATURES)):
            for key, decision in fields.get(group, {}).items():
                if key not in keys or decision.get("state") not in STATES:
                    raise ValueError(f"Invalid {group}.{key} for {model}")
                url = decision.get("url", "")
                if url and not url.startswith("https://"):
                    raise ValueError(f"Invalid evidence URL: {model} {group}.{key}")
                if decision["state"] == "yes" and not decision.get("url"):
                    raise ValueError(f"Positive {group}.{key} needs evidence URL: {model}")
    return reviews


def _candidate(state: str = "unknown", *, detail: str = "", url: str = "", suggestion: str = "") -> dict:
    return {"state": state, "detail": detail, "url": url, "suggestion": suggestion, "reviewed": False}


def build_profile(model: str, model_url: str, metadata: dict, reviews: dict) -> dict:
    """Build a display profile; Hub tags do not prove code or data openness."""
    tags = metadata.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    card = metadata.get("cardData") or {}
    if not isinstance(card, dict):
        card = {}
    repo_url = model_url if model_url.startswith("https://huggingface.co/") else ""
    license_tag = next((t.split(":", 1)[1] for t in tags if isinstance(t, str) and t.startswith("license:")), "")
    license_tag = str(card.get("license") or license_tag).lower()
    license_state = "yes" if license_tag in OPEN_LICENSES else "unknown"
    if re.search(r"(^|[-_])(nc|non-commercial|noncommercial)([-_]|$)", license_tag):
        license_state = "no"
    license_detail = f"Hub license: {license_tag}" if license_tag else "License not reviewed"
    if license_tag and license_state == "unknown":
        license_detail += "; terms need review"
    license_candidate = _candidate(license_state, detail=license_detail, url=repo_url)
    license_candidate["reviewed"] = license_state != "unknown"

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
    arxiv = next((t.split(":", 1)[1] for t in tags if isinstance(t, str) and t.startswith("arxiv:")), "")
    paper_url = f"https://arxiv.org/abs/{arxiv}" if re.fullmatch(r"\d{4}\.\d{4,5}", arxiv) else ""
    paper_candidate = _candidate(
        detail="Paper link in Hub metadata; confirm it describes this model" if paper_url else "Paper or report not reviewed",
        url=paper_url, suggestion="linked" if paper_url else "",
    )
    openness = {
        "data": data_candidate,
        "code": _candidate(detail="Training code not reviewed", url=repo_url),
        "paper": paper_candidate,
        "model_card": _candidate(detail="Card completeness not reviewed", url=repo_url),
        "license": license_candidate,
    }
    features = {key: _candidate(detail="Capability not reviewed") for key in FEATURES}
    reviewed = reviews.get(model, {})
    for group, entries in (("openness", openness), ("features", features)):
        for key, decision in reviewed.get(group, {}).items():
            entries[key] = {"state": decision["state"],
                            "detail": decision.get("detail", ""),
                            "url": decision.get("url", ""),
                            "suggestion": "", "reviewed": True}
    return {"openness": openness, "features": features,
            "openness_score": sum(openness[key]["state"] == "yes" for key in OPENNESS),
            "feature_count": sum(features[key]["state"] == "yes" for key in FEATURES)}
