"""Model profile decisions must distinguish evidence from Hub hints."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from model_profiles import build_profile, load_reviews


REPO = "https://huggingface.co/example/asr"


def test_hub_hints_do_not_claim_open_data_code_or_model_specific_paper():
    profile = build_profile("example/asr", REPO, {
        "tags": ["dataset:example/speech", "arxiv:2401.12345", "license:apache-2.0"],
        "cardData": {"datasets": ["example/speech"]},
    }, {})

    assert profile["openness_score"] == 0
    assert profile["openness"]["license"]["state"] == "unknown"
    assert profile["report"]["state"] == "unknown"
    assert profile["report"]["url"] == "https://arxiv.org/abs/2401.12345"
    assert profile["openness"]["data"]["url"] == "https://huggingface.co/datasets/example/speech"
    assert all(profile["openness"][key]["state"] == "unknown"
               for key in ("data", "code", "model_card"))
    assert profile["feature_count"] == 0


@pytest.mark.parametrize("tag,expected", [
    ("license:mit", "unknown"),
    ("license:cc-by-nc-4.0", "no"),
    ("license:other", "unknown"),
])
def test_license_lookup(tag, expected):
    profile = build_profile("example/asr", REPO, {"tags": [tag]}, {})
    assert profile["openness"]["license"]["state"] == expected


def test_review_overrides_hub_license_and_adds_linked_capabilities():
    reviews = {"example/asr": {
        "openness": {
            "license": {"state": "no", "url": REPO, "detail": "Changed terms"},
            "data": {"state": "yes", "url": "https://example.org/data"},
        },
        "features": {"timestamps": {"state": "yes", "url": "https://example.org/docs"}},
    }}
    profile = build_profile("example/asr", REPO, {"tags": ["license:mit"]}, reviews)
    assert profile["openness_score"] == 1
    assert profile["openness"]["license"]["state"] == "no"
    assert profile["feature_count"] == 1
    assert profile["features"]["timestamps"]["reviewed"] is True


def test_model_specific_report_adds_paper_point():
    reviews = {"example/asr": {
        "report": {"state": "yes", "url": "https://example.org/report"},
        "openness": {"license": {"state": "partial", "url": REPO,
                                 "detail": "Commercial use with field restrictions"}},
    }}
    profile = build_profile("example/asr", REPO, {"tags": ["license:mit"]}, reviews)
    assert profile["report"]["state"] == "yes"
    assert profile["openness"]["paper"]["state"] == "yes"
    assert profile["openness_score"] == 1
    assert profile["openness"]["license"]["state"] == "partial"


def test_downloadable_weights_count_separately_from_license():
    open_model = build_profile("example/asr", REPO, {"tags": ["license:cc-by-nc-4.0"]},
                               {}, access="open")
    api_model = build_profile("api-model", "https://example.org/docs", {}, {},
                              access="proprietary")
    assert open_model["openness"]["weights"]["state"] == "yes"
    assert open_model["openness"]["license"]["state"] == "no"
    assert open_model["openness_score"] == 1
    assert api_model["openness"]["weights"]["state"] == "no"
    assert api_model["openness_score"] == 0


def test_review_requires_evidence_for_positive_claim(tmp_path):
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"example/asr": {
        "openness": {"code": {"state": "yes", "detail": "source available"}}
    }}))
    with pytest.raises(ValueError, match="needs evidence URL"):
        load_reviews(path)


def test_reviewed_source_marks_missing_signals_unknown_without_scoring_them(tmp_path):
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"example/asr": {
        "reviewed_source": REPO,
        "reviewed_on": "2026-09-25",
        "features": {"timestamps": {"state": "yes", "url": REPO}},
    }}))
    reviews = load_reviews(path)
    profile = build_profile("example/asr", REPO, {}, reviews)
    assert profile["openness_score"] == 0
    assert profile["feature_count"] == 1
    assert profile["features"]["diarization"]["state"] == "unknown"
    assert profile["features"]["diarization"]["reviewed"] is True
    assert profile["features"]["diarization"]["url"] == REPO
    assert profile["features"]["timestamps"]["state"] == "yes"
