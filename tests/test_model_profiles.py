"""Model profile decisions must distinguish evidence from Hub hints."""
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from model_profiles import (build_profile, formatting_from_outputs, load_reviews,
                            license_tag_class, OPENNESS, FEATURES)
from refresh_model_profile_candidates import collect, evidence_leads, needs_review


REPO = "https://huggingface.co/example/asr"


def test_hub_hints_do_not_claim_open_data_code_or_model_specific_paper():
    profile = build_profile("example/asr", REPO, {
        "tags": ["dataset:example/speech", "arxiv:2401.12345", "license:apache-2.0"],
        "cardData": {"datasets": ["example/speech"]},
    }, {})

    assert profile["openness_score"] == 0
    assert profile["openness"]["license"]["state"] == "unknown"
    assert profile["report"]["state"] == "unknown"
    assert profile["report"]["url"] == REPO
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
    profile = build_profile("example/asr", REPO, {"tags": ["license:cc-by-4.0"]}, reviews)
    assert profile["openness_score"] == 1
    assert profile["openness"]["license"]["state"] == "no"
    assert profile["feature_count"] == 1
    assert profile["features"]["timestamps"]["reviewed"] is True


def test_model_specific_report_is_unscored_without_model_card():
    reviews = {"example/asr": {
        "report": {"state": "yes", "url": "https://example.org/report"},
        "openness": {"license": {"state": "partial", "url": REPO,
                                 "detail": "Commercial use with field restrictions"}},
    }}
    profile = build_profile("example/asr", REPO, {"tags": ["license:mit"]}, reviews)
    assert profile["report"]["state"] == "yes"
    assert profile["openness"]["paper"]["state"] == "yes"
    assert profile["openness_score"] == 0
    assert profile["openness"]["license"]["state"] == "partial"


def test_substantive_model_card_and_training_code_are_scored():
    reviews = {"example/asr": {"openness": {
        "model_card": {"state": "yes", "url": REPO,
                       "detail": "Checkpoint-specific training and use documented"},
        "code": {"state": "yes", "url": "https://example.org/train.py",
                 "detail": "Checkpoint-specific training script"},
    }}}
    profile = build_profile("example/asr", REPO, {}, reviews)
    assert profile["openness_score"] == 2


def test_downloadable_weights_count_separately_from_license():
    open_model = build_profile("example/asr", REPO, {"tags": ["license:cc-by-nc-4.0"]},
                               {}, access="open")
    api_model = build_profile("api-model", "https://example.org/docs", {}, {},
                              access="proprietary")
    assert open_model["openness"]["weights"]["state"] == "yes"
    assert open_model["openness"]["license"]["state"] == "no"
    assert open_model["openness_score"] == 1
    assert api_model["openness"]["weights"]["state"] == "no"
    assert all(api_model["openness"][key]["state"] == "no" for key in OPENNESS)
    assert api_model["openness_score"] == 0


def test_proprietary_openness_policy_overrides_review_without_changing_features():
    reviews = {"api-model": {
        "openness": {
            "data": {"state": "yes", "url": "https://example.org/data"},
            "model_card": {"state": "yes", "url": "https://example.org/card"},
        },
        "report": {"state": "yes", "url": "https://example.org/paper"},
        "features": {"timestamps": {"state": "yes", "url": "https://example.org/docs"}},
    }}
    profile = build_profile("api-model", "https://example.org/docs", {}, reviews,
                            access="proprietary")
    assert profile["openness_score"] == 0
    assert all(profile["openness"][key]["state"] == "no" for key in OPENNESS)
    assert all("policy" in profile["openness"][key]["detail"] for key in OPENNESS)
    assert profile["features"]["timestamps"]["state"] == "yes"
    assert profile["feature_count"] == 1
    assert profile["report"]["state"] == "yes"


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


def test_new_leaderboard_models_get_complete_unknown_safe_profiles():
    results = Path(__file__).resolve().parent.parent / "results"
    names = set()
    for path in results.glob("*.json"):
        model = json.loads(path.read_text())["model"]
        match = re.match(r"\[([^]]+)\]", model)
        if match:
            names.add(match.group(1))
    reviews = load_reviews()
    for name in names:
        profile = build_profile(name, f"https://huggingface.co/{name}", {}, reviews,
                                access="open")
        assert set(OPENNESS) <= profile["openness"].keys()
        assert set(FEATURES) <= profile["features"].keys()
        assert profile["openness_score"] == sum(
            profile["openness"][key]["state"] == "yes" for key in OPENNESS)


def test_license_lookup_is_a_lead_not_a_positive_score():
    assert license_tag_class("apache-2.0") == "open_candidate"
    assert license_tag_class("cc-by-nc-4.0") == "noncommercial"
    assert license_tag_class("custom") == "needs_review"
    profile = build_profile("example/asr", REPO,
                            {"tags": ["license:apache-2.0"]}, {})
    assert profile["openness"]["license"]["state"] == "unknown"
    assert "appears open" in profile["openness"]["license"]["detail"]


def test_candidate_collector_does_not_turn_tags_into_scores():
    info = {
        "tags": ["license:mit", "dataset:example/speech", "arxiv:2401.12345"],
        "cardData": {"license": "mit", "datasets": ["example/speech"]},
        "siblings": [{"rfilename": "model.safetensors"},
                     {"rfilename": "train.py"},
                     {"rfilename": "preprocessor_config.json"}],
        "lastModified": "2026-10-09T00:00:00Z", "sha": "abc", "gated": False,
    }
    candidate = evidence_leads("example/asr", REPO, info,
                               "# Model\n## Limitations\nhttps://github.com/example/asr/blob/main/train.py")
    assert candidate["license_tag_class"] == "open_candidate"
    assert candidate["dataset_links"] == ["https://huggingface.co/datasets/example/speech"]
    assert candidate["unverified_paper_links"] == ["https://arxiv.org/abs/2401.12345"]
    assert candidate["weight_files"] == ["model.safetensors"]
    assert len(candidate["training_code_leads"]) == 2
    assert not any("preprocessor_config" in url for url in candidate["training_code_leads"])
    assert needs_review({"reviewed_on": "2026-10-08"}, info)
    assert not needs_review({"reviewed_on": "2026-10-09"}, info)


def test_collector_queues_new_models_and_changed_sources(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "new.json").write_text(json.dumps({
        "model": "[example/new](https://huggingface.co/example/new)"}))
    (results / "api.json").write_text(json.dumps({
        "model": "[hosted-api](https://example.org/docs)"}))
    reviews = tmp_path / "reviews.json"
    reviews.write_text("{}")

    class Response:
        ok = True
        status_code = 200
        text = "# New model\n## Limitations"

        def json(self):
            return {"tags": ["license:mit"], "siblings": [],
                    "lastModified": "2026-10-09T00:00:00Z"}

    class Session:
        def get(self, url, **kwargs):
            assert url in {"https://huggingface.co/api/models/example/new",
                           "https://huggingface.co/example/new/resolve/main/README.md"}
            return Response()

    candidates = collect(Session(), results=results, reviews_path=reviews)
    assert set(candidates) == {"example/new", "hosted-api"}
    assert candidates["example/new"]["license_tag_class"] == "open_candidate"
    assert candidates["hosted-api"]["status"] == "new"


def test_saved_outputs_confirm_formatting_without_overriding_review(tmp_path):
    model_dir = tmp_path / "example__asr"
    model_dir.mkdir()
    rows = [json.dumps({"hypothesis": "Hej, verden." if i < 80 else "hej verden"})
            for i in range(1000)]
    (model_dir / "sample.jsonl").write_text("\n".join(rows) + "\n")
    evidence = formatting_from_outputs("example/asr", tmp_path)
    assert evidence["state"] == "yes"
    assert "80 of 1,000" in evidence["detail"]
    profile = build_profile("example/asr", REPO, {}, {}, formatting_evidence=evidence)
    assert profile["feature_count"] == 1
    reviewed = {"example/asr": {"features": {"punctuation_case": {
        "state": "no", "url": REPO, "detail": "Publisher documents no formatted mode"}}}}
    profile = build_profile("example/asr", REPO, {}, reviewed,
                            formatting_evidence=evidence)
    assert profile["feature_count"] == 0
    assert profile["features"]["punctuation_case"]["state"] == "no"


def test_absence_of_observed_output_is_not_documented_no():
    reviews = load_reviews()
    assert reviews["CoRal-project/roest-v2-wav2vec2-1B"]["features"]["punctuation_case"]["state"] == "unknown"
    assert reviews["Qwen/Qwen3-ASR-1.7B"]["features"]["timestamps"]["state"] == "unknown"
    assert reviews["danish-foundation-models/edda-v0.1"]["features"]["streaming"]["state"] == "unknown"
    assert reviews["3dio-ai/svale-110M"]["features"]["punctuation_case"]["state"] == "no"
