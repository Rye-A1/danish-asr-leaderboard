"""The reviewer must stay within openness and reject invented quotes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from openness_reviewer_agent import FIELDS, RUBRIC, response_schema, validate_suggestions


def test_only_openness_fields_are_requested():
    assert FIELDS == ("license", "training_data", "training_code", "model_card")
    assert response_schema()["properties"]["fields"]["required"] == list(FIELDS)
    assert "pseudo-labeled" in RUBRIC
    assert "inherited base terms" in RUBRIC
    assert "Capabilities" not in RUBRIC


def test_unquoted_claim_is_rejected():
    sources = [{"id": "S1", "kind": "model_card", "url": "https://example.test/card",
                "text": "This checkpoint has a model card."}]
    raw = {"fields": {
        "model_card": {"state": "yes", "evidence": [
            {"source_id": "S1", "quote": "This checkpoint has a model card."}],
            "explanation": "Exact quote"},
        "license": {"state": "yes", "evidence": [
            {"source_id": "S1", "quote": "Commercial use is allowed."}],
            "explanation": "Invented"},
    }}
    fields = validate_suggestions(raw, sources)
    assert fields["model_card"]["state"] == "yes"
    assert fields["license"]["state"] == "unknown"
