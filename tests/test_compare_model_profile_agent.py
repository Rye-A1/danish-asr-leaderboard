"""The agent comparison reports differences without changing reviewed scores."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from compare_model_profile_agent import compare, markdown


def test_comparison_uses_effective_proprietary_policy_and_shows_disagreements():
    records = {"hosted": {"source": "https://example.org/api", "access": "proprietary"},
               "open/asr": {"source": "https://huggingface.co/open/asr", "access": "open"}}
    reviews = {"hosted": {"features": {"timestamps": {
        "state": "yes", "url": "https://example.org/api"}}},
               "open/asr": {"openness": {"license": {
                   "state": "yes", "url": "https://huggingface.co/open/asr/blob/main/LICENSE"}}}}
    suggestions = {"hosted": {"fields": {
        "license": {"state": "no"}, "data": {"state": "no"},
        "code": {"state": "no"}, "model_card": {"state": "no"},
        "timestamps": {"state": "unknown"}}},
                   "open/asr": {"fields": {"license": {"state": "yes"}}}}
    report = compare(suggestions, reviews, records)
    assert report["models_compared"] == 2
    assert report["fields_compared"] == 16
    assert report["exact_matches"] == 15
    text = markdown(report)
    assert "| `hosted` | timestamps | yes | unknown |" in text
    assert "| `hosted` | license |" not in text
