"""Local profile pilot saves a comparison, never its OpenRouter key."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import run_model_profile_pilot as pilot


def test_local_pilot_saves_only_public_review_material(tmp_path, monkeypatch):
    record = {"example/asr": {"source": "https://huggingface.co/example/asr",
                              "access": "open"}}
    monkeypatch.setattr(pilot, "model_records", lambda: record)
    monkeypatch.setattr(pilot, "load_reviews", lambda: {})
    monkeypatch.setattr(pilot, "collect", lambda session, only_models: {
        "example/asr": {"source": record["example/asr"]["source"]}})
    monkeypatch.setattr(pilot, "collect_sources", lambda session, candidate: [{
        "id": "S1", "kind": "model_card", "url": candidate["source"], "text": "# ASR"}])
    monkeypatch.setattr(pilot, "_add_saved_output_source", lambda sources, name: None)
    monkeypatch.setattr(pilot, "ask_agent", lambda session, key, name, candidate, sources: {
        "fields": {"license": {"state": "unknown", "evidence": [], "explanation": ""}}})
    monkeypatch.setattr(pilot, "validate_suggestions", lambda raw, sources: raw["fields"])
    report = pilot.run({"example/asr"}, "private-test-key", output_dir=tmp_path)
    assert report["models_compared"] == 1
    assert (tmp_path / "suggestions.json").exists()
    assert (tmp_path / "comparison.md").exists()
    assert "private-test-key" not in "".join(
        path.read_text() for path in tmp_path.iterdir())
