"""The optional agent must produce auditable suggestions, never score claims."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from model_profile_agent import (FIELDS, RUBRIC, ask_agent, collect_sources, response_schema,
                                 validate_suggestions)


def test_agent_rubric_covers_checkpoint_data_and_all_scored_capabilities():
    assert set(response_schema()["properties"]["fields"]["required"]) == set(FIELDS)
    for text in ("exact named checkpoint", "inherited base terms", "responsible-use conditions",
                 "COMPLETE training or fine-tuning",
                 "publicly obtainable", "pseudo-labeled", "direct link", "preprocessing code",
                 "Danish word or segment timestamps",
                 "response schema", "model compatibility list", "WHILE live audio arrives"):
        assert text in RUBRIC


def test_source_bundle_fetches_pinned_card_dataset_code_and_base_terms():
    revision = "a" * 40
    source = "https://huggingface.co/example/asr"
    candidate = {
        "source": source, "revision": revision,
        "dataset_links": ["https://huggingface.co/datasets/example/speech"],
        "training_code_leads": ["https://github.com/example/asr/blob/main/train.py",
                                "https://untrusted.example/train.py"],
    }

    class Response:
        def __init__(self, text="", status=200, data=None):
            self.text, self.status_code, self._data = text, status, data
            self.ok = status == 200

        def raise_for_status(self):
            assert self.ok

        def json(self):
            return self._data

    responses = {
        f"{source}/resolve/{revision}/README.md": Response("Exact model card"),
        f"{source}/resolve/{revision}/LICENSE": Response("Checkpoint terms"),
        "https://huggingface.co/datasets/example/speech/resolve/main/README.md": Response("Dataset terms"),
        "https://raw.githubusercontent.com/example/asr/main/train.py": Response("Training script"),
        f"https://huggingface.co/api/models/example/asr/revision/{revision}": Response(
            data={"cardData": {"base_model": "example/base"}}),
        "https://huggingface.co/example/base/resolve/main/README.md": Response("Base card"),
        "https://huggingface.co/example/base/resolve/main/LICENSE": Response("Base terms"),
    }

    class Session:
        def get(self, url, **kwargs):
            assert not url.startswith("https://untrusted.example")
            return responses.get(url, Response(status=404))

    sources = collect_sources(Session(), candidate)
    assert [item["kind"] for item in sources] == [
        "model_card", "checkpoint_license", "dataset_card", "training_code",
        "base_model_card", "base_license"]
    assert sources[0]["url"].endswith(f"/{revision}/README.md")
    assert [item["id"] for item in sources] == [f"S{i}" for i in range(1, 7)]


def test_provider_review_uses_only_official_api_docs_and_visible_main_text():
    class Response:
        status_code = 200
        ok = True
        headers = {"Content-Type": "text/html"}
        text = ("<html><nav>Unrelated models</nav><main><h1>Speech API</h1>"
                "<p>gpt-transcribe supports punctuated output.</p></main>"
                "<script>secret-looking page script</script></html>")

        def raise_for_status(self):
            pass

    class Session:
        def get(self, url, **kwargs):
            assert url == "https://developers.openai.com/api/docs/guides/speech-to-text"
            return Response()

    sources = collect_sources(Session(), {"source_kind": "provider_api", "provider_docs": [
        "https://developers.openai.com/api/docs/guides/speech-to-text",
        "https://example.org/private", "http://developers.openai.com/insecure"]})
    assert len(sources) == 1
    assert sources[0]["kind"] == "provider_api"
    assert "gpt-transcribe supports punctuated output" in sources[0]["text"]
    assert "Unrelated models" not in sources[0]["text"]


def test_fabricated_quotes_and_missing_fields_cannot_pass_validation():
    sources = [{"id": "S1", "kind": "model_card", "url": "https://example.org/card",
                "text": "Supports Danish word timestamps."},
               {"id": "S2", "kind": "base_license", "url": "https://example.org/license",
                "text": "Commercial use is permitted."}]
    raw = {"fields": {
        "timestamps": {"state": "yes", "evidence": [
            {"source_id": "S1", "quote": "Supports Danish word timestamps."}],
                       "explanation": "Exact model."},
        "license": {"state": "yes", "evidence": [
            {"source_id": "S2", "quote": "Commercial use is permitted."},
            {"source_id": "S1", "quote": "Apache 2.0 commercial license"}],
                    "explanation": "One quote was invented."},
        "streaming": {"state": "no", "evidence": [
            {"source_id": "S9", "quote": "No streaming"}],
                      "explanation": "Wrong source."},
    }}
    fields = validate_suggestions(raw, sources)
    assert fields["timestamps"]["state"] == "yes"
    assert fields["timestamps"]["evidence"][0]["source_id"] == "S1"
    assert fields["license"]["state"] == "unknown"
    assert fields["streaming"]["state"] == "unknown"
    assert fields["data"]["state"] == "unknown"


def test_training_format_does_not_prove_timestamps_are_unsupported():
    sources = [{"id": "S1", "kind": "model_card", "url": "https://example.org/card",
                "text": "Trained on <|notimestamps|> text."}]
    raw = {"fields": {"timestamps": {"state": "no", "evidence": [
        {"source_id": "S1", "quote": "Trained on <|notimestamps|> text."}],
        "explanation": "The format has no timestamp token."}}}
    assert validate_suggestions(raw, sources)["timestamps"]["state"] == "unknown"


def test_agent_request_uses_structured_output_and_source_bundle():
    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"tool_calls": [{"function": {
                "name": "submit_profile_review",
                "arguments": json.dumps({"fields": {}}),
            }}]}}]}

    class Session:
        def post(self, url, *, headers, json, timeout):
            assert url == "https://openrouter.ai/api/v1/chat/completions"
            assert headers == {"Authorization": "Bearer example-key"}
            assert json["tools"][0]["function"]["name"] == "submit_profile_review"
            assert json["tool_choice"]["function"]["name"] == "submit_profile_review"
            assert json["provider"] == {"only": ["nvidia"],
                                         "allow_fallbacks": False,
                                         "require_parameters": True}
            assert json["reasoning"] == {"enabled": False}
            assert "Exact checkpoint card" in json["messages"][1]["content"]
            assert timeout == 120
            return Response()

    result = ask_agent(Session(), "example-key", "example/asr", {"revision": "a" * 40},
                       [{"id": "S1", "kind": "model_card", "url": "https://example.org/card",
                         "text": "Exact checkpoint card"}])
    assert result == {"fields": {}}
