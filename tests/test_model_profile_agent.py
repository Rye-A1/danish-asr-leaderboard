"""The optional agent must produce auditable suggestions, never score claims."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from model_profile_agent import (FIELDS, RUBRIC, _add_saved_output_source, _card_excerpt,
                                 ask_agent, collect_sources, response_schema,
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


def test_long_hub_front_matter_does_not_hide_checkpoint_evidence():
    card = "---\n" + "dataset: example/speech\n" * 6000 + "---\n"
    card += "# Released checkpoint\nThe training mix uses only public gold transcripts.\n"
    excerpt = _card_excerpt(card)
    assert "Released checkpoint" in excerpt
    assert "only public gold transcripts" in excerpt
    assert "dataset: example/speech" not in excerpt


def test_source_bundle_reads_effective_license_file_and_trusted_link():
    revision = "b" * 40
    source = "https://huggingface.co/example/asr"

    class Response:
        def __init__(self, text="", status=200, data=None):
            self.text, self.status_code, self._data = text, status, data
            self.ok = status == 200

        def raise_for_status(self):
            assert self.ok

        def json(self):
            return self._data

    responses = {
        f"{source}/resolve/{revision}/README.md": Response("# Checkpoint"),
        f"{source}/resolve/{revision}/MODEL_LICENSE.md": Response("Effective checkpoint terms"),
        f"https://huggingface.co/api/models/example/asr/revision/{revision}": Response(
            data={"cardData": {"license_link": "https://www.nvidia.com/en-us/ai/"}}),
        "https://www.nvidia.com/en-us/ai/": Response("Published model terms"),
    }

    class Session:
        def get(self, url, **kwargs):
            return responses.get(url, Response(status=404))

    sources = collect_sources(Session(), {"source": source, "revision": revision})
    assert [(item["kind"], item["text"]) for item in sources] == [
        ("model_card", "# Checkpoint"),
        ("checkpoint_license", "Effective checkpoint terms"),
        ("linked_license", "Published model terms"),
    ]


def test_untrusted_license_link_is_never_fetched():
    revision = "c" * 40
    source = "https://huggingface.co/example/asr"

    class Response:
        status_code = 200
        ok = True
        text = "# Checkpoint"

        def raise_for_status(self):
            pass

        def json(self):
            return {"cardData": {"license_link": "https://untrusted.example/terms"}}

    class Session:
        def get(self, url, **kwargs):
            assert not url.startswith("https://untrusted.example")
            if url.endswith("README.md") or "/api/models/" in url:
                return Response()
            return type("Missing", (), {"status_code": 404})()

    sources = collect_sources(Session(), {"source": source, "revision": revision})
    assert [item["kind"] for item in sources] == ["model_card"]


def test_public_saved_transcripts_supply_only_positive_formatting_evidence(tmp_path):
    model_dir = tmp_path / "example__asr"
    model_dir.mkdir()
    (model_dir / "test.jsonl").write_text(
        ('{"hypothesis": "Hej, verden."}\n' * 1000), encoding="utf-8")
    sources = []
    _add_saved_output_source(sources, "example/asr", tmp_path)
    assert len(sources) == 1
    assert sources[0]["kind"] == "saved_outputs"
    assert "1,000 of 1,000" in sources[0]["text"]
    assert sources[0]["url"].endswith("example__asr.parquet")
    (model_dir / "test.jsonl").write_text(
        ('{"hypothesis": "hej verden"}\n' * 1000), encoding="utf-8")
    from model_profiles import formatting_from_outputs
    formatting_from_outputs.cache_clear()
    sources = []
    _add_saved_output_source(sources, "example/asr", tmp_path)
    assert sources == []


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


def test_whitespace_only_quote_variation_resolves_to_exact_source_span():
    sources = [{"id": "S1", "kind": "model_card", "url": "https://example.org/card",
                "text": "Weights are released under\n  Apache License 2.0."}]
    raw = {"fields": {"license": {"state": "yes", "evidence": [
        {"source_id": "S1", "quote": "Weights are released under Apache License 2.0."}],
        "explanation": "Checkpoint terms."}}}
    field = validate_suggestions(raw, sources)["license"]
    assert field["state"] == "yes"
    assert field["evidence"][0]["quote"] == sources[0]["text"]


def test_stitched_long_quote_is_split_only_when_every_prose_line_matches():
    card = ("# Exact checkpoint\n\nThis model recognizes Danish speech.\n\n"
            "| corpus | rows |\n|---|---|\n| Public set | 100 |\n\n"
            "## Limitations\nNoisy overlapping speech was not evaluated.")
    quote = ("# Exact checkpoint\n\nThis model recognizes Danish speech.\n\n"
             "| corpus | rows |\n|:---|---:|\n| Public set | 100 |\n\n"
             "# ...\n...\n## Limitations\nNoisy overlapping speech was not evaluated."
             + " " * 301)
    sources = [{"id": "S1", "kind": "model_card", "text": card}]
    raw = {"fields": {"model_card": {"state": "yes", "evidence": [
        {"source_id": "S1", "quote": quote}], "explanation": "Complete card."}}}
    field = validate_suggestions(raw, sources)["model_card"]
    assert field["state"] == "yes"
    assert len(field["evidence"]) > 1
    assert all(quote["quote"] in card for quote in field["evidence"])

    raw["fields"]["model_card"]["evidence"][0]["quote"] = quote.replace(
        "This model recognizes Danish speech.", "This model recognizes every language.")
    assert validate_suggestions(raw, sources)["model_card"]["state"] == "unknown"


def test_training_mix_anchor_can_survive_bad_ancillary_dataset_license_quote():
    sources = [
        {"id": "S1", "kind": "model_card", "text":
         "Six public corpora, training splits only: Foo/bar train and Baz/qux train."},
        {"id": "S2", "kind": "dataset_card", "text": "license: other"},
    ]
    raw = {"fields": {"data": {"state": "yes", "evidence": [
        {"source_id": "S1", "quote": sources[0]["text"]},
        {"source_id": "S2", "quote": "license: cc-by-4.0"},
        {"source_id": "S2", "quote": ""}], "explanation": "Complete public mix."}}}
    field = validate_suggestions(raw, sources)["data"]
    assert field["state"] == "yes"
    assert field["evidence"] == [{"source_id": "S1", "quote": sources[0]["text"]}]
    assert "validation_warning" in field
    raw["fields"]["data"]["evidence"][0]["quote"] = "Six private corpora"
    assert validate_suggestions(raw, sources)["data"]["state"] == "unknown"


def test_training_format_does_not_prove_timestamps_are_unsupported():
    sources = [{"id": "S1", "kind": "model_card", "url": "https://example.org/card",
                "text": "Trained on <|notimestamps|> text."}]
    raw = {"fields": {"timestamps": {"state": "no", "evidence": [
        {"source_id": "S1", "quote": "Trained on <|notimestamps|> text."}],
        "explanation": "The format has no timestamp token."}}}
    assert validate_suggestions(raw, sources)["timestamps"]["state"] == "unknown"


def test_unevaluated_overlapping_speech_does_not_prove_no_diarization():
    sources = [{"id": "S1", "kind": "model_card", "url": "https://example.org/card",
                "text": "Noisy audio and overlapping speakers were not evaluated separately.\nNo speaker diarization."}]
    speculative = {"fields": {"diarization": {"state": "no", "evidence": [
        {"source_id": "S1", "quote": "Noisy audio and overlapping speakers were not evaluated separately."}],
        "explanation": "Not evaluated."}}}
    explicit = {"fields": {"diarization": {"state": "no", "evidence": [
        {"source_id": "S1", "quote": "No speaker diarization."}],
        "explanation": "Explicitly absent."}}}
    assert validate_suggestions(speculative, sources)["diarization"]["state"] == "unknown"
    assert validate_suggestions(explicit, sources)["diarization"]["state"] == "no"


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
