"""Check the language parameter sent to each Azure transcription deployment."""

from types import SimpleNamespace

import pytest

from danish_asr_leaderboard.backends.api.azure_openai import (
    AzureOpenAIBackend,
    _resource_endpoint,
)


@pytest.mark.parametrize(
    ("deployment", "expected_hint"),
    [
        ("gpt-transcribe", {"languages": ["da"]}),
        ("gpt-4o-transcribe-benchmark", {"language": "da"}),
        ("gpt-4o-mini-transcribe-benchmark", {"language": "da"}),
    ],
)
def test_transcription_requests_danish(tmp_path, deployment, expected_hint):
    calls = []

    def create(**kwargs):
        calls.append({key: value for key, value in kwargs.items() if key != "file"})
        assert kwargs["file"].read() == b"sample audio"
        return SimpleNamespace(text="  dansk tale  ")

    client = SimpleNamespace(
        audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create))
    )
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"sample audio")

    backend = AzureOpenAIBackend(client, deployment)
    assert backend.transcribe_one(str(audio_path)) == "dansk tale"
    assert calls == [{"model": deployment, **expected_hint}]


def test_azure_resource_url_does_not_duplicate_deployment_path():
    assert _resource_endpoint(
        "https://example.openai.azure.com/openai/v1/"
    ) == "https://example.openai.azure.com"
    assert _resource_endpoint(
        "https://example.cognitiveservices.azure.com/openai/audio"
    ) == "https://example.cognitiveservices.azure.com"
    assert _resource_endpoint("https://proxy.example.com/custom/path") == (
        "https://proxy.example.com/custom/path"
    )
