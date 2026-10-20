"""Paid Mistral requests must be correct, resumable and fail fast."""
from types import SimpleNamespace

import pytest

from danish_asr_leaderboard.backends.base import LoadOptions
from danish_asr_leaderboard.backends.api.mistral import MistralBackend, load
from danish_asr_leaderboard.results import model_link


class FakeTranscriptions:
    def __init__(self, *, fail_on=()):
        self.calls = []
        self.fail_on = set(fail_on)

    def complete(self, *, model, file, language):
        content = file["content"].read()
        self.calls.append((model, file["file_name"], content, language))
        if content in self.fail_on:
            raise RuntimeError("insufficient credit")
        return SimpleNamespace(
            model=model,
            text=f"  heard {content.decode()}  ",
            usage=SimpleNamespace(prompt_audio_seconds=2),
        )


def backend(tmp_path, fake, *, model="voxtral-mini-transcribe-3-0"):
    client = SimpleNamespace(audio=SimpleNamespace(transcriptions=fake))
    options = LoadOptions(mistral_cache=str(tmp_path / "paid.jsonl"), mistral_concurrency=1,
                          mistral_rpm=0)
    return MistralBackend(client, model, options=options)


def test_versioned_request_and_cache_resume(tmp_path):
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"one")
    fake = FakeTranscriptions()
    first = backend(tmp_path, fake)
    assert first.transcribe([str(wav)], batch_size=16) == ["heard one"]
    assert fake.calls == [("voxtral-mini-transcribe-3-0", "clip.wav", b"one", "da")]
    assert first.new_requests == 1
    assert first.reported_audio_seconds == 2

    resumed = backend(tmp_path, fake)
    assert resumed.transcribe([str(wav)], batch_size=16) == ["heard one"]
    assert len(fake.calls) == 1
    assert resumed.new_requests == 0

    wav.write_bytes(b"changed")
    assert resumed.transcribe([str(wav)], batch_size=16) == ["heard changed"]
    assert len(fake.calls) == 2


def test_first_failure_stops_before_other_paid_calls(tmp_path):
    files = []
    for n in range(4):
        wav = tmp_path / f"{n}.wav"
        wav.write_bytes(str(n).encode())
        files.append(str(wav))
    fake = FakeTranscriptions(fail_on={b"0"})
    with pytest.raises(RuntimeError, match="insufficient credit"):
        backend(tmp_path, fake).transcribe(files, batch_size=16)
    assert len(fake.calls) == 1


def test_successes_survive_later_failure(tmp_path):
    files = []
    for n in range(3):
        wav = tmp_path / f"{n}.wav"
        wav.write_bytes(str(n).encode())
        files.append(str(wav))
    fake = FakeTranscriptions(fail_on={b"1"})
    first = backend(tmp_path, fake)
    with pytest.raises(RuntimeError, match="insufficient credit"):
        first.transcribe(files, batch_size=16)
    assert backend(tmp_path, fake).transcribe_one(files[0]) == "heard 0"
    assert sum(call[2] == b"0" for call in fake.calls) == 1


def test_requires_cache_and_secret(tmp_path, monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="mistral-cache"):
        load("voxtral-mini-transcribe-3-0", LoadOptions())
    with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
        load("voxtral-mini-transcribe-3-0",
             LoadOptions(mistral_cache=str(tmp_path / "paid.jsonl")))


def test_results_link_to_vendor_page():
    assert model_link("voxtral-small-transcribe-3-0") == (
        "[voxtral-small-transcribe-3-0]"
        "(https://docs.mistral.ai/models/voxtral-small-transcribe-3-0)"
    )
