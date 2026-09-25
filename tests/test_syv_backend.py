"""Unit tests for the syv.ai backend (no network, no openai SDK needed)."""
import json
import threading

import pytest

from danish_asr_leaderboard.backends.api.syv import SyvBackend, load
from danish_asr_leaderboard.backends.base import LoadOptions


class _Raw:
    def __init__(self, text, headers):
        self._text, self.headers = text, headers

    def parse(self):
        return type("Resp", (), {"text": self._text})()


class _FakeClient:
    """Mimics ``client.audio.transcriptions.with_raw_response.create``."""

    def __init__(self, served_model="syv-transcribe"):
        self.calls = []
        self.served_model = served_model
        self._lock = threading.Lock()
        self.audio = self
        self.transcriptions = self
        self.with_raw_response = self

    def create(self, *, model, file):
        with self._lock:
            self.calls.append((model, file.name))
        return _Raw(f"tekst {file.name.rsplit('/', 1)[-1]}",
                    {"x-model": self.served_model, "x-credits-charged": "3.0"})


def _clips(tmp_path, n):
    paths = []
    for i in range(n):
        p = tmp_path / f"{i}.wav"
        p.write_bytes(b"RIFF")
        paths.append(str(p))
    return paths


def test_cache_means_a_rerun_pays_only_for_missing_clips(tmp_path):
    clips = _clips(tmp_path, 4)
    cache = str(tmp_path / "cache.jsonl")

    first = _FakeClient()
    out1 = SyvBackend(first, "syv-transcribe", concurrency=2, cache_path=cache) \
        .transcribe(clips[:3], batch_size=1)
    assert len(first.calls) == 3

    second = _FakeClient()
    backend = SyvBackend(second, "syv-transcribe", concurrency=2, cache_path=cache)
    out2 = backend.transcribe(clips, batch_size=1)
    assert second.calls == [("syv-transcribe", clips[3])]  # only the new clip is requested
    assert out2[:3] == out1 and out2[3] == "tekst 3.wav"
    assert backend.credits_charged == 3.0
    rows = [json.loads(l) for l in open(cache, encoding="utf-8")]
    assert len(rows) == 4 and {r["model"] for r in rows} == {"syv-transcribe"}
    assert {r["request_mode"] for r in rows} == {"default"}
    assert {r["requested_model"] for r in rows} == {"syv-transcribe"}


def test_a_different_served_model_is_not_scored_or_cached(tmp_path):
    clips = _clips(tmp_path, 2)
    cache = tmp_path / "cache.jsonl"
    backend = SyvBackend(_FakeClient(served_model="syv-other"), "syv-transcribe",
                         concurrency=1, cache_path=str(cache))
    with pytest.raises(RuntimeError, match="failed for all"):
        backend.transcribe(clips, batch_size=1)
    assert not cache.exists()


def test_missing_key_is_reported_before_importing_the_sdk(monkeypatch):
    monkeypatch.delenv("SYV_API_KEY", raising=False)
    with pytest.raises(ValueError, match="SYV_API_KEY"):
        load("syv-transcribe", LoadOptions())


def test_only_model_and_file_are_sent_and_legacy_cache_is_ignored(tmp_path):
    clips = _clips(tmp_path, 1)
    cache = str(tmp_path / "cache.jsonl")
    with open(cache, "w", encoding="utf-8") as f:
        f.write(json.dumps({"audio": clips[0], "text": "old VAD output",
                            "model": "syv-transcribe", "vad": False}) + "\n")
    client = _FakeClient()
    SyvBackend(client, "syv-transcribe", concurrency=1, cache_path=cache) \
        .transcribe(clips, batch_size=1)
    assert client.calls == [("syv-transcribe", clips[0])]


def test_cache_does_not_cross_requested_models(tmp_path):
    clip = _clips(tmp_path, 1)[0]
    cache = str(tmp_path / "cache.jsonl")
    with open(cache, "w", encoding="utf-8") as f:
        f.write(json.dumps({"audio": clip, "text": "wrong model",
                            "requested_model": "syv-other", "request_mode": "default"}) + "\n")
    client = _FakeClient()
    SyvBackend(client, "syv-transcribe", concurrency=1, cache_path=cache) \
        .transcribe([clip], batch_size=1)
    assert client.calls == [("syv-transcribe", clip)]
