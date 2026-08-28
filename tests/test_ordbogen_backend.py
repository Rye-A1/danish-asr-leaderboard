"""Unit tests for the Ordbogen backend (no network, no openai SDK needed)."""
import threading
import time

import pytest

from danish_asr_leaderboard.backends.base import LoadOptions
from danish_asr_leaderboard.backends.api._base import _Pacer
from danish_asr_leaderboard.backends.api.ordbogen import (
    DEFAULT_RPM,
    OrdbogenBackend,
    load,
)


# --------------------------------------------------------------------------
# Pacer
# --------------------------------------------------------------------------

def test_pacer_spaces_calls():
    """Ten calls at 600 rpm (100 ms apart) cannot finish faster than ~0.9 s."""
    pacer = _Pacer(600)
    start = time.monotonic()
    for _ in range(10):
        pacer.wait()
    assert time.monotonic() - start >= 0.85


def test_pacer_zero_rpm_is_a_noop():
    """rpm<=0 disables pacing rather than dividing by zero."""
    pacer = _Pacer(0)
    start = time.monotonic()
    for _ in range(50):
        pacer.wait()
    assert time.monotonic() - start < 0.1


def test_pacer_is_thread_safe():
    """Concurrent waiters must still be spaced: the pacer is shared by the pool."""
    pacer = _Pacer(600)          # 100 ms apart
    start = time.monotonic()
    threads = [threading.Thread(target=pacer.wait) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 8 slots at 100 ms => the last one starts no earlier than 0.7 s in.
    assert time.monotonic() - start >= 0.6


# --------------------------------------------------------------------------
# Batch behaviour
# --------------------------------------------------------------------------

class _FakeTranscriptions:
    def __init__(self, table, fail_on=()):
        self.table, self.fail_on = table, set(fail_on)
        self.calls = []
        self._lock = threading.Lock()

    def create(self, *, model, file, language):
        name = getattr(file, "name", "")
        with self._lock:
            self.calls.append((model, language, name))
        if name in self.fail_on:
            raise RuntimeError(f"boom: {name}")
        return type("R", (), {"text": self.table[name]})()


class _FakeClient:
    def __init__(self, table, fail_on=()):
        self.audio = type("A", (), {"transcriptions": _FakeTranscriptions(table, fail_on)})()


def _backend(tmp_path, texts, fail_on=(), concurrency=4):
    paths = []
    table = {}
    for i, text in enumerate(texts):
        p = tmp_path / f"{i}.wav"
        p.write_bytes(b"RIFF")
        paths.append(str(p))
        table[str(p)] = text
    fail = {str(tmp_path / f"{i}.wav") for i in fail_on}
    client = _FakeClient(table, fail)
    return OrdbogenBackend(client, "ordbogen/whisper", rpm=0, concurrency=concurrency), paths


def test_batch_preserves_order(tmp_path):
    """The pool completes out of order; results must still line up with inputs."""
    texts = [f"utterance {i}" for i in range(20)]
    backend, paths = _backend(tmp_path, texts)
    assert backend.transcribe_batch(paths, batch_size=16) == texts


def test_batch_tolerates_individual_failures(tmp_path):
    backend, paths = _backend(tmp_path, ["a", "b", "c", "d"], fail_on=(1, 2))
    assert backend.transcribe_batch(paths, batch_size=16) == ["a", "", "", "d"]


def test_batch_raises_when_every_file_fails(tmp_path):
    """A backend that fails on everything must stop the run, not score 100% WER."""
    backend, paths = _backend(tmp_path, ["a", "b"], fail_on=(0, 1))
    with pytest.raises(RuntimeError, match="failed for all 2 files"):
        backend.transcribe_batch(paths, batch_size=16)


def test_empty_batch(tmp_path):
    backend, _ = _backend(tmp_path, [])
    assert backend.transcribe_batch([], batch_size=16) == []


def test_sends_danish_language_hint(tmp_path):
    backend, paths = _backend(tmp_path, ["hej"])
    backend.transcribe_batch(paths, batch_size=16)
    model, language, _ = backend.model.audio.transcriptions.calls[0]
    assert (model, language) == ("ordbogen/whisper", "da")


def test_strips_whitespace_and_handles_missing_text(tmp_path):
    backend, paths = _backend(tmp_path, ["  spaced  "])
    assert backend.transcribe_batch(paths, batch_size=16) == ["spaced"]


# --------------------------------------------------------------------------
# Loader
# --------------------------------------------------------------------------

def test_load_requires_api_key(monkeypatch):
    monkeypatch.delenv("ORDBOGEN_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ordbogen-api-key"):
        load("ordbogen/whisper", LoadOptions())


def test_default_rpm_stays_under_tier_0_limit():
    """Tier 0 permits 120 requests/min; pacing must leave headroom for jitter."""
    assert DEFAULT_RPM < 120
    assert LoadOptions().ordbogen_rpm == DEFAULT_RPM
