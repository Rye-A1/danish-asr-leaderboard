"""Unit tests for the shared hosted-API retry (no SDKs, no network, no sleeping)."""
import pytest

from danish_asr_leaderboard.backends.api._base import (
    ApiBackend,
    is_retryable,
    retry_after,
    status_of,
)


class _Resp:
    def __init__(self, status=None, headers=None):
        if status is not None:
            self.status_code = status
        self.headers = headers or {}


def _exc(message="boom", *, status=None, headers=None, attr=None):
    e = RuntimeError(message)
    if attr and status is not None:
        setattr(e, attr, status)
    elif status is not None or headers is not None:
        e.response = _Resp(status, headers)
    return e


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

@pytest.mark.parametrize("attr", ["status_code", "status", "http_status"])
def test_status_read_from_each_sdk_convention(attr):
    assert status_of(_exc(status=429, attr=attr)) == 429


def test_status_read_from_nested_response():
    assert status_of(_exc(status=503)) == 503


def test_status_none_when_absent():
    assert status_of(RuntimeError("no status here")) is None


@pytest.mark.parametrize("status, expected", [
    (429, True),    # throttled
    (408, True), (500, True), (502, True), (503, True), (504, True),  # transient
    (401, False),   # bad key — will never fix itself
    (400, False), (404, False), (413, False),
])
def test_retryable_by_status(status, expected):
    assert is_retryable(_exc(status=status)) is expected


@pytest.mark.parametrize("message, expected", [
    ("Rate limit exceeded", True),
    ("429 Too Many Requests", True),
    ("Read timed out", True),
    ("Service Unavailable", True),
    ("Invalid API key provided", False),
    ("unsupported audio format", False),
])
def test_retryable_by_message_when_no_status(message, expected):
    assert is_retryable(RuntimeError(message)) is expected


def test_status_wins_over_message():
    """A 401 whose text happens to mention rate limits must not be retried."""
    assert is_retryable(_exc("rate limit note", status=401)) is False


# --------------------------------------------------------------------------
# Retry-After
# --------------------------------------------------------------------------

def test_retry_after_seconds():
    assert retry_after(_exc(status=429, headers={"retry-after": "2.5"})) == 2.5


def test_retry_after_capitalised():
    assert retry_after(_exc(status=429, headers={"Retry-After": "7"})) == 7.0


def test_retry_after_http_date_falls_back_to_backoff():
    """The HTTP-date form is not parsed; None means 'use exponential backoff'."""
    assert retry_after(_exc(status=429, headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"})) is None


def test_retry_after_absent():
    assert retry_after(_exc(status=429)) is None


# --------------------------------------------------------------------------
# Retry loop
# --------------------------------------------------------------------------

class _Flaky(ApiBackend):
    """Fails with `errors` in order, then returns 'ok'."""

    name = "flaky"

    def __init__(self, errors):
        super().__init__(object())
        self.errors = list(errors)
        self.attempts = 0

    def _call(self, audio_path):
        self.attempts += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


@pytest.fixture(autouse=True)
def _no_sleeping(monkeypatch):
    slept = []
    monkeypatch.setattr("danish_asr_leaderboard.backends.api._base.time.sleep", slept.append)
    return slept


def test_succeeds_after_throttling(_no_sleeping):
    backend = _Flaky([_exc(status=429), _exc(status=429)])
    assert backend.transcribe_one("a.wav") == "ok"
    assert backend.attempts == 3


def test_does_not_retry_auth_error(_no_sleeping):
    backend = _Flaky([_exc("bad key", status=401)])
    with pytest.raises(RuntimeError, match="bad key"):
        backend.transcribe_one("a.wav")
    assert backend.attempts == 1
    assert _no_sleeping == []


def test_gives_up_after_max_attempts(_no_sleeping):
    backend = _Flaky([_exc(status=429) for _ in range(20)])
    with pytest.raises(RuntimeError):
        backend.transcribe_one("a.wav")
    assert backend.attempts == ApiBackend.max_attempts


def test_honours_retry_after_header(_no_sleeping):
    backend = _Flaky([_exc(status=429, headers={"retry-after": "3"})])
    backend.transcribe_one("a.wav")
    assert _no_sleeping == [3.0]


def test_backoff_grows_and_is_jittered(_no_sleeping):
    backend = _Flaky([_exc(status=503) for _ in range(4)])
    backend.transcribe_one("a.wav")
    waits = _no_sleeping
    assert len(waits) == 4
    # Jitter is 0.5x-1.5x of a doubling base, so waits are bounded but not equal.
    for i, w in enumerate(waits):
        assert 0.5 * 2**i <= w <= 1.5 * 2**i
    assert len(set(waits)) > 1


def test_wait_is_capped(_no_sleeping):
    class _Slow(_Flaky):
        max_attempts = 12
        base_delay = 100.0
        max_delay = 5.0

    backend = _Slow([_exc(status=429) for _ in range(3)])
    backend.transcribe_one("a.wav")
    assert all(w <= 5.0 * 1.5 for w in _no_sleeping)
