"""Shared behaviour for hosted-API backends: rate-limit and transient retry.

Every vendor here throttles, and a full sweep is ~27k requests, so a 429 part
way through is expected rather than exceptional. Without this a throttled
request surfaces as a per-file failure and is scored as an empty hypothesis —
a silent WER penalty that looks like a bad model rather than a bad run.

Each SDK reports throttling differently (an ``APIStatusError`` with
``status_code``, a ``response.status_code``, or just a message), so the
classifier looks for a status code in the usual places and falls back to
matching the message. Vendor ``Retry-After`` is honoured when present;
otherwise the wait is exponential with jitter, so a pool of workers that all
get throttled at once does not retry in lockstep.

Subclasses implement :meth:`_call`; :meth:`transcribe_one` is the retry wrapper
around it and should not be overridden.
"""
from __future__ import annotations

import random
import sys
import time
from abc import abstractmethod

from danish_asr_leaderboard.backends.base import Backend

# 429 is the throttle; the 5xx family and 408 are transient and worth another
# attempt over a multi-hour run. Anything else (401, 400, 404) is a
# configuration error that will not fix itself, so it is raised immediately.
RETRY_STATUS = {408, 429, 500, 502, 503, 504}

_RETRY_HINTS = (
    "rate limit", "too many requests", "429", "timed out", "timeout",
    "connection reset", "connection aborted", "temporarily unavailable",
    "service unavailable", "bad gateway",
)


def status_of(exc: Exception) -> int | None:
    """HTTP status carried by an SDK exception, if it exposes one."""
    for attr in ("status_code", "status", "http_status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def retry_after(exc: Exception) -> float | None:
    """Vendor-requested wait in seconds, from a ``Retry-After`` header."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None or not hasattr(headers, "get"):
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None  # HTTP-date form; fall back to exponential backoff


def is_retryable(exc: Exception) -> bool:
    status = status_of(exc)
    if status is not None:
        return status in RETRY_STATUS
    message = str(exc).lower()
    return any(hint in message for hint in _RETRY_HINTS)


class ApiBackend(Backend):
    """Base for hosted-API backends. Adds retry around each vendor call."""

    max_attempts = 6
    base_delay = 1.0
    max_delay = 60.0

    @abstractmethod
    def _call(self, audio_path: str) -> str:
        """Perform one vendor transcription request."""

    def transcribe_one(self, audio_path: str) -> str:
        delay = self.base_delay
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self._call(audio_path)
            except Exception as exc:  # noqa: BLE001 - classified below
                if attempt == self.max_attempts or not is_retryable(exc):
                    raise
                wait = retry_after(exc)
                if wait is None:
                    # Jitter so concurrent workers do not retry in lockstep.
                    wait = min(delay, self.max_delay) * (0.5 + random.random())
                    delay = min(delay * 2, self.max_delay)
                print(
                    f"  rate-limited/transient ({status_of(exc) or type(exc).__name__}), "
                    f"retry {attempt}/{self.max_attempts - 1} in {wait:.1f}s",
                    file=sys.stderr,
                )
                time.sleep(wait)
        raise AssertionError("unreachable")  # pragma: no cover

    def release(self) -> None:
        """No GPU memory to free for a network client."""
        self.model = None
