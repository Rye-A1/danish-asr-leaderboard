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

Requests are also overlapped across a small thread pool, so a ~27k-request
sweep is limited by pacing rather than by round-trip latency. This lives here
rather than per-backend because it was previously implemented only in the
Ordbogen backend, which made the published ``speed_x`` incomparable: that model
appeared ~2.5x faster than the sequential gpt-4o rows purely because of client
code. Concurrency and optional client-side pacing are per-backend knobs, since
vendor limits differ.

Subclasses implement :meth:`_call`; :meth:`transcribe_one` is the retry wrapper
around it and should not be overridden.
"""
from __future__ import annotations

import random
import sys
import threading
import time
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

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


class _Pacer:
    """Evenly space request starts so they never exceed ``rpm`` per minute.

    Vendor limits bind on requests, not audio duration: the benchmark's mean
    utterance is ~5.6 s, so a request-per-minute cap is reached long before any
    duration allowance. rpm<=0 disables pacing.
    """

    def __init__(self, rpm: int) -> None:
        self.min_gap = 60.0 / rpm if rpm > 0 else 0.0
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if self.min_gap <= 0:
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.min_gap
        delay = start - now
        if delay > 0:
            time.sleep(delay)


class ApiBackend(Backend):
    """Base for hosted-API backends. Adds retry and concurrency around calls."""

    max_attempts = 6
    base_delay = 1.0
    max_delay = 60.0
    # Overlap requests across a small pool. 8 is well inside every vendor tier
    # we use, and 429s are absorbed by the retry above rather than lost.
    concurrency = 8
    # Client-side requests/minute cap; 0 disables. Set per backend where the
    # vendor's limit is low enough that retry-on-429 would dominate the run.
    rpm = 0

    @abstractmethod
    def _call(self, audio_path: str) -> str:
        """Perform one vendor transcription request."""

    def _pace(self) -> None:
        """Block until this worker's turn, if the backend sets an rpm cap."""
        pacer = getattr(self, "_pacer_obj", None)
        if pacer is None:
            pacer = _Pacer(self.rpm)
            self._pacer_obj = pacer
        pacer.wait()

    def transcribe_batch(self, audio_paths: list[str], *, batch_size: int) -> list[str]:
        """Overlap paced requests across a thread pool, preserving input order.

        Errors are tolerated per file, matching the sequential path, so one bad
        file does not abort a multi-hour sweep. If *every* file fails the run is
        stopped rather than scored as a plausible-looking 100% WER — a total
        failure here is almost always an immediate auth or configuration error.
        """
        if not audio_paths:
            return []
        if self.concurrency <= 1:
            return self._sequential(audio_paths)

        results = [""] * len(audio_paths)
        failures = 0
        first_error: Exception | None = None
        workers = min(self.concurrency, len(audio_paths))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self.transcribe_one, path): i
                for i, path in enumerate(audio_paths)
            }
            for done, future in enumerate(as_completed(futures), 1):
                i = futures[future]
                try:
                    results[i] = future.result()
                except Exception as exc:  # noqa: BLE001 - per-file tolerance
                    failures += 1
                    if first_error is None:
                        first_error = exc
                    print(
                        f"  WARNING: transcription failed for {audio_paths[i]}: {exc}",
                        file=sys.stderr,
                    )
                if done % 500 == 0:
                    print(f"  {done}/{len(audio_paths)} done...")

        if failures == len(audio_paths):
            raise RuntimeError(
                f"transcription failed for all {failures} files; "
                f"first error: {first_error}"
            ) from first_error
        return results

    def transcribe_one(self, audio_path: str) -> str:
        delay = self.base_delay
        for attempt in range(1, self.max_attempts + 1):
            try:
                self._pace()
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
