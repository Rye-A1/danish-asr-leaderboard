"""Ordbogen.ai / OdinCore transcription backend (ordbogen/whisper).

OpenAI-compatible: the vendor documents ``https://api.ordbogen.ai/v1`` as a
drop-in base URL for the OpenAI SDK, so this is the standard client pointed
somewhere else.

Two things about this API shape the backend:

* **Rate limits bind on requests, not audio.** Tier 0 allows 120 requests/min
  and 5,000,000 ms/min of audio; the benchmark's mean utterance is ~5.6 s, so
  the request limit is reached while barely a seventh of the duration
  allowance is used. Requests are therefore paced client-side rather than
  fired as fast as the network allows, and overlapped across a small thread
  pool so the sweep is limited by the pacing rather than by round-trip
  latency. (Tier 1 doubles both limits on an organisation's first payment;
  raise ``--ordbogen-rpm`` to match.)
* **Metering is per-millisecond of audio**, with no per-request minimum, so
  splitting the corpus into ~27k short requests costs the same as sending it
  in bulk. Verified against the live meter: two 7.02 s clips consumed exactly
  14,040 ms of the duration allowance. Rounding to 15 s would have charged
  30,000 ms, and a one-minute minimum 120,000 ms — a 10x difference at this
  corpus's 5.6 s mean utterance.
"""
from __future__ import annotations

import importlib
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from danish_asr_leaderboard.backends.base import LoadOptions, register
from danish_asr_leaderboard.backends.api._base import ApiBackend

BASE_URL = "https://api.ordbogen.ai/v1"
# Tier 0 is 120 RPM. Sit just under it so normal jitter does not trip a 429;
# ApiBackend's retry absorbs whatever still gets through.
DEFAULT_RPM = 110
DEFAULT_CONCURRENCY = 8


class _Pacer:
    """Evenly space request starts so they never exceed ``rpm`` per minute."""

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


class OrdbogenBackend(ApiBackend):
    name = "ordbogen"

    def __init__(self, client, model_ref, *, rpm, concurrency, options=None):
        super().__init__(client, options=options)
        self.model_ref = model_ref
        self.concurrency = max(1, concurrency)
        self._pacer = _Pacer(rpm)

    def _call(self, audio_path: str) -> str:
        # Pace inside the retried call so a retry is paced too.
        self._pacer.wait()
        with open(audio_path, "rb") as f:
            resp = self.model.audio.transcriptions.create(
                model=self.model_ref, file=f, language="da"
            )
        return (getattr(resp, "text", "") or "").strip()

    def transcribe_batch(self, audio_paths: list[str], *, batch_size: int) -> list[str]:
        """Overlap paced requests across a thread pool, preserving input order.

        Errors are tolerated per file, matching the sequential path. If *every*
        file fails the run is stopped rather than scored as a plausible-looking
        100% WER — note that the base class will retry sequentially once before
        the error surfaces, which is cheap because a total failure here is
        almost always an immediate auth or configuration error.
        """
        if not audio_paths:
            return []
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


@register("ordbogen")
def load(model_ref: str, options: LoadOptions) -> ApiBackend:
    # Validate configuration before importing the SDK, so a missing key reports
    # itself as a missing key rather than as an ImportError.
    api_key = options.ordbogen_api_key or os.environ.get("ORDBOGEN_API_KEY")
    if not api_key:
        raise ValueError(
            "Ordbogen requires --ordbogen-api-key (or ORDBOGEN_API_KEY)"
        )
    base_url = options.ordbogen_base_url or BASE_URL
    openai_mod = importlib.import_module("openai")
    client = openai_mod.OpenAI(
        api_key=api_key,
        base_url=base_url,
        # Retrying lives in ApiBackend so every hosted backend behaves the
        # same way; disable the SDK's own so the two do not compound.
        max_retries=0,
        timeout=120.0,
    )
    print(
        f"  Ordbogen client ready (model={model_ref}, {options.ordbogen_rpm} rpm, "
        f"{options.ordbogen_concurrency} workers) [API — speed is network-bound]"
    )
    return OrdbogenBackend(
        client,
        model_ref,
        rpm=options.ordbogen_rpm,
        concurrency=options.ordbogen_concurrency,
        options=options,
    )
