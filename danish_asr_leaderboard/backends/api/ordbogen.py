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

from danish_asr_leaderboard.backends.base import LoadOptions, register
from danish_asr_leaderboard.backends.api._base import ApiBackend

BASE_URL = "https://api.ordbogen.ai/v1"
# Tier 0 is 120 RPM. Sit just under it so normal jitter does not trip a 429;
# ApiBackend's retry absorbs whatever still gets through.
DEFAULT_RPM = 110
DEFAULT_CONCURRENCY = 8



class OrdbogenBackend(ApiBackend):
    name = "ordbogen"

    def __init__(self, client, model_ref, *, rpm, concurrency, options=None):
        super().__init__(client, options=options)
        self.model_ref = model_ref
        # Pooling, pacing and retry all live in ApiBackend; this backend only
        # supplies the vendor call and its tier-specific limits.
        self.concurrency = max(1, concurrency)
        self.rpm = rpm

    def _call(self, audio_path: str) -> str:
        with open(audio_path, "rb") as f:
            resp = self.model.audio.transcriptions.create(
                model=self.model_ref, file=f, language="da"
            )
        return (getattr(resp, "text", "") or "").strip()



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
