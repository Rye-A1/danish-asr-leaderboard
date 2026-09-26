"""syv.ai transcription backend (syv-transcribe).

OpenAI-compatible: ``POST https://platform.syv.ai/v1/audio/transcriptions``, so
this is the standard client pointed somewhere else.

Three things about this API shape the backend:

* **Metering is per-millisecond of audio at 3 credits/second**, with no
  per-request minimum: an 8.208 s clip was charged 24.62 credits. A full sweep
  (~41.9 h) costs ~452k credits, so paying twice for any part of it matters.
* **Every transcript is cached as it arrives** (``--syv-cache``, one JSON line
  per clip). An interrupted or crashed sweep resumes by re-requesting only the
  clips that are missing, instead of paying for the whole set again.
* **The served model is checked.** Each response names its model in
  ``X-Model``; a response from anything other than the requested model fails
  the clip rather than being scored under this entry's name. Charged credits
  (``X-Credits-Charged``) are recorded in the cache next to each transcript.
* **The request is the vendor's default call**: only ``model`` and ``file`` are
  sent. Cache rows are marked with that request mode, so transcripts from older
  runs with explicit VAD or language settings cannot be reused accidentally.
"""
from __future__ import annotations

import importlib
import json
import os
import threading

from danish_asr_leaderboard.backends.base import LoadOptions, register
from danish_asr_leaderboard.backends.api._base import ApiBackend

BASE_URL = "https://platform.syv.ai/v1"
REQUEST_MODE = "default"


class SyvBackend(ApiBackend):
    name = "syv"

    def __init__(self, client, model_ref, *, concurrency, cache_path=None, options=None):
        super().__init__(client, options=options)
        self.model_ref = model_ref
        self.concurrency = max(1, concurrency)
        self.cache_path = cache_path
        self._lock = threading.Lock()
        self._cache: dict[str, str] = {}
        self.credits_charged = 0.0
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        if (row.get("request_mode") == REQUEST_MODE
                                and row.get("requested_model") == model_ref):
                            self._cache[row["audio"]] = row["text"]
            print(f"  syv: {len(self._cache)} default-call transcripts will not be re-requested")

    def _call(self, audio_path: str) -> str:
        key = os.path.realpath(audio_path)
        if key in self._cache:
            return self._cache[key]
        with open(audio_path, "rb") as f:
            raw = self.model.audio.transcriptions.with_raw_response.create(
                model=self.model_ref, file=f,
            )
        served = raw.headers.get("x-model")
        if served and served != self.model_ref:
            raise RuntimeError(f"syv served model {served!r}, not {self.model_ref!r}")
        text = (getattr(raw.parse(), "text", "") or "").strip()
        charged = float(raw.headers.get("x-credits-charged") or 0.0)
        with self._lock:
            self._cache[key] = text
            self.credits_charged += charged
            if self.cache_path:
                with open(self.cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"audio": key, "text": text, "model": served,
                                        "requested_model": self.model_ref,
                                        "request_mode": REQUEST_MODE,
                                        "credits": charged},
                                       ensure_ascii=False) + "\n")
        return text


@register("syv")
def load(model_ref: str, options: LoadOptions) -> ApiBackend:
    # Validate configuration before importing the SDK, so a missing key reports
    # itself as a missing key rather than as an ImportError.
    api_key = options.syv_api_key or os.environ.get("SYV_API_KEY")
    if not api_key:
        raise ValueError("syv requires --syv-api-key (or SYV_API_KEY)")
    openai_mod = importlib.import_module("openai")
    client = openai_mod.OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        # Retrying lives in ApiBackend so every hosted backend behaves the
        # same way; disable the SDK's own so the two do not compound.
        max_retries=0,
        timeout=300.0,
    )
    print(
        f"  syv client ready (model={model_ref}, {options.syv_concurrency} workers, "
        f"default call, cache={options.syv_cache or 'off'}) [API — speed is network-bound]"
    )
    return SyvBackend(
        client,
        model_ref,
        concurrency=options.syv_concurrency,
        cache_path=options.syv_cache,
        options=options,
    )
