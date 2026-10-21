"""Mistral's hosted Voxtral 3 file-transcription API.

The local voxtral backend handles the 2025 open-weight models. This backend
uses the paid file endpoint for the two 2026 Transcribe 3 models. A durable
per-clip cache is mandatory so interrupted benchmark runs do not buy the same
transcripts twice. Unlike the generic API fallback, any failed clip aborts the
run rather than silently scoring an empty transcript.
"""
from __future__ import annotations

import importlib
import json
import os
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from danish_asr_leaderboard.backends.base import Backend, LoadOptions, register
from danish_asr_leaderboard.backends.api._base import ApiBackend

MODEL_ALIASES = {
    "voxtral-mini-transcribe-3-0": "voxtral-mini-transcribe-3",
    "voxtral-small-transcribe-3-0": "voxtral-small-transcribe-3",
}
LANGUAGE = "da"
CACHE_VERSION = 1


class MistralBackend(ApiBackend):
    name = "mistral"

    def __init__(self, client, model_ref: str, *, options: LoadOptions):
        super().__init__(client, options=options)
        if model_ref not in MODEL_ALIASES:
            raise ValueError(f"Unsupported Mistral model: {model_ref}")
        if not options.mistral_cache:
            raise ValueError("Mistral requires --mistral-cache to avoid repeat charges")
        self.model_ref = model_ref
        self.cache_path = Path(options.mistral_cache)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.concurrency = max(1, options.mistral_concurrency)
        self.rpm = max(0, options.mistral_rpm)
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, int, int], str] = {}
        self.new_requests = 0
        self.reported_audio_seconds = 0.0
        if self.cache_path.exists():
            with self.cache_path.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        row = json.loads(line)
                        if (row.get("version") == CACHE_VERSION
                                and row.get("model") == model_ref
                                and row.get("language") == LANGUAGE):
                            key = (row["audio"], row["size"], row["mtime_ns"])
                            self._cache[key] = row["text"]
                    except (ValueError, KeyError, TypeError):
                        continue  # an interrupted final write is safe to re-request
            print(f"  Mistral: {len(self._cache)} matching cached transcripts")

    @staticmethod
    def _identity(audio_path: str) -> tuple[str, int, int]:
        path = Path(audio_path).resolve()
        stat = path.stat()
        return str(path), stat.st_size, stat.st_mtime_ns

    def transcribe_one(self, audio_path: str) -> str:
        key = self._identity(audio_path)
        if key in self._cache:
            return self._cache[key]
        return super().transcribe_one(audio_path)

    def _call(self, audio_path: str) -> str:
        key = self._identity(audio_path)
        if key in self._cache:
            return self._cache[key]
        with open(audio_path, "rb") as audio:
            response = self.model.audio.transcriptions.complete(
                model=self.model_ref,
                file={"file_name": Path(audio_path).name, "content": audio},
                language=LANGUAGE,
            )
        served = getattr(response, "model", None)
        if served and served not in {self.model_ref, MODEL_ALIASES[self.model_ref]}:
            raise RuntimeError(f"Mistral served {served!r}, expected {self.model_ref!r}")
        transcript = (getattr(response, "text", None) or "").strip()
        usage = getattr(response, "usage", None)
        reported_seconds = getattr(usage, "prompt_audio_seconds", None)
        row = {
            "version": CACHE_VERSION,
            "model": self.model_ref,
            "served_model": served,
            "language": LANGUAGE,
            "audio": key[0],
            "size": key[1],
            "mtime_ns": key[2],
            "text": transcript,
            "reported_audio_seconds": reported_seconds,
        }
        with self._lock:
            self._cache[key] = transcript
            self.new_requests += 1
            if isinstance(reported_seconds, (int, float)):
                self.reported_audio_seconds += reported_seconds
            with self.cache_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        return transcript

    def transcribe(self, audio_paths: list[str], *, batch_size: int) -> list[str]:
        """Keep a small request window and stop at the first failed clip."""
        if not audio_paths:
            return []
        results = [""] * len(audio_paths)
        # Check credentials and model access before opening a pool of paid calls.
        results[0] = self.transcribe_one(audio_paths[0])
        next_index = 1
        completed = 1
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            pending = {}

            def fill() -> None:
                nonlocal next_index
                while next_index < len(audio_paths) and len(pending) < self.concurrency * 2:
                    future = pool.submit(self.transcribe_one, audio_paths[next_index])
                    pending[future] = next_index
                    next_index += 1

            fill()
            try:
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        index = pending.pop(future)
                        results[index] = future.result()
                        completed += 1
                        if completed % 500 == 0:
                            print(f"  {completed}/{len(audio_paths)} done...")
                    fill()
            except Exception:
                for future in pending:
                    future.cancel()
                raise
        print(
            f"  Mistral new calls: {self.new_requests}; "
            f"API-reported audio: {self.reported_audio_seconds:.0f}s"
        )
        return results


@register("mistral")
def load(model_ref: str, options: LoadOptions) -> Backend:
    if model_ref not in MODEL_ALIASES:
        raise ValueError(f"Use a versioned Voxtral 3 transcription ID: {', '.join(MODEL_ALIASES)}")
    if not options.mistral_cache:
        raise ValueError("Mistral requires --mistral-cache to avoid repeat charges")
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("Set MISTRAL_API_KEY in the environment; do not pass it on the command line")
    try:
        Mistral = importlib.import_module("mistralai.client").Mistral
    except ImportError as exc:
        raise RuntimeError("Install the Mistral SDK: pip install '.[mistral]'") from exc
    client = Mistral(api_key=api_key, timeout_ms=120_000)
    print(
        f"  Mistral client ready ({model_ref}, language={LANGUAGE}, "
        f"workers={options.mistral_concurrency}, rpm={options.mistral_rpm})"
    )
    return MistralBackend(client, model_ref, options=options)
