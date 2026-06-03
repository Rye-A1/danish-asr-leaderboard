"""Backend interface, load options, and the backend registry.

A backend wraps a loaded ASR model and turns WAV paths into hypothesis strings.
Subclasses implement :meth:`transcribe_one` (per-file) and may override
:meth:`transcribe_batch` when the underlying model has a native batch API. The
public :meth:`transcribe` adds the shared batch->sequential error fallback.

Backend modules register a loader via :func:`register`. Heavy imports (torch,
transformers, nemo, vendor SDKs) must stay *inside* the loader / transcribe
methods so that importing the registry is cheap and dependency-free.
"""
from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable


@dataclass
class LoadOptions:
    """Everything a backend loader might need, populated from the CLI."""

    device: str = "cuda"
    compute_type: str = "float16"
    # NeMo
    nemo_model_type: str = "canary"   # "canary" | "parakeet"
    nemo_beam_size: int = 1
    kenlm_model: str | None = None
    kenlm_alpha: float = 0.075
    kenlm_beam_size: int = 5
    # ElevenLabs
    elevenlabs_api_key: str | None = None
    elevenlabs_model_id: str = "scribe_v2"
    # Azure OpenAI
    azure_openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_api_version: str = "2025-01-01-preview"
    # Google Chirp
    google_cloud_project: str | None = None
    google_credentials_file: str | None = None
    google_chirp_model_id: str = "chirp_3"
    # Soniox
    soniox_api_key: str | None = None
    soniox_model: str = "soniox-v1"


class Backend(ABC):
    """Base class for all ASR backends."""

    name: str = ""

    def __init__(self, model: object, *, options: LoadOptions | None = None) -> None:
        self.model = model
        self.options = options or LoadOptions()

    @abstractmethod
    def transcribe_one(self, audio_path: str) -> str:
        """Transcribe a single WAV file."""

    def transcribe_batch(self, audio_paths: list[str], *, batch_size: int) -> list[str]:
        """Transcribe a list of files. Default: robust per-file loop.

        Backends with a native multi-file API override this.
        """
        return self._sequential(audio_paths)

    def transcribe(self, audio_paths: list[str], *, batch_size: int) -> list[str]:
        """Public entry point with a batch->sequential error fallback."""
        try:
            return self.transcribe_batch(audio_paths, batch_size=batch_size)
        except Exception as exc:  # noqa: BLE001 - intentional broad fallback
            print(
                f"  WARNING: batch transcription failed ({exc}), falling back to sequential...",
                file=sys.stderr,
            )
            return self._sequential(audio_paths)

    def _sequential(self, audio_paths: list[str]) -> list[str]:
        hyps: list[str] = []
        for i, audio_path in enumerate(audio_paths):
            try:
                hyps.append(self.transcribe_one(audio_path))
            except Exception as exc:  # noqa: BLE001
                print(f"  WARNING: transcription failed for {audio_path}: {exc}", file=sys.stderr)
                hyps.append("")
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(audio_paths)} done...")
        return hyps

    def release(self) -> None:
        """Free model memory (GPU cache)."""
        import gc

        self.model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BackendLoader = Callable[[str, LoadOptions], Backend]
_REGISTRY: dict[str, BackendLoader] = {}


def register(*names: str) -> Callable[[BackendLoader], BackendLoader]:
    """Register a loader under one or more backend names."""

    def deco(fn: BackendLoader) -> BackendLoader:
        for n in names:
            _REGISTRY[n] = fn
        return fn

    return deco


def load_backend(name: str, model_ref: str, options: LoadOptions) -> Backend:
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown backend: {name!r}. Available: {', '.join(sorted(_REGISTRY))}"
        )
    print(f"\n=== Loading model: {model_ref} (backend={name}) ===")
    return _REGISTRY[name](model_ref, options)


def available_backends() -> list[str]:
    return sorted(_REGISTRY)
