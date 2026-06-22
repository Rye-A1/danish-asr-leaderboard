"""Transcription orchestration shared by every backend.

A backend is any object satisfying :class:`Backend` — it transcribes a list of
WAV paths to a list of hypothesis strings. This module times inference, measures
audio duration, and applies the shared normalisation to both sides before the
metrics are computed by the caller.
"""
from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

from danish_asr_leaderboard.audio import wav_duration
from danish_asr_leaderboard.normalizer import normalise


@runtime_checkable
class Backend(Protocol):
    """Structural interface every backend implements."""

    name: str

    def transcribe(self, audio_paths: list[str], *, batch_size: int) -> list[str]:
        """Return one hypothesis string per input path (same order)."""
        ...


def transcribe_dataset(
    backend: Backend,
    rows: list[dict],
    *,
    batch_size: int = 16,
    unicode_form: str = "NFC",
) -> tuple[list[str], list[str], float, float, list[dict]]:
    """Transcribe ``rows`` and return ``(refs, hyps, inference_secs, audio_secs, raw)``.

    ``refs`` and ``hyps`` are normalised (under ``unicode_form``) and ready for
    ``compute_wer``/``compute_cer``. ``raw`` is one record per row carrying the
    *un-normalised* reference and hypothesis plus a stable sample id, so the exact
    model output can be persisted and re-scored offline under any normaliser
    configuration without re-running inference.
    """
    audio_paths = [row["audio_path"] for row in rows]
    refs_raw = [row["reference_text"] for row in rows]
    refs = [normalise(r, unicode_form=unicode_form) for r in refs_raw]
    total_audio_secs = sum(wav_duration(p) for p in audio_paths)

    t0 = time.perf_counter()
    hyps_raw = backend.transcribe(audio_paths, batch_size=batch_size)
    total_infer_secs = time.perf_counter() - t0

    rtf = total_infer_secs / total_audio_secs if total_audio_secs > 0 else float("nan")
    print(f"  {len(rows)} files in {total_infer_secs:.1f}s  (RTF {rtf:.3f})")

    hyps = [normalise(h, unicode_form=unicode_form) for h in hyps_raw]
    raw = [
        {"id": path, "reference": ref, "hypothesis": hyp}
        for path, ref, hyp in zip(audio_paths, refs_raw, hyps_raw)
    ]
    return refs, hyps, total_infer_secs, total_audio_secs, raw
