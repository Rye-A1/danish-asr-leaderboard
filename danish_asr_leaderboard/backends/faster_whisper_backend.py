"""faster-whisper (CTranslate2) backend. No multi-file batch API -> per-file."""
from __future__ import annotations

import importlib

from danish_asr_leaderboard.backends.base import Backend, LoadOptions, register


class FasterWhisperBackend(Backend):
    name = "faster-whisper"
    beam_size = 5

    def transcribe_one(self, audio_path: str) -> str:
        segments, _ = self.model.transcribe(
            audio_path, language="da", beam_size=self.beam_size, vad_filter=True
        )
        return " ".join(seg.text.strip() for seg in segments).strip()


@register("faster-whisper")
def load(model_ref: str, options: LoadOptions) -> Backend:
    fw = importlib.import_module("faster_whisper")
    model = fw.WhisperModel(model_ref, device=options.device, compute_type=options.compute_type)
    return FasterWhisperBackend(model, options=options)
