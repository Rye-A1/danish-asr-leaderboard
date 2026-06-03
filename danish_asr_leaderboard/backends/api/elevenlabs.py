"""ElevenLabs speech-to-text API backend (scribe_v2 etc.)."""
from __future__ import annotations

import importlib
import os

from danish_asr_leaderboard.backends.base import Backend, LoadOptions, register


class ElevenLabsBackend(Backend):
    name = "elevenlabs"

    def __init__(self, client, model_id, *, options=None):
        super().__init__(client, options=options)
        self.model_id = model_id

    def transcribe_one(self, audio_path: str) -> str:
        with open(audio_path, "rb") as f:
            resp = self.model.speech_to_text.convert(
                file=f, model_id=self.model_id, language_code="da"
            )
        return (resp.text or "").strip()


@register("elevenlabs")
def load(model_ref: str, options: LoadOptions) -> Backend:
    elevenlabs_mod = importlib.import_module("elevenlabs")
    api_key = options.elevenlabs_api_key or os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError(
            "ElevenLabs API key required: pass --elevenlabs-api-key or set ELEVENLABS_API_KEY"
        )
    client = elevenlabs_mod.ElevenLabs(api_key=api_key)
    print(f"  ElevenLabs client ready (model_id={options.elevenlabs_model_id}) [API — speed is network-bound]")
    return ElevenLabsBackend(client, options.elevenlabs_model_id, options=options)
