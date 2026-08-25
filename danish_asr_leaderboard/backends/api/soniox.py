"""Soniox speech-to-text API backend."""
from __future__ import annotations

import importlib
import os

from danish_asr_leaderboard.backends.base import LoadOptions, register
from danish_asr_leaderboard.backends.api._base import ApiBackend


class SonioxBackend(ApiBackend):
    name = "soniox"

    def __init__(self, client, model, *, options=None):
        super().__init__(client, options=options)
        self.model_name = model

    def _call(self, audio_path: str) -> str:
        with open(audio_path, "rb") as f:
            response = self.model.transcribe(
                file=f, model=self.model_name, language_hints=["da"]
            )
        return (response.text or "").strip()


@register("soniox")
def load(model_ref: str, options: LoadOptions) -> Backend:
    try:
        soniox_mod = importlib.import_module("soniox.client")
        Client = soniox_mod.Client
    except Exception as exc:
        raise RuntimeError("soniox is not installed. Install: pip install soniox") from exc

    api_key = options.soniox_api_key or os.environ.get("SONIOX_API_KEY", "")
    if not api_key:
        raise ValueError("Soniox API key required: pass --soniox-api-key or set SONIOX_API_KEY")
    client = Client(api_key=api_key)
    print(f"  Soniox client ready (model={options.soniox_model}) [API — speed is network-bound]")
    return SonioxBackend(client, options.soniox_model, options=options)
