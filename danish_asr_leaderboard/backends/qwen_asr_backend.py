"""Qwen3-ASR backend (also used for Qwen3-ASR fine-tunes, e.g. Saga)."""
from __future__ import annotations

import importlib

from danish_asr_leaderboard.backends._torch_util import device_map
from danish_asr_leaderboard.backends.base import Backend, LoadOptions, register

_LANGUAGE = "Danish"


class QwenAsrBackend(Backend):
    name = "qwen-asr"

    def transcribe_one(self, audio_path: str) -> str:
        results = self.model.transcribe(audio=audio_path, language=_LANGUAGE)
        if not results:
            return ""
        return (results[0].text or "").strip()


@register("qwen-asr")
def load(model_ref: str, options: LoadOptions) -> Backend:
    import torch

    qwen_asr = importlib.import_module("qwen_asr")
    model = qwen_asr.Qwen3ASRModel.from_pretrained(
        model_ref,
        dtype=torch.bfloat16,
        device_map=device_map(options.device),
        max_inference_batch_size=1,
        max_new_tokens=256,
    )
    return QwenAsrBackend(model, options=options)
