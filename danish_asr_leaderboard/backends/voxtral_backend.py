"""Voxtral backend (transformers >= 4.54.0 + mistral-common[audio]).

Voxtral is decoder-only: generate() returns the full sequence including the
prompt, so prompt tokens are sliced off before decoding. Audio is prepared via
``processor.apply_transcription_request`` rather than the raw pipeline API.
"""
from __future__ import annotations

import importlib

from danish_asr_leaderboard.backends._torch_util import bf16_dtype, device_map
from danish_asr_leaderboard.backends.base import Backend, LoadOptions, register


class VoxtralBackend(Backend):
    name = "voxtral"

    def __init__(self, model, processor, model_id, *, options=None):
        super().__init__(model, options=options)
        self.processor = processor
        self.model_id = model_id

    def transcribe_one(self, audio_path: str) -> str:
        import torch

        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        inputs = self.processor.apply_transcription_request(
            language="da", audio=audio_path, model_id=self.model_id
        )
        prompt_len = inputs.input_ids.shape[1]
        inputs = {
            k: (v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device=device))
            for k, v in inputs.items()
        }
        with torch.no_grad():
            out_ids = self.model.generate(**inputs, max_new_tokens=440)
        return self.processor.batch_decode(
            out_ids[:, prompt_len:], skip_special_tokens=True
        )[0].strip()


@register("voxtral")
def load(model_ref: str, options: LoadOptions) -> Backend:
    transformers = importlib.import_module("transformers")
    proc = transformers.AutoProcessor.from_pretrained(model_ref)
    model_obj = transformers.VoxtralForConditionalGeneration.from_pretrained(
        model_ref, dtype=bf16_dtype(options.device), device_map=device_map(options.device)
    )
    return VoxtralBackend(model_obj, proc, model_ref, options=options)
