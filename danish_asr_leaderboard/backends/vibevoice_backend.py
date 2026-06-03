"""VibeVoice ASR backend (transformers >= 5.3.0).

The model emits a JSON-like string with speaker/timestamp metadata;
``return_format="transcription_only"`` extracts plain text.
"""
from __future__ import annotations

import importlib

from danish_asr_leaderboard.backends._torch_util import bf16_dtype, device_map
from danish_asr_leaderboard.backends.base import Backend, LoadOptions, register


class VibeVoiceBackend(Backend):
    name = "vibevoice"

    def __init__(self, model, processor, *, options=None):
        super().__init__(model, options=options)
        self.processor = processor

    def transcribe_batch(self, audio_paths: list[str], *, batch_size: int) -> list[str]:
        import torch

        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        results: list[str] = []
        for i in range(0, len(audio_paths), batch_size):
            batch = audio_paths[i : i + batch_size]
            inputs = self.processor.apply_transcription_request(
                batch if len(batch) > 1 else batch[0]
            )
            prompt_len = inputs["input_ids"].shape[1]
            inputs = {
                k: (v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device=device))
                for k, v in inputs.items()
            }
            with torch.no_grad():
                out_ids = self.model.generate(**inputs)
            texts = self.processor.decode(
                out_ids[:, prompt_len:], return_format="transcription_only"
            )
            if isinstance(texts, list):
                results.extend((t or "").strip() for t in texts)
            else:
                results.append((texts or "").strip())
        return results

    def transcribe_one(self, audio_path: str) -> str:
        return self.transcribe_batch([audio_path], batch_size=1)[0]


@register("vibevoice")
def load(model_ref: str, options: LoadOptions) -> Backend:
    transformers = importlib.import_module("transformers")
    proc = transformers.AutoProcessor.from_pretrained(model_ref)
    model_obj = transformers.VibeVoiceAsrForConditionalGeneration.from_pretrained(
        model_ref, torch_dtype=bf16_dtype(options.device), device_map=device_map(options.device)
    )
    return VibeVoiceBackend(model_obj, proc, options=options)
