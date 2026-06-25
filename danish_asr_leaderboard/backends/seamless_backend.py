"""SeamlessM4Tv2 speech-to-text backend (101 input languages incl. Danish)."""
from __future__ import annotations

import importlib

from danish_asr_leaderboard.audio import load_audio_array
from danish_asr_leaderboard.backends._torch_util import bf16_dtype, device_map
from danish_asr_leaderboard.backends.base import Backend, LoadOptions, register


class SeamlessBackend(Backend):
    name = "seamless"
    tgt_lang = "dan"

    def __init__(self, model, processor, *, options=None):
        super().__init__(model, options=options)
        self.processor = processor

    def transcribe_batch(self, audio_paths: list[str], *, batch_size: int) -> list[str]:
        import torch

        results: list[str] = []
        device = next(self.model.parameters()).device
        for i in range(0, len(audio_paths), batch_size):
            batch = audio_paths[i : i + batch_size]
            audios = [load_audio_array(p) for p in batch]
            # transformers >= 5.0 renamed the processor's audio kwarg `audios` -> `audio`
            # and made the old name a hard error; fall back to `audios` on < 5.0.
            try:
                inputs = self.processor(
                    audio=audios, sampling_rate=16000, return_tensors="pt", padding=True
                )
            except (TypeError, ValueError):
                inputs = self.processor(
                    audios=audios, sampling_rate=16000, return_tensors="pt", padding=True
                )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.model.generate(**inputs, tgt_lang=self.tgt_lang, generate_speech=False)
            texts = self.processor.batch_decode(out[0], skip_special_tokens=True)
            results.extend(t.strip() for t in texts)
        return results

    def transcribe_one(self, audio_path: str) -> str:
        return self.transcribe_batch([audio_path], batch_size=1)[0]


@register("seamless")
def load(model_ref: str, options: LoadOptions) -> Backend:
    transformers = importlib.import_module("transformers")
    proc = transformers.AutoProcessor.from_pretrained(model_ref)
    model_obj = transformers.SeamlessM4Tv2Model.from_pretrained(
        model_ref, torch_dtype=bf16_dtype(options.device), device_map=device_map(options.device)
    )
    return SeamlessBackend(model_obj, proc, options=options)
