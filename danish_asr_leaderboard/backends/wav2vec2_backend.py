"""CTC ASR (wav2vec2 / MMS) via the HF ``transformers`` pipeline."""
from __future__ import annotations

import importlib

from danish_asr_leaderboard.backends._torch_util import cuda_ok, pipeline_device
from danish_asr_leaderboard.backends.base import Backend, LoadOptions, register


class Wav2Vec2Backend(Backend):
    name = "wav2vec2"

    def transcribe_batch(self, audio_paths: list[str], *, batch_size: int) -> list[str]:
        # CTC models have no autoregressive generation kwargs.
        raw = self.model(audio_paths, batch_size=batch_size)
        return [((r or {}).get("text", "").strip() if r else "") for r in raw]

    def transcribe_one(self, audio_path: str) -> str:
        result = self.model(audio_path)
        return (result or {}).get("text", "").strip() if result else ""


@register("wav2vec2")
def load(model_ref: str, options: LoadOptions) -> Backend:
    transformers = importlib.import_module("transformers")
    pipeline_fn = transformers.pipeline
    device = pipeline_device(options.device)

    # MMS multilingual models need a Danish language adapter.
    if "mms" in model_ref.lower():
        proc = transformers.AutoProcessor.from_pretrained(model_ref)
        proc.tokenizer.set_target_lang("dan")
        model_obj = transformers.Wav2Vec2ForCTC.from_pretrained(model_ref)
        model_obj.load_adapter("dan")
        if cuda_ok(options.device):
            model_obj = model_obj.to("cuda")
        pipe = pipeline_fn(
            "automatic-speech-recognition",
            model=model_obj,
            tokenizer=proc.tokenizer,
            feature_extractor=proc.feature_extractor,
            device=device,
        )
        return Wav2Vec2Backend(pipe, options=options)

    pipe = pipeline_fn("automatic-speech-recognition", model=model_ref, device=device)
    return Wav2Vec2Backend(pipe, options=options)
