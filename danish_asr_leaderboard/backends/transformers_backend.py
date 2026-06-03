"""Whisper-family (and other seq2seq) ASR via the HF ``transformers`` pipeline."""
from __future__ import annotations

import importlib

from danish_asr_leaderboard.backends._torch_util import half_dtype, pipeline_device
from danish_asr_leaderboard.backends.base import Backend, LoadOptions, register

_GEN_KWARGS = {"task": "transcribe", "language": "da"}


def _extract(result: dict) -> str:
    if not result:
        return ""
    if "chunks" in result:
        return " ".join(c["text"] for c in result["chunks"]).strip()
    return (result or {}).get("text", "").strip()


class TransformersBackend(Backend):
    name = "transformers"

    def transcribe_batch(self, audio_paths: list[str], *, batch_size: int) -> list[str]:
        raw = self.model(
            audio_paths,
            batch_size=batch_size,
            return_timestamps=True,
            max_new_tokens=440,
            generate_kwargs=_GEN_KWARGS,
        )
        return [_extract(r) for r in raw]

    def transcribe_one(self, audio_path: str) -> str:
        result = self.model(
            audio_path,
            return_timestamps=True,
            max_new_tokens=440,
            generate_kwargs=_GEN_KWARGS,
        )
        return _extract(result)


@register("transformers")
def load(model_ref: str, options: LoadOptions) -> Backend:
    import torch

    transformers = importlib.import_module("transformers")
    AutoModel = transformers.AutoModelForSpeechSeq2Seq
    AutoProcessor = transformers.AutoProcessor
    pipeline_fn = transformers.pipeline

    torch_dtype = half_dtype(options.device)
    try:
        model = AutoModel.from_pretrained(model_ref, torch_dtype=torch_dtype)
    except ValueError:
        # Whisper fine-tunes whose config.json lacks model_type
        model = transformers.WhisperForConditionalGeneration.from_pretrained(
            model_ref, torch_dtype=torch_dtype
        )
    if options.device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
    processor = AutoProcessor.from_pretrained(model_ref)
    pipe = pipeline_fn(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=pipeline_device(options.device),
    )
    return TransformersBackend(pipe, options=options)
