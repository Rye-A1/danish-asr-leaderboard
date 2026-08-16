"""Whisper-family (and other seq2seq) ASR via the HF ``transformers`` pipeline."""
from __future__ import annotations

import importlib

from danish_asr_leaderboard.audio import wav_duration
from danish_asr_leaderboard.backends._torch_util import half_dtype, pipeline_device
from danish_asr_leaderboard.backends.base import Backend, LoadOptions, register

_GEN_KWARGS = {"task": "transcribe", "language": "da"}

# Whisper's encoder window. Clips at or under this are short-form; longer ones
# need the sequential long-form path.
_SHORT_FORM_MAX_S = 30.0

# Short-form and long-form are decoded differently, matching the two reference
# implementations:
#
#   * HF Open ASR Leaderboard (transformers/run_eval.py) branches on --longform:
#       longform -> model.generate(..., return_timestamps=True)
#       else     -> model.generate(...)                      # no timestamps
#   * CoRal's evaluate_model.py uses the plain short-form call.
#
# We previously passed return_timestamps=True (plus max_new_tokens=440) for
# *every* clip. On short audio that sends Whisper into repetition loops — a
# one-word reference ("hvis") producing 100+ repeated tokens, sometimes switching
# language ("and the best and the best ...") — and one such utterance contributes
# thousands of percent WER. Applying the long-form tool to short-form audio was
# the bug; our corpus is ~99% short-form (>30 s: FTSpeech 1.2%, FLEURS 0.1%,
# nothing elsewhere).
#
# Routing by duration keeps each regime on its reference decoding path, and is
# strictly better than either reference alone: HF simply truncates long clips in
# short-form mode, whereas we still transcribe them in full.


def _extract(result: dict) -> str:
    if not result:
        return ""
    if "chunks" in result:
        return " ".join(c["text"] for c in result["chunks"]).strip()
    return (result or {}).get("text", "").strip()


class TransformersBackend(Backend):
    name = "transformers"

    def _short_form(self, audio_paths: list[str], *, batch_size: int) -> list[str]:
        raw = self.model(audio_paths, batch_size=batch_size, generate_kwargs=_GEN_KWARGS)
        return [_extract(r) for r in raw]

    def _long_form(self, audio_paths: list[str]) -> list[str]:
        # One at a time: batching variable-length >30 s clips trips the pipeline's
        # feature padding. They are a fraction of a percent, so the cost is noise.
        return [
            _extract(self.model(p, return_timestamps=True, generate_kwargs=_GEN_KWARGS))
            for p in audio_paths
        ]

    def transcribe_batch(self, audio_paths: list[str], *, batch_size: int) -> list[str]:
        # An unreadable duration reads as 0.0 and routes to short-form, which is
        # the overwhelmingly common case and the one without the loop pathology.
        long_idx = [i for i, p in enumerate(audio_paths)
                    if wav_duration(p) > _SHORT_FORM_MAX_S]
        if not long_idx:
            return self._short_form(audio_paths, batch_size=batch_size)

        long_set = set(long_idx)
        short_idx = [i for i in range(len(audio_paths)) if i not in long_set]
        out: list[str] = [""] * len(audio_paths)
        if short_idx:
            for i, text in zip(
                short_idx, self._short_form([audio_paths[i] for i in short_idx],
                                            batch_size=batch_size)
            ):
                out[i] = text
        for i, text in zip(long_idx, self._long_form([audio_paths[i] for i in long_idx])):
            out[i] = text
        return out

    def transcribe_one(self, audio_path: str) -> str:
        return self.transcribe_batch([audio_path], batch_size=1)[0]


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
