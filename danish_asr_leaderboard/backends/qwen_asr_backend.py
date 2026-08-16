"""Qwen3-ASR backend (also used for Qwen3-ASR fine-tunes, e.g. Saga)."""
from __future__ import annotations

import importlib

from danish_asr_leaderboard.backends._torch_util import device_map
from danish_asr_leaderboard.backends.base import Backend, LoadOptions, register

_LANGUAGE = "Danish"
_MAX_INFERENCE_BATCH = 32  # upper bound the qwen_asr lib batches to internally

# Generation cap. 256 tokens truncated the longest references: measured on the
# saved outputs, refs >=150 words came back at ~0.85 of their length against
# ~0.99 for an uncapped backend. The effect is small on the current corpus
# (32 utterances >=100 words, <=0.06pp WER) but it is a silent ceiling that
# would bite on a corpus with longer utterances, so it is raised to match the
# voxtral backend, which shows no truncation.
_MAX_NEW_TOKENS = 440


def _text(result) -> str:
    return (getattr(result, "text", "") or "").strip() if result is not None else ""


class QwenAsrBackend(Backend):
    name = "qwen-asr"

    def transcribe_batch(self, audio_paths: list[str], *, batch_size: int) -> list[str]:
        # qwen_asr.transcribe takes a list and batches internally (up to the model's
        # max_inference_batch_size). Chunk here to bound peak memory — the lib loads
        # every passed clip up front.
        out: list[str] = []
        for i in range(0, len(audio_paths), batch_size):
            chunk = audio_paths[i : i + batch_size]
            results = self.model.transcribe(audio=chunk, language=_LANGUAGE)
            if len(results) != len(chunk):
                # Guard against silent hyp↔ref misalignment; raising lets
                # Backend.transcribe fall back to the per-file path.
                raise RuntimeError(
                    f"qwen-asr returned {len(results)} results for {len(chunk)} inputs"
                )
            out.extend(_text(r) for r in results)
        return out

    def transcribe_one(self, audio_path: str) -> str:
        results = self.model.transcribe(audio=audio_path, language=_LANGUAGE)
        return _text(results[0]) if results else ""


@register("qwen-asr")
def load(model_ref: str, options: LoadOptions) -> Backend:
    import torch

    qwen_asr = importlib.import_module("qwen_asr")
    model = qwen_asr.Qwen3ASRModel.from_pretrained(
        model_ref,
        dtype=torch.bfloat16,
        device_map=device_map(options.device),
        max_inference_batch_size=_MAX_INFERENCE_BATCH,
        max_new_tokens=_MAX_NEW_TOKENS,
    )
    return QwenAsrBackend(model, options=options)
