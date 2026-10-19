"""Generic backend for models whose decoding code ships in their own Hub repo (``trust_remote_code``).

The repo's config maps ``AutoModelForSpeechSeq2Seq`` (and, if it needs one, ``AutoProcessor``) to its code,
and the model exposes

    model.transcribe(processor=..., language="da", audio_arrays=[...], sample_rates=[...]) -> list[str]

taking 16 kHz mono float32 arrays and returning one transcript per array. The weights are loaded in the
dtype the repo's config declares (``dtype="auto"``), so the model card, not the harness, decides the
precision; on CPU everything runs in float32.
"""
from __future__ import annotations

import re
from pathlib import Path

from danish_asr_leaderboard.audio import load_audio_array
from danish_asr_leaderboard.backends._torch_util import cuda_ok
from danish_asr_leaderboard.backends.base import Backend, LoadOptions, register


class TransformersRemoteBackend(Backend):
    name = "transformers-remote"

    def __init__(self, model, processor, *, options: LoadOptions | None = None):
        super().__init__(model, options=options)
        self.processor = processor

    def transcribe_batch(self, audio_paths: list[str], *, batch_size: int) -> list[str]:
        out: list[str] = []
        for i in range(0, len(audio_paths), batch_size):
            audios = [load_audio_array(p) for p in audio_paths[i:i + batch_size]]
            texts = self.model.transcribe(processor=self.processor, language="da",
                                          audio_arrays=audios, sample_rates=[16000] * len(audios))
            if len(texts) != len(audios):
                raise RuntimeError(f"got {len(texts)} transcripts for {len(audios)} inputs")
            out.extend((t or "").strip() for t in texts)
        return out

    def transcribe_one(self, audio_path: str) -> str:
        return self.transcribe_batch([audio_path], batch_size=1)[0]


def _pinned_revision(model_ref: str, options: LoadOptions) -> str | None:
    if Path(model_ref).is_dir():
        return None
    revision = options.model_revision
    if not revision or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError(
            "transformers-remote executes Hub code; pass --model-revision "
            "with its full 40-character commit SHA"
        )
    return revision.lower()


@register("transformers-remote")
def load(model_ref: str, options: LoadOptions) -> Backend:
    revision = _pinned_revision(model_ref, options)
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    gpu = cuda_ok(options.device)
    pinned = {"revision": revision} if revision else {}
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_ref, trust_remote_code=True, dtype="auto" if gpu else torch.float32, **pinned
    )
    model = model.to("cuda" if gpu else "cpu").eval()
    print(f"  transformers-remote: {type(model).__name__}, dtype {model.dtype}")
    processor = AutoProcessor.from_pretrained(model_ref, trust_remote_code=True, **pinned)
    return TransformersRemoteBackend(model, processor, options=options)
