"""Saga 2 backend (capacit-ai/saga-2-*).

Saga 2 is a custom ``WQwenForASR`` architecture — a RoPE-patched Whisper encoder
feeding a Qwen3 causal LM — so it cannot be loaded through ``AutoModel`` or the
``qwen-asr`` library used for Saga 1. The model repo ships its own inference
package (``saga2/``) next to the weights; the loader downloads the snapshot and
imports that package from it, so the decode used here is exactly the one the model
card documents (greedy, per-clip token cap, repeated-token loop guard, and
silence-aware chunking for audio over 30 s).
"""
from __future__ import annotations

import importlib
import os
import sys

from danish_asr_leaderboard.backends._torch_util import cuda_ok
from danish_asr_leaderboard.backends.base import Backend, LoadOptions, register


class Saga2Backend(Backend):
    name = "saga2"

    def transcribe_batch(self, audio_paths: list[str], *, batch_size: int) -> list[str]:
        # Saga2.transcribe takes a list of paths, length-sorts the clips and batches
        # internally, returning one string per input in the original order.
        texts = self.model.transcribe(list(audio_paths), batch_size=batch_size)
        if len(texts) != len(audio_paths):
            # Guard against silent hyp↔ref misalignment; raising lets
            # Backend.transcribe fall back to the per-file path.
            raise RuntimeError(
                f"saga2 returned {len(texts)} transcripts for {len(audio_paths)} inputs"
            )
        return [(t or "").strip() for t in texts]

    def transcribe_one(self, audio_path: str) -> str:
        return (self.model.transcribe(audio_path) or "").strip()


def _resolve_snapshot(model_ref: str) -> str:
    """Local directory holding the weights *and* the ``saga2`` package."""
    if os.path.isdir(model_ref):
        return model_ref
    from huggingface_hub import snapshot_download

    return snapshot_download(model_ref)


@register("saga2")
def load(model_ref: str, options: LoadOptions) -> Backend:
    local_dir = _resolve_snapshot(model_ref)
    if not os.path.isfile(os.path.join(local_dir, "saga2", "__init__.py")):
        raise FileNotFoundError(
            f"{model_ref} does not ship the saga2/ inference package; "
            "the saga2 backend only supports capacit-ai/saga-2-* repos"
        )
    # The inference code lives in the model repo (like run.py does it), so import
    # it from the snapshot rather than vendoring a copy that could drift.
    if local_dir not in sys.path:
        sys.path.insert(0, local_dir)
    saga2 = importlib.import_module("saga2")

    device = "cuda" if cuda_ok(options.device) else "cpu"
    model = saga2.Saga2(local_dir, device=device)  # bf16 on GPU, fp32 on CPU
    return Saga2Backend(model, options=options)
