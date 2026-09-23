"""Backend for syvai/hviske-v6 (WhisperQwen). Eval infrastructure, not part of the model repo."""
from __future__ import annotations
import os, sys
from danish_asr_leaderboard.backends._torch_util import cuda_ok
from danish_asr_leaderboard.backends.base import Backend, LoadOptions, register


class HviskeBackend(Backend):
    name = "hviske-v6"

    def transcribe_batch(self, audio_paths, *, batch_size):
        texts = self.model.transcribe(list(audio_paths), batch_size=batch_size)
        if len(texts) != len(audio_paths):
            raise RuntimeError(f"got {len(texts)} for {len(audio_paths)} inputs")
        return [(t or "").strip() for t in texts]

    def transcribe_one(self, audio_path):
        return (self.model.transcribe(audio_path) or "").strip()


@register("hviske-v6")
def load(model_ref: str, options: LoadOptions) -> Backend:
    local = model_ref
    if not os.path.isdir(local):
        from huggingface_hub import snapshot_download
        local = snapshot_download(model_ref)
    if local not in sys.path:
        sys.path.insert(0, local)
    from processing_whisper_qwen import HviskeASR
    device = "cuda" if cuda_ok(options.device) else "cpu"
    return HviskeBackend(HviskeASR.from_pretrained(local, device=device), options=options)
