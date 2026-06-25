"""Cohere ASR backend (custom modelling code on the Hub, via trust_remote_code).

Run on transformers 4.57.x: transformers 5.x regressed remote-model loading
(``list | set`` in ``_adjust_missing_and_unexpected_keys``), so the CohereAsr
remote code fails to load there. The eval sweep routes this backend accordingly.
"""
from __future__ import annotations

import importlib

from danish_asr_leaderboard.backends._torch_util import bf16_dtype, cuda_ok
from danish_asr_leaderboard.backends.base import Backend, LoadOptions, register


class CohereAsrBackend(Backend):
    name = "cohere-asr"

    def __init__(self, model, processor, *, options=None):
        super().__init__(model, options=options)
        self.processor = processor

    def transcribe_batch(self, audio_paths: list[str], *, batch_size: int) -> list[str]:
        import numpy as np

        try:
            import soundfile as sf
        except ImportError:
            import librosa

            sf = None
        results: list[str] = []
        for i in range(0, len(audio_paths), batch_size):
            batch = audio_paths[i : i + batch_size]
            audio_arrays, sample_rates = [], []
            for p in batch:
                if sf is not None:
                    arr, sr = sf.read(p, dtype="float32", always_2d=False)
                else:
                    arr, sr = librosa.load(p, sr=None, mono=True)
                audio_arrays.append(np.asarray(arr, dtype=np.float32))
                sample_rates.append(sr)
            texts = self.model.transcribe(
                processor=self.processor,
                language="da",
                audio_arrays=audio_arrays,
                sample_rates=sample_rates,
            )
            results.extend((t or "").strip() for t in texts)
        return results

    def transcribe_one(self, audio_path: str) -> str:
        return self.transcribe_batch([audio_path], batch_size=1)[0]


@register("cohere-asr")
def load(model_ref: str, options: LoadOptions) -> Backend:
    transformers = importlib.import_module("transformers")
    proc = transformers.AutoProcessor.from_pretrained(model_ref, trust_remote_code=True)
    model_obj = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(
        model_ref, dtype=bf16_dtype(options.device), trust_remote_code=True
    )
    if cuda_ok(options.device):
        model_obj = model_obj.to("cuda")
    model_obj = model_obj.eval()
    return CohereAsrBackend(model_obj, proc, options=options)
