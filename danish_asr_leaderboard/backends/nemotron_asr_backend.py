"""NVIDIA Nemotron 3.5 ASR (cache-aware FastConformer-RNNT) via native transformers.

The checkpoint ships a transformers export next to its ``.nemo`` (transformers
>=5.13), so no NeMo install is needed. The model is multilingual and conditions on
a language-ID prompt: without one the processor defaults to ``auto`` detection,
so the Danish prompt (``da-DK``) is passed explicitly. Transcription is offline
(full utterance), in float32 like the NeMo backends, with the largest supported
right context (13 frames = 1120 ms chunks): the model's most accurate setting,
since the leaderboard scores whole utterances rather than live latency. The
transformers default is 3 (320 ms), 2.5 WER points worse on FLEURS-da per the
model card.
"""
from __future__ import annotations

import importlib

from danish_asr_leaderboard.backends._torch_util import cuda_ok
from danish_asr_leaderboard.backends.base import Backend, LoadOptions, register

LANGUAGE = "da-DK"
NUM_LOOKAHEAD_TOKENS = 13


class NemotronAsrBackend(Backend):
    name = "nemotron-asr"

    def __init__(self, model, processor, *, options=None):
        super().__init__(model, options=options)
        self.processor = processor

    def transcribe_batch(self, audio_paths: list[str], *, batch_size: int) -> list[str]:
        import soundfile as sf

        sr = self.processor.feature_extractor.sampling_rate
        results: list[str] = []
        for i in range(0, len(audio_paths), batch_size):
            audios = [sf.read(p, dtype="float32")[0] for p in audio_paths[i : i + batch_size]]
            inputs = self.processor(audios, sampling_rate=sr, language=LANGUAGE)
            inputs = inputs.to(self.model.device, dtype=self.model.dtype)
            output = self.model.generate(**inputs, return_dict_in_generate=True)
            texts = self.processor.batch_decode(output.sequences, skip_special_tokens=True)
            results.extend(t.strip() for t in texts)
        return results

    def transcribe_one(self, audio_path: str) -> str:
        return self.transcribe_batch([audio_path], batch_size=1)[0]


@register("nemotron-asr")
def load(model_ref: str, options: LoadOptions) -> Backend:
    transformers = importlib.import_module("transformers")
    processor = transformers.AutoProcessor.from_pretrained(model_ref)
    processor.set_num_lookahead_tokens(NUM_LOOKAHEAD_TOKENS)
    model = transformers.AutoModelForRNNT.from_pretrained(model_ref, dtype="float32")
    if cuda_ok(options.device):
        model = model.to("cuda")
    return NemotronAsrBackend(model.eval(), processor, options=options)
