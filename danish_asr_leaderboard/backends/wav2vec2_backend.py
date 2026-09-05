"""CTC ASR (wav2vec2 / MMS) via the HF ``transformers`` pipeline.

Repos that ship a pyctcdecode n-gram LM under ``language_model/`` are decoded
with it — that is part of the released artifact, it is what the documented
``pipeline(model=…)`` call does, and it is what the model's own evaluation uses
(CoRal's eval defaults to ``no_lm: false``). On the Røst models it is worth ~6pp
WER, so decoding without it would materially misrepresent them.

The pipeline cannot be used for that path. It hands pyctcdecode the *padded*
logits array, and beam search happily turns batch padding into confident words —
MediaCatch produced "subjekt battistiofrenes" for a reference of "subjekt B" at
batch 16, "subjekt battisti" at batch 4, and a clean "subjekt b" at batch 1, so
the garbage scales with padding length. This module therefore runs the forward
pass itself and trims each item's logits to its true encoder-frame length (via
``_get_feat_extract_output_lengths``) before decoding.

Beam search is CPU-bound, so decoding is spread over a process pool.
"""
from __future__ import annotations

import importlib
import multiprocessing as mp
import os
from pathlib import Path

from danish_asr_leaderboard.backends._torch_util import cuda_ok, pipeline_device
from danish_asr_leaderboard.backends.base import Backend, LoadOptions, register

_LM_DIR = "language_model/"
_SAMPLE_RATE = 16_000
# pyctcdecode releases the GIL poorly; a pool keeps beam search off the critical
# path. Capped because each worker holds its own copy of the LM.
_DECODE_WORKERS = min(8, os.cpu_count() or 1)


class Wav2Vec2Backend(Backend):
    """Greedy CTC via the HF pipeline (models that ship no n-gram LM)."""

    name = "wav2vec2"

    def transcribe_batch(self, audio_paths: list[str], *, batch_size: int) -> list[str]:
        raw = self.model(audio_paths, batch_size=batch_size)
        return [((r or {}).get("text", "").strip() if r else "") for r in raw]

    def transcribe_one(self, audio_path: str) -> str:
        result = self.model(audio_path)
        return (result or {}).get("text", "").strip() if result else ""


class Wav2Vec2LMBackend(Backend):
    """CTC + shipped n-gram LM, with per-item logit trimming."""

    name = "wav2vec2-lm"

    def __init__(self, model, processor, *, device: str, options=None):
        super().__init__(model, options=options)
        self.processor = processor
        self.device = device

    def _read(self, path: str):
        import numpy as np
        import soundfile as sf

        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != _SAMPLE_RATE:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=_SAMPLE_RATE)
        return np.asarray(audio, dtype="float32")

    def transcribe_batch(self, audio_paths: list[str], *, batch_size: int) -> list[str]:
        import torch

        out: list[str] = []
        ctx = mp.get_context("fork")
        with ctx.Pool(_DECODE_WORKERS) as pool:
            for i in range(0, len(audio_paths), batch_size):
                chunk = audio_paths[i : i + batch_size]
                audios = [self._read(p) for p in chunk]
                inputs = self.processor(
                    audios,
                    sampling_rate=_SAMPLE_RATE,
                    return_tensors="pt",
                    padding="longest",
                    return_attention_mask=True,
                )
                # Match the checkpoint's dtype — Røst ships bf16 weights while
                # MediaCatch ships fp32, and a mismatch raises in the conv stack.
                dtype = next(self.model.parameters()).dtype
                input_values = inputs.input_values.to(self.device, dtype=dtype)
                mask = inputs.attention_mask.to(self.device)
                with torch.no_grad():
                    logits = self.model(input_values, attention_mask=mask).logits
                # True encoder-frame count per item, so beam search never sees
                # padding. This is the whole point of not using the pipeline.
                frames = self.model._get_feat_extract_output_lengths(
                    mask.sum(-1)
                ).cpu().numpy()
                lg = logits.float().cpu().numpy()
                trimmed = [lg[j, : int(frames[j]), :] for j in range(len(chunk))]
                # alpha/beta come from the repo's language_model/attrs.json — the
                # values the model author tuned, not ours.
                out.extend(t.strip() for t in self.processor.decoder.decode_batch(pool, trimmed))
        return out

    def transcribe_one(self, audio_path: str) -> str:
        return self.transcribe_batch([audio_path], batch_size=1)[0]


def ships_lm(model_ref: str) -> bool:
    """Whether the repo (or local dir) carries a pyctcdecode n-gram LM."""
    local = Path(model_ref).expanduser()
    if local.is_dir():
        return (local / "language_model").is_dir()
    try:
        from huggingface_hub import HfApi

        return any(f.startswith(_LM_DIR) for f in HfApi().list_repo_files(model_ref))
    except Exception:
        return False


@register("wav2vec2")
def load(model_ref: str, options: LoadOptions) -> Backend:
    transformers = importlib.import_module("transformers")
    device = "cuda" if cuda_ok(options.device) else "cpu"

    # MMS multilingual models need a Danish language adapter (and ship no LM).
    if "mms" in model_ref.lower():
        proc = transformers.AutoProcessor.from_pretrained(model_ref)
        proc.tokenizer.set_target_lang("dan")
        model_obj = transformers.Wav2Vec2ForCTC.from_pretrained(model_ref)
        model_obj.load_adapter("dan")
        if device == "cuda":
            model_obj = model_obj.to("cuda")
        pipe = transformers.pipeline(
            "automatic-speech-recognition",
            model=model_obj,
            tokenizer=proc.tokenizer,
            feature_extractor=proc.feature_extractor,
            device=pipeline_device(options.device),
        )
        return Wav2Vec2Backend(pipe, options=options)

    if ships_lm(model_ref):
        try:
            importlib.import_module("pyctcdecode")
            importlib.import_module("kenlm")
        except ImportError as exc:
            raise RuntimeError(
                f"{model_ref} ships an n-gram LM under {_LM_DIR}, which is worth "
                "several pp WER on these models, but pyctcdecode/kenlm are not "
                "installed. Install the wav2vec2 extra: pip install -e '.[wav2vec2]'"
            ) from exc
        proc = transformers.Wav2Vec2ProcessorWithLM.from_pretrained(model_ref)
        model_obj = transformers.Wav2Vec2ForCTC.from_pretrained(model_ref)
        if device == "cuda":
            model_obj = model_obj.to("cuda")
        model_obj.eval()
        print(f"  n-gram LM decoding enabled ({_DECODE_WORKERS} decode workers)")
        return Wav2Vec2LMBackend(model_obj, proc, device=device, options=options)

    pipe = transformers.pipeline(
        "automatic-speech-recognition", model=model_ref,
        device=pipeline_device(options.device),
    )
    return Wav2Vec2Backend(pipe, options=options)
