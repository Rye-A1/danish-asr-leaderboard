"""CTC ASR (wav2vec2 / MMS) via the HF ``transformers`` pipeline."""
from __future__ import annotations

import importlib
from pathlib import Path

from danish_asr_leaderboard.backends._torch_util import cuda_ok, pipeline_device
from danish_asr_leaderboard.backends.base import Backend, LoadOptions, register

# A CTC repo that ships an n-gram LM stores it under this prefix, alongside the
# pyctcdecode artefacts (alphabet.json, unigrams.txt).
_LM_DIR = "language_model/"


class Wav2Vec2Backend(Backend):
    name = "wav2vec2"

    def transcribe_batch(self, audio_paths: list[str], *, batch_size: int) -> list[str]:
        # CTC models have no autoregressive generation kwargs.
        raw = self.model(audio_paths, batch_size=batch_size)
        return [((r or {}).get("text", "").strip() if r else "") for r in raw]

    def transcribe_one(self, audio_path: str) -> str:
        result = self.model(audio_path)
        return (result or {}).get("text", "").strip() if result else ""


def _ships_lm(model_ref: str) -> bool:
    """Whether the repo (or local dir) carries a pyctcdecode n-gram LM."""
    local = Path(model_ref).expanduser()
    if local.is_dir():
        return (local / "language_model").is_dir()
    try:
        from huggingface_hub import HfApi

        return any(f.startswith(_LM_DIR) for f in HfApi().list_repo_files(model_ref))
    except Exception:
        return False  # not resolvable as a repo — treat as no LM


def _processor_with_lm(transformers, model_ref):
    """Load the LM-aware processor, failing loudly if the decoder deps are absent.

    ``AutoProcessor`` silently returns a plain (LM-free) processor when
    pyctcdecode/kenlm are missing, which would score the model on greedy CTC
    output while its published numbers assume n-gram decoding — a large, silent
    handicap. Models shipping an LM are meant to be run with it.
    """
    try:
        importlib.import_module("pyctcdecode")
        importlib.import_module("kenlm")
    except ImportError as exc:
        raise RuntimeError(
            f"{model_ref} ships an n-gram language model under {_LM_DIR}, but "
            "pyctcdecode and/or kenlm are not installed, so transformers would "
            "silently fall back to greedy CTC decoding and under-report the "
            "model. Install the wav2vec2 extra (pip install -e '.[wav2vec2]')."
        ) from exc
    return transformers.Wav2Vec2ProcessorWithLM.from_pretrained(model_ref)


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

    # Repos shipping an n-gram LM are decoded with it — that is how the model is
    # intended to run, and how its published numbers were produced.
    if _ships_lm(model_ref):
        proc = _processor_with_lm(transformers, model_ref)
        print(f"  n-gram LM decoding enabled ({_LM_DIR} found)")
        pipe = pipeline_fn(
            "automatic-speech-recognition",
            model=model_ref,
            tokenizer=proc.tokenizer,
            feature_extractor=proc.feature_extractor,
            decoder=proc.decoder,
            device=device,
        )
        return Wav2Vec2Backend(pipe, options=options)

    pipe = pipeline_fn("automatic-speech-recognition", model=model_ref, device=device)
    return Wav2Vec2Backend(pipe, options=options)
