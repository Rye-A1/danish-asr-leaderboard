"""NVIDIA NeMo backends: Canary / Parakeet-TDT (with optional KenLM) and SALM."""
from __future__ import annotations

import importlib
from pathlib import Path

from danish_asr_leaderboard.backends.base import Backend, LoadOptions, register


# ---------------------------------------------------------------------------
# KenLM helpers (Canary beam search)
# ---------------------------------------------------------------------------

def _resolve_kenlm_path(kenlm_model: str) -> str:
    """Local path to a KenLM ``.nemo`` file; accepts ``repo_id:filename`` for HF."""
    p = Path(kenlm_model).expanduser()
    if p.exists():
        return str(p)
    if ":" in kenlm_model:
        repo_id, filename = kenlm_model.split(":", 1)
        from huggingface_hub import hf_hub_download

        print(f"  Downloading KenLM from HF: {repo_id}/{filename}")
        return hf_hub_download(repo_id=repo_id, filename=filename)
    raise FileNotFoundError(
        f"KenLM model not found locally and no ':' separator for HF download: {kenlm_model}"
    )


def _configure_canary_kenlm(model, *, kenlm_path: str, ngram_lm_alpha: float, beam_size: int) -> None:
    from omegaconf import OmegaConf

    if not hasattr(model, "change_decoding_strategy"):
        raise RuntimeError("Loaded model does not expose change_decoding_strategy")
    decoding_cfg = OmegaConf.create({
        "strategy": "beam",
        "beam": {
            "beam_size": beam_size,
            "ngram_lm_model": kenlm_path,
            "ngram_lm_alpha": ngram_lm_alpha,
            "return_best_hypothesis": True,
        },
    })
    model.change_decoding_strategy(decoding_cfg)
    print(f"  KenLM configured: alpha={ngram_lm_alpha}, beam_size={beam_size}")


def _configure_beam(model, beam_size: int) -> None:
    from omegaconf import OmegaConf

    decoding_cfg = OmegaConf.create({
        "strategy": "beam",
        "beam": {"beam_size": beam_size, "return_best_hypothesis": True},
    })
    model.change_decoding_strategy(decoding_cfg)
    print(f"  Beam search configured: beam_size={beam_size} (no LM)")


def _text_of(out) -> str:
    if out is None:
        return ""
    # A NeMo Hypothesis always carries a `.text` attribute — use it even when it's
    # an empty string (silence / nothing decoded). The earlier truthiness check
    # (`if getattr(out, "text", None)`) fell through to `str(out)` on empty text,
    # serialising the whole Hypothesis(...) repr as the "transcription".
    text = getattr(out, "text", None)
    if text is not None:
        return text.strip()
    return str(out).strip()


# ---------------------------------------------------------------------------
# Canary / Parakeet
# ---------------------------------------------------------------------------

class NemoBackend(Backend):
    name = "nemo"

    @property
    def is_parakeet(self) -> bool:
        return self.options.nemo_model_type == "parakeet"

    def transcribe_batch(self, audio_paths: list[str], *, batch_size: int) -> list[str]:
        if self.is_parakeet:
            outputs = self.model.transcribe(audio_paths, batch_size=batch_size)
        else:
            outputs = self.model.transcribe(
                audio_paths, batch_size=batch_size,
                source_lang="da", target_lang="da", pnc="no",
            )
        return [_text_of(o) for o in outputs]

    def transcribe_one(self, audio_path: str) -> str:
        if self.is_parakeet:
            output = self.model.transcribe([audio_path])
        else:
            output = self.model.transcribe([audio_path], source_lang="da", target_lang="da", pnc="no")
        return _text_of(output[0]) if output else ""


def _is_lm_nemo(filename: str) -> bool:
    """Heuristic: a ``.nemo`` that is a KenLM/n-gram language model, not an ASR model.

    Some repos ship the acoustic model *and* its KenLM beam-search LM as separate
    ``.nemo`` archives (e.g. ``RyeAI/krumme-v1`` has ``canary-….nemo`` +
    ``nemo_kenlm_6gram_….nemo``). The LM is passed separately via ``--kenlm-model``,
    so it must be excluded when picking the acoustic model to restore.
    """
    low = Path(filename).name.lower()
    # Only unambiguous LM tokens — a bare "_lm" substring would wrongly match
    # acoustic models like "conformer_lm_ctc.nemo".
    return "kenlm" in low or "ngram" in low or "n_gram" in low


def _hf_nemo_file(model_ref: str) -> str | None:
    """If ``model_ref`` is an HF repo whose only ASR payload is a ``.nemo`` file,
    download it and return the local path; otherwise None.

    ``ASRModel.from_pretrained`` only handles repos NeMo publishes as unpacked
    model cards. Fine-tunes are commonly uploaded as a single raw ``.nemo``
    archive (no ``model_config.yaml``), which ``from_pretrained`` can't restore —
    those need ``hf_hub_download`` + ``restore_from`` instead. KenLM ``.nemo``
    files that ride along in the same repo are ignored (they go via --kenlm-model).
    """
    if "/" not in model_ref or Path(model_ref).expanduser().exists():
        return None
    try:
        from huggingface_hub import HfApi, hf_hub_download

        files = HfApi().list_repo_files(model_ref)
        nemo_files = [f for f in files if f.endswith(".nemo") and not _is_lm_nemo(f)]
        if len(nemo_files) != 1:
            return None  # native model card, or ambiguous — let from_pretrained try
        print(f"  HF repo ships a raw .nemo ({nemo_files[0]}); downloading…")
        return hf_hub_download(repo_id=model_ref, filename=nemo_files[0])
    except Exception:
        return None  # not resolvable as an HF repo — fall back to from_pretrained


@register("nemo")
def load(model_ref: str, options: LoadOptions) -> Backend:
    import torch

    nemo_asr = importlib.import_module("nemo.collections.asr.models")
    ASRModel = nemo_asr.ASRModel
    ref_path = Path(model_ref).expanduser()
    if ref_path.exists() and ref_path.suffix == ".nemo":
        model = ASRModel.restore_from(restore_path=str(ref_path))
    elif (hf_nemo := _hf_nemo_file(model_ref)) is not None:
        model = ASRModel.restore_from(restore_path=hf_nemo)
    else:
        model = ASRModel.from_pretrained(model_name=model_ref)
    if options.device == "cuda" and torch.cuda.is_available():
        model = model.cuda()

    if options.kenlm_model:
        _configure_canary_kenlm(
            model,
            kenlm_path=_resolve_kenlm_path(options.kenlm_model),
            ngram_lm_alpha=options.kenlm_alpha,
            beam_size=options.kenlm_beam_size,
        )
    elif options.nemo_beam_size > 1:
        _configure_beam(model, options.nemo_beam_size)

    return NemoBackend(model, options=options)


# ---------------------------------------------------------------------------
# SALM (e.g. nvidia/canary-qwen-2.5b — English only)
# ---------------------------------------------------------------------------

class NemoSalmBackend(Backend):
    name = "nemo-salm"

    def _prompts(self, audio_paths: list[str]) -> list[list[dict]]:
        tag = self.model.audio_locator_tag
        return [
            [{"role": "user", "content": f"Transcribe the following: {tag}", "audio": [p]}]
            for p in audio_paths
        ]

    def transcribe_batch(self, audio_paths: list[str], *, batch_size: int) -> list[str]:
        answer_ids = self.model.generate(prompts=self._prompts(audio_paths), max_new_tokens=256)
        return [
            self.model.tokenizer.ids_to_text(ids.cpu()).strip() if ids is not None else ""
            for ids in answer_ids
        ]

    def transcribe_one(self, audio_path: str) -> str:
        return self.transcribe_batch([audio_path], batch_size=1)[0]


@register("nemo-salm")
def load_salm(model_ref: str, options: LoadOptions) -> Backend:
    import torch

    salm_mod = importlib.import_module("nemo.collections.speechlm2.models")
    model = salm_mod.SALM.from_pretrained(model_ref)
    if options.device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    return NemoSalmBackend(model, options=options)
