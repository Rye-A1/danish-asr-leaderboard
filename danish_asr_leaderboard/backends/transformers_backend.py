"""Whisper-family (and other seq2seq) ASR via the HF ``transformers`` pipeline."""
from __future__ import annotations

import importlib

from danish_asr_leaderboard.backends._torch_util import half_dtype, pipeline_device
from danish_asr_leaderboard.backends.base import Backend, LoadOptions, register

_GEN_KWARGS = {"task": "transcribe", "language": "da"}

# Long audio is handled by chunking, NOT by `return_timestamps=True`.
#
# The pipeline previously ran with `return_timestamps=True, max_new_tokens=440`.
# That handled >30 s clips, but on *short* clips it sent Whisper into repetition
# loops: a one-word reference ("hvis") produced 100+ repeated tokens, sometimes
# switching language ("and the best and the best ..."), and a single such
# utterance contributes thousands of percent WER. It affected every
# Whisper-family model on the board — 1.19% of FTSpeech utterances for
# roest-v3-whisper-1.5b (21.06 -> 16.40 mean WER once excluded), and it is why
# whisper-tiny/base scored above 100%.
#
# Dropping both args (as the CoRal eval script does) fixes the loops but errors
# out on >30 s audio. `chunk_length_s=30` fixes both: measured 0/66 loops on the
# previously-looping clips and a 1.00 hypothesis/reference length ratio on >30 s
# clips. ~1.2% of FTSpeech and ~0.1% of FLEURS exceed 30 s; nothing else does.
_CHUNK_LENGTH_S = 30


def _extract(result: dict) -> str:
    if not result:
        return ""
    if "chunks" in result:
        return " ".join(c["text"] for c in result["chunks"]).strip()
    return (result or {}).get("text", "").strip()


class TransformersBackend(Backend):
    name = "transformers"

    def transcribe_batch(self, audio_paths: list[str], *, batch_size: int) -> list[str]:
        raw = self.model(
            audio_paths,
            batch_size=batch_size,
            chunk_length_s=_CHUNK_LENGTH_S,
            generate_kwargs=_GEN_KWARGS,
        )
        return [_extract(r) for r in raw]

    def transcribe_one(self, audio_path: str) -> str:
        result = self.model(
            audio_path,
            chunk_length_s=_CHUNK_LENGTH_S,
            generate_kwargs=_GEN_KWARGS,
        )
        return _extract(result)


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
