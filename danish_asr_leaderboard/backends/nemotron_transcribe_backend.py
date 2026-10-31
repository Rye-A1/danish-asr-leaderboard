"""NVIDIA Nemotron 3.5 Transcribe through its native Transformers processor."""
from __future__ import annotations

from danish_asr_leaderboard.audio import load_audio_array
from danish_asr_leaderboard.backends._torch_util import cuda_ok
from danish_asr_leaderboard.backends.base import Backend, LoadOptions, register

TRANSCRIPTION_PROMPT = (
    "Audio transcription. Language: Danish. Wording: verbatim, including fillers and "
    "repetitions. Formatting: punctuation and capitalization. Output: transcript only."
)
MAX_NEW_TOKENS = 1024


class NemotronTranscribeBackend(Backend):
    name = "nemotron-transcribe"

    def __init__(self, model, processor, *, options=None):
        super().__init__(model, options=options)
        self.processor = processor
        self.fail_on_error = True

    def transcribe_batch(self, audio_paths: list[str], *, batch_size: int) -> list[str]:
        import torch

        texts = []
        for start in range(0, len(audio_paths), batch_size):
            audio = [load_audio_array(path) for path in audio_paths[start:start + batch_size]]
            inputs = self.processor.apply_transcription_request(
                audio, prompt=TRANSCRIPTION_PROMPT
            ).to(self.model.device)
            with torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            decoded = self.processor.batch_decode(
                output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
            )
            if len(decoded) != len(audio):
                raise RuntimeError(f"got {len(decoded)} transcripts for {len(audio)} inputs")
            texts.extend(text.strip() for text in decoded)
            if len(texts) % 400 == 0 or len(texts) == len(audio_paths):
                print(f"  {len(texts)}/{len(audio_paths)} transcribed", flush=True)
        return texts

    def transcribe_one(self, audio_path: str) -> str:
        return self.transcribe_batch([audio_path], batch_size=1)[0]


@register("nemotron-transcribe")
def load(model_ref: str, options: LoadOptions) -> Backend:
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    revision = {"revision": options.model_revision} if options.model_revision else {}
    processor = AutoProcessor.from_pretrained(model_ref, **revision)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_ref, device_map="auto" if cuda_ok(options.device) else "cpu", dtype="auto", **revision
    ).eval()
    device_map = getattr(model, "hf_device_map", {})
    devices = sorted({str(d) for d in device_map.values()}) if device_map else [str(model.device)]
    print(f"  nemotron-transcribe devices: {devices}")
    return NemotronTranscribeBackend(model, processor, options=options)
