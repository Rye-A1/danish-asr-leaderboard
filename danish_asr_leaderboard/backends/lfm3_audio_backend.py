"""Liquid LFM3 Audio transcription using its documented Danish ASR prompt."""
from __future__ import annotations

from danish_asr_leaderboard.backends._torch_util import bf16_dtype, device_map
from danish_asr_leaderboard.backends.base import Backend, LoadOptions, register

ASR_PROMPT = "Perform ASR in Danish."
MAX_NEW_TOKENS = 1024


class Lfm3AudioBackend(Backend):
    name = "lfm3-audio"

    def __init__(self, model, processor, *, options=None):
        super().__init__(model, options=options)
        self.processor = processor
        self.fail_on_error = True

    def transcribe_batch(self, audio_paths: list[str], *, batch_size: int) -> list[str]:
        import torch

        texts = []
        for start in range(0, len(audio_paths), batch_size):
            chunk = audio_paths[start:start + batch_size]
            conversations = [
                [
                    {"role": "system", "content": ASR_PROMPT},
                    {"role": "user", "content": [{"type": "audio", "audio": path}]},
                ]
                for path in chunk
            ]
            inputs = self.processor.apply_chat_template(
                conversations, add_generation_prompt=True, tokenize=True,
                processor_kwargs={"padding": True}, return_dict=True, return_tensors="pt",
            ).to(self.model.device)
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                    suppress_mixed=True, suppress_audio_out=True, return_dict_in_generate=True,
                )
            decoded = self.processor.batch_decode(
                output.sequences[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            if len(decoded) != len(chunk):
                raise RuntimeError(f"got {len(decoded)} transcripts for {len(chunk)} inputs")
            texts.extend(text.strip() for text in decoded)
            if len(texts) % 400 == 0 or len(texts) == len(audio_paths):
                print(f"  {len(texts)}/{len(audio_paths)} transcribed", flush=True)
        return texts

    def transcribe_one(self, audio_path: str) -> str:
        return self.transcribe_batch([audio_path], batch_size=1)[0]


@register("lfm3-audio")
def load(model_ref: str, options: LoadOptions) -> Backend:
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    revision = {"revision": options.model_revision} if options.model_revision else {}
    processor = AutoProcessor.from_pretrained(model_ref, trust_remote_code=True, **revision)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForMultimodalLM.from_pretrained(
        model_ref, trust_remote_code=True, dtype=bf16_dtype(options.device),
        device_map=device_map(options.device), **revision,
    ).eval()
    return Lfm3AudioBackend(model, processor, options=options)
