"""Danish decoding for Nemotron Unified and TranscribePsy's non-English route."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile

from danish_asr_leaderboard.backends.base import Backend, LoadOptions, register
from danish_asr_leaderboard.backends.nemo_backend import _text_of


class DanishNemoBackend(Backend):
    def __init__(self, model, *, options, prompt=None, adapter_counts=None, hooks=()):
        super().__init__(model, options=options)
        self.prompt = prompt
        self.adapter_counts = adapter_counts
        self.hooks = hooks

    def transcribe(self, audio_paths: list[str], *, batch_size: int) -> list[str]:
        import soundfile
        import torch

        texts = []
        for start in range(0, len(audio_paths), 400):
            chunk = audio_paths[start:start + 400]
            with torch.inference_mode():
                if self.prompt:
                    # NeMo's Lhotse loader otherwise loses target_lang and can use auto.
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl") as manifest:
                        for path in chunk:
                            json.dump({"audio_filepath": path, "duration": soundfile.info(path).duration,
                                       "text": "", "lang": "da", "prompt_mode": "langID"}, manifest)
                            manifest.write("\n")
                        manifest.flush()
                        outputs = self.model.transcribe([manifest.name], batch_size=batch_size,
                                                        verbose=False, target_lang="da")
                else:
                    outputs = self.model.transcribe(chunk, batch_size=batch_size, verbose=False)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if len(outputs) != len(chunk):
                raise RuntimeError(f"got {len(outputs)} transcripts for {len(chunk)} clips")
            if self.adapter_counts is not None:
                if not all(self.adapter_counts):
                    raise RuntimeError("one or more non-English adapters did not run")
                for hook in self.hooks:
                    hook.remove()
                self.adapter_counts = None
            texts.extend(_text_of(output) for output in outputs)
            print(f"  {len(texts)}/{len(audio_paths)} transcribed", flush=True)
        return texts

    def transcribe_one(self, audio_path: str) -> str:
        return self.transcribe([audio_path], batch_size=1)[0]


def _restore(model_ref: str, options: LoadOptions):
    import torch
    from nemo.collections.asr.models import ASRModel

    if options.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    model = ASRModel.restore_from(restore_path=model_ref, map_location=options.device)
    return model.to(options.device).eval()


def _check_greedy(model):
    strategy = str(model.cfg.decoding.strategy)
    if strategy not in {"greedy", "greedy_batch"}:
        raise RuntimeError(f"expected greedy decoding, got {strategy}")


def _snapshot(model_ref: str, options: LoadOptions, files: list[str]) -> Path:
    path = Path(model_ref).expanduser()
    if path.is_file():
        return path.parent
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_ref, revision=options.model_revision, allow_patterns=files))


@register("nemotron-unified-da")
def load_nemotron(model_ref: str, options: LoadOptions) -> Backend:
    import torch

    filename = "nemotron-labs-unified-asr-0.6b.nemo"
    path = Path(model_ref).expanduser()
    checkpoint = path if path.is_file() else _snapshot(model_ref, options, [filename]) / filename
    model = _restore(str(checkpoint), options)
    _check_greedy(model)
    expected = model.cfg.model_defaults.prompt_dictionary["da"]
    if expected == model.cfg.model_defaults.prompt_dictionary["auto"]:
        raise RuntimeError("Danish and automatic prompt indices must differ")
    original_forward = model._transcribe_forward

    def checked_forward(batch, trcfg):
        if (trcfg.target_lang != "da" or len(batch) < 5 or
                not torch.all(batch[4] == expected).item()):
            raise RuntimeError("transcription batch did not use the Danish prompt")
        return original_forward(batch, trcfg)

    model._transcribe_forward = checked_forward
    return DanishNemoBackend(model, options=options, prompt="da")


@register("transcribepsy-da")
def load_transcribepsy(model_ref: str, options: LoadOptions) -> Backend:
    import torch

    filename = "TranscribePsy-250M-EU.nemo"
    root = _snapshot(model_ref, options, [filename, "inference.py", "adapters.safetensors"])
    path = Path(model_ref).expanduser()
    checkpoint = path if path.is_file() else root / filename
    spec = importlib.util.spec_from_file_location("transcribepsy_official", root / "inference.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load TranscribePsy inference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = _restore(str(checkpoint), options)
    dtype = torch.bfloat16 if options.device == "cuda" else torch.float32
    model = model.to(dtype).eval()
    module.safe_decoding(model, cuda_graph=False)
    _check_greedy(model)
    module.attach_adapters(model, str(root / "adapters.safetensors"),
                           device=model.device, dtype=next(model.parameters()).dtype)
    if len(model._adapters) != 6 or len(model._adapter_handles) != 6:
        raise RuntimeError("expected six non-English adapters")
    counts = [0] * 6
    hooks = []
    for index, adapter in enumerate(model._adapters):
        def count(_module, _inputs, _output, *, index=index):
            counts[index] += 1
        hooks.append(adapter.register_forward_hook(count))
    return DanishNemoBackend(model, options=options, adapter_counts=counts, hooks=hooks)
