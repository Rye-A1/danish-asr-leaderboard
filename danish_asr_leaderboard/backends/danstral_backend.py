"""Backend for hinge/danstral-v1. Eval infrastructure, not part of the model repo.

Danstral is a LoRA adapter on Voxtral-Small-24B whose audio tower is replaced by
the Whisper encoder of CoRal's roest-v1-whisper-1.5b, following the model card:
load Voxtral, copy in the Whisper encoder, then apply the adapter. Base and
encoder are pinned to the revisions the leaderboard scored.

The adapter was saved with an older transformers module layout. A key that fails
to map leaves its LoRA delta at zero without an error, so loading checks that
every adapter tensor landed and refuses to score a partially applied adapter.
"""
from __future__ import annotations

import importlib

from danish_asr_leaderboard.backends._torch_util import bf16_dtype, device_map
from danish_asr_leaderboard.backends.base import Backend, LoadOptions, register
from danish_asr_leaderboard.backends.voxtral_backend import VoxtralBackend

BASE = "mistralai/Voxtral-Small-24B-2507"
BASE_REVISION = "da5b42409f279fdd92febee0511a6c32828569c1"
ENCODER = "CoRal-project/roest-v1-whisper-1.5b"
ENCODER_REVISION = "a6c1e24d9f10e6289607a1ba32341b68e8660688"


class DanstralBackend(VoxtralBackend):
    name = "danstral"


def _check_adapter_loaded(peft_model, adapter_dir: str) -> int:
    """Every lora_B in the adapter file must be non-zero in the model."""
    import os

    from safetensors import safe_open

    with safe_open(os.path.join(adapter_dir, "adapter_model.safetensors"), "pt") as f:
        expected = sum(1 for k in f.keys() if k.endswith("lora_B.weight"))
    loaded = [
        name for name, p in peft_model.named_parameters()
        if "lora_B" in name and bool(p.detach().abs().sum() > 0)
    ]
    if len(loaded) != expected:
        raise RuntimeError(
            f"danstral adapter only partly applied: {len(loaded)} of {expected} LoRA "
            "modules carry weights (module names did not match this transformers version)"
        )
    return expected


@register("danstral")
def load(model_ref: str, options: LoadOptions) -> Backend:
    import os

    from huggingface_hub import snapshot_download

    transformers = importlib.import_module("transformers")
    peft = importlib.import_module("peft")
    dtype = bf16_dtype(options.device)

    proc = transformers.AutoProcessor.from_pretrained(BASE, revision=BASE_REVISION)
    model = transformers.VoxtralForConditionalGeneration.from_pretrained(
        BASE, revision=BASE_REVISION, dtype=dtype, device_map=device_map(options.device)
    )
    whisper = transformers.WhisperForConditionalGeneration.from_pretrained(
        ENCODER, revision=ENCODER_REVISION, dtype=dtype
    )
    model.audio_tower.load_state_dict(whisper.model.encoder.state_dict())
    del whisper

    # The adapter repo also ships trainer pickles (optimizer.pt, rng_state.pth,
    # training_args.bin); fetch only the adapter itself.
    adapter_dir = model_ref if os.path.isdir(model_ref) else snapshot_download(
        model_ref, allow_patterns=["adapter_config.json", "adapter_model.safetensors"]
    )
    model = peft.PeftModel.from_pretrained(model, adapter_dir)
    n = _check_adapter_loaded(model, adapter_dir)
    print(f"  danstral: {n} LoRA modules applied; merging")
    model = model.merge_and_unload()
    return DanstralBackend(model, proc, BASE, options=options)
