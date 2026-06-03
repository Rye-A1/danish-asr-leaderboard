"""Backend registry.

Importing this package imports every backend module so that each registers its
loader. Heavy third-party imports stay inside the loaders, so this stays cheap.
"""
from danish_asr_leaderboard.backends.base import (
    Backend,
    LoadOptions,
    available_backends,
    load_backend,
    register,
)

# Import for side effect: each module calls register() at import time.
from danish_asr_leaderboard.backends import (  # noqa: F401,E402
    cohere_backend,
    faster_whisper_backend,
    nemo_backend,
    qwen_asr_backend,
    seamless_backend,
    transformers_backend,
    vibevoice_backend,
    voxtral_backend,
    wav2vec2_backend,
)
from danish_asr_leaderboard.backends.api import (  # noqa: F401,E402
    azure_openai,
    elevenlabs,
    google_chirp,
    soniox,
)

__all__ = [
    "Backend",
    "LoadOptions",
    "load_backend",
    "register",
    "available_backends",
]
