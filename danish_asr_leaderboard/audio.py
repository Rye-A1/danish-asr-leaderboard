"""Audio helpers: decode dataset audio to 16 kHz mono WAV and read durations.

Dataset audio is loaded undecoded (raw bytes via ``datasets.Audio(decode=False)``)
and converted to 16 kHz mono WAV with ffmpeg. This avoids the ``torchcodec``
dependency and gives every backend a uniform, seekable input format.

Requires the ``ffmpeg`` binary on PATH.
"""
from __future__ import annotations

import subprocess
import tempfile
import wave
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def audio_to_wav(src: Path, dst: Path) -> bool:
    """Transcode any audio file at ``src`` to 16 kHz mono WAV at ``dst``."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-ar", "16000", "-ac", "1", str(dst)],
            capture_output=True,
            timeout=60,
        )
        return result.returncode == 0 and dst.exists()
    except Exception:
        return False


def audio_bytes_to_wav(audio_field: dict, dst: Path) -> bool:
    """Write raw audio bytes (from ``Audio(decode=False)``) to 16 kHz mono WAV."""
    raw_bytes = audio_field.get("bytes")
    if not raw_bytes:
        return False
    suffix = Path(audio_field.get("path") or "audio.mp3").suffix or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name
    ok = audio_to_wav(Path(tmp_path), dst)
    Path(tmp_path).unlink(missing_ok=True)
    return ok


def wav_duration(path: str) -> float:
    """Duration in seconds of a WAV file (0.0 if unreadable)."""
    try:
        with wave.open(path) as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return 0.0


def load_audio_array(audio_path: str, target_sr: int = 16000) -> "np.ndarray":
    """Load an audio file as a float32 mono numpy array resampled to ``target_sr``.

    Used by backends (e.g. Seamless, Cohere ASR) that consume raw arrays rather
    than file paths.
    """
    import numpy as np

    try:
        import soundfile as sf

        audio, sr = sf.read(audio_path, dtype="float32")
    except Exception:
        import torchaudio

        waveform, sr = torchaudio.load(audio_path)
        audio = waveform.mean(0).numpy().astype("float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio
