"""Unit tests for update_space.py helper functions (no network required)."""
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from update_space import (
    OFFICIAL_SIZE,
    PROVIDER_DOCS,
    PROVIDER_HF_ORG,
    PROVIDER_LOGO,
    THUMBNAIL_SIZE,
    _api_docs_url,
    _fmt_size,
    _official_size,
    _parse_model,
    _size_from_name,
    generate_cover_image,
)


# ---------------------------------------------------------------------------
# _fmt_size
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("val, expected", [
    (2.0,   "2.0"),
    (1.7,   "1.7"),
    (24.0,  "24.0"),
    (0.315, "0.3"),     # 0.315 >= 0.1 → 1 dp
    (0.07,  "0.07"),    # sub-0.1 → 2 dp
    (0,     "—"),
    (None,  "—"),
    (float("nan"), "—"),
])
def test_fmt_size(val, expected):
    assert _fmt_size(val) == expected


# ---------------------------------------------------------------------------
# _size_from_name
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name, expected", [
    ("mistralai/Voxtral-Small-24B-2507",    24.0),
    ("mistralai/Voxtral-Mini-3B-2507",       3.0),
    ("Qwen/Qwen3-ASR-1.7B",                  1.7),
    ("nvidia/parakeet-tdt-0.6b-v3",          0.6),
    ("CoRal-project/roest-v3-wav2vec2-315m", 0.315),
    ("CoRal-project/roest-v2-wav2vec2-2B",   2.0),
    ("openai/whisper-large-v3",             None),   # no size in name
    ("syvai/hviske-v5",                     None),   # no size in name
    ("scribe_v2",                           None),   # API model, no size
])
def test_size_from_name(name, expected):
    assert _size_from_name(name) == expected


# ---------------------------------------------------------------------------
# _official_size — 3-tier precedence
# ---------------------------------------------------------------------------
def test_official_size_manual_override():
    # Manual OFFICIAL_SIZE takes highest priority
    assert _official_size("syvai/hviske-v5", 999.0) == "2.0"
    assert _official_size("openai/whisper-large-v3", 999.0) == "2.0"


def test_official_size_from_name():
    # Size in model name beats safetensors count
    result = _official_size("mistralai/Voxtral-Small-24B-2507", 24.3)
    assert result == "24.0"


def test_official_size_fallback_to_params():
    # No name match and not in OFFICIAL_SIZE → use params_b
    result = _official_size("some/unknown-model", 8.0)
    assert result == "8.0"


def test_official_size_api_model():
    assert _official_size("scribe_v2", None) == "—"
    assert _official_size("gpt-4o-transcribe-benchmark", None) == "—"


# ---------------------------------------------------------------------------
# _api_docs_url
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name, key_fragment", [
    ("scribe_v2",                      "elevenlabs.io"),
    ("gpt-4o-transcribe-benchmark",    "platform.openai.com"),
    ("gpt-4o-mini-transcribe-benchmark", "platform.openai.com"),
    ("chirp_3",                        "cloud.google.com"),
    ("soniox-v1",                      "soniox.com"),
    ("azure-da-speech",                "microsoft.com"),
    # "transcribe" alone should NOT match the scribe_v key
    ("some-transcribe-model",          ""),
])
def test_api_docs_url(name, key_fragment):
    url = _api_docs_url(name)
    assert key_fragment in url


# ---------------------------------------------------------------------------
# _parse_model
# ---------------------------------------------------------------------------
def test_parse_model_markdown_link():
    name, url = _parse_model("[openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)")
    assert name == "openai/whisper-large-v3"
    assert url == "https://huggingface.co/openai/whisper-large-v3"


def test_parse_model_plain():
    name, url = _parse_model("scribe_v2")
    assert name == "scribe_v2"
    assert url == ""


def test_parse_model_non_string():
    name, url = _parse_model(None)
    assert url == ""


def test_generate_cover_image(tmp_path):
    out = tmp_path / "cover.jpeg"
    data = {
        "updated": "2026-06-23",
        "wer": [
            {"name": "alpha/model-a", "mean_wer": 7.12, "mean_cer": 2.91, "speed_x": 18.4},
            {"name": "beta/model-b", "mean_wer": 7.48, "mean_cer": 3.05, "speed_x": 12.1},
            {"name": "gamma/model-c", "mean_wer": 8.04, "mean_cer": 3.44, "speed_x": 9.7},
        ],
        "cer": [],
    }

    result = generate_cover_image(data, out)

    assert result == out
    assert out.exists()
    image = Image.open(out)
    assert image.size == THUMBNAIL_SIZE
