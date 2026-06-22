"""Unit tests for NeMo backend helpers (no torch/nemo import needed)."""
import pytest

from danish_asr_leaderboard.backends.nemo_backend import _is_lm_nemo


@pytest.mark.parametrize("filename, is_lm", [
    # KenLM / n-gram language models ride along in some repos → must be excluded.
    ("nemo_kenlm_6gram_light_100pct.nemo", True),
    ("RyeAI/krumme-v1/nemo_kenlm_6gram.nemo", True),
    ("ngram_lm.nemo", True),
    ("model_lm.nemo", True),
    # Actual acoustic models → kept.
    ("canary-1b-v2-da-pnc-v2.nemo", False),
    ("parakeet-tdt-da-v3-1.nemo", False),
    ("parakeet-rnnt-110m-da-dk.nemo", False),
    ("nemotron-3.5-asr-streaming-0.6b.nemo", False),
])
def test_is_lm_nemo(filename, is_lm):
    assert _is_lm_nemo(filename) is is_lm
