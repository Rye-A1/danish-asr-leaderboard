"""Unit tests for results helpers."""
from danish_asr_leaderboard.results import model_link, slugify


def test_model_link_plain():
    assert model_link("openai/whisper-large-v3") == \
        "[openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)"


def test_model_link_decoding_variant_url_strips_suffix():
    # Display name keeps "+kenlm"; URL points at the base repo.
    assert model_link("RyeAI/krumme-v1+kenlm") == \
        "[RyeAI/krumme-v1+kenlm](https://huggingface.co/RyeAI/krumme-v1)"


def test_model_link_local_path_untouched():
    assert model_link("/data/models/best.nemo") == "/data/models/best.nemo"


def test_slug_distinguishes_kenlm_variant():
    # The two krumme rows must not collide on disk.
    assert slugify("RyeAI/krumme-v1") != slugify("RyeAI/krumme-v1+kenlm")
