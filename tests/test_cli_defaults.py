"""Published evaluation defaults remain aligned with the normaliser contract."""
import sys

from danish_asr_leaderboard.cli import parse_args


def test_published_normalisation_defaults(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["danish-asr-eval", "--model", "openai/whisper-small", "--backend", "transformers"],
    )

    args = parse_args()

    assert args.unicode_form == "NFKC"
    assert args.number_words is True
    assert args.filler_words is True