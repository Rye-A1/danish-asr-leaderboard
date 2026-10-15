"""Confidence intervals must use the same normalizer as published scores."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from compute_ci import per_utterance_stats


def test_ci_counts_ignore_fillers_and_match_decimal_units():
    word_edits, ref_words, char_edits, ref_chars = per_utterance_stats(
        ["øh 2.4ghz og hundrede procents sikkerhed"],
        ["2,4 GHz og 100% sikkerhed"],
    )
    assert word_edits.tolist() == [0]
    assert char_edits.tolist() == [0]
    assert ref_words[0] > 0 and ref_chars[0] > 0
