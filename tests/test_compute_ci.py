"""Confidence intervals must use the same normalizer as published scores."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from compute_ci import per_utterance_stats


def test_ci_counts_use_published_number_and_clock_rules():
    word_edits, ref_words, char_edits, ref_chars = per_utterance_stats(
        ["øh 2.4ghz og hundrede procents sikkerhed klokken 06:30 format 3:2"],
        ["2,4 GHz og 100% sikkerhed kl. 6.30 format 3 til 2"],
    )
    assert word_edits.tolist() == [0]
    assert char_edits.tolist() == [0]
    assert ref_words[0] > 0 and ref_chars[0] > 0
