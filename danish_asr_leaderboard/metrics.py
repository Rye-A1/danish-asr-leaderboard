"""Corpus-level WER and CER.

Both metrics are aggregated over the whole list of (reference, hypothesis) pairs
(total edits / total reference units), i.e. true corpus WER/CER rather than a mean
of per-utterance rates. This matches the methodology of the HF Open ASR Leaderboard.

Inputs are expected to be already normalised (see ``normalizer.normalise``). Pairs
whose reference is empty after normalisation are dropped before scoring: modern
``jiwer`` raises on empty references, and an empty reference carries no word/char
information to score against. This is robustness only — the dataset loaders already
skip empty raw transcripts, so this affects only degenerate punctuation-only refs.
"""
from __future__ import annotations


def _drop_empty_refs(refs: list[str], hyps: list[str]) -> tuple[list[str], list[str]]:
    out_refs: list[str] = []
    out_hyps: list[str] = []
    for ref, hyp in zip(refs, hyps):
        if ref.strip():
            out_refs.append(ref)
            out_hyps.append(hyp)
    return out_refs, out_hyps


def compute_wer(refs: list[str], hyps: list[str]) -> float:
    """Corpus Word Error Rate as a percentage (lower is better)."""
    refs, hyps = _drop_empty_refs(refs, hyps)
    if not refs:
        return 0.0
    try:
        import jiwer

        return jiwer.wer(refs, hyps) * 100.0
    except ImportError:
        pass
    # Fallback: pure-Python word-level edit distance, aggregated globally.
    total_ref_words = 0
    total_errors = 0
    for ref, hyp in zip(refs, hyps):
        r = ref.split()
        h = hyp.split()
        total_ref_words += len(r)
        dp = list(range(len(h) + 1))
        for i, rw in enumerate(r):
            new_dp = [i + 1] + [0] * len(h)
            for j, hw in enumerate(h):
                new_dp[j + 1] = dp[j] if rw == hw else 1 + min(dp[j], dp[j + 1], new_dp[j])
            dp = new_dp
        total_errors += dp[len(h)]
    if total_ref_words == 0:
        return 0.0
    return (total_errors / total_ref_words) * 100.0


def compute_cer(refs: list[str], hyps: list[str]) -> float:
    """Corpus Character Error Rate as a percentage (lower is better)."""
    refs, hyps = _drop_empty_refs(refs, hyps)
    if not refs:
        return 0.0
    try:
        import jiwer

        return jiwer.cer(refs, hyps) * 100.0
    except ImportError:
        pass
    # Fallback: pure-Python character-level edit distance, aggregated globally.
    total_ref_chars = 0
    total_errors = 0
    for ref, hyp in zip(refs, hyps):
        r = list(ref)
        h = list(hyp)
        total_ref_chars += len(r)
        dp = list(range(len(h) + 1))
        for i, rc in enumerate(r):
            new_dp = [i + 1] + [0] * len(h)
            for j, hc in enumerate(h):
                new_dp[j + 1] = dp[j] if rc == hc else 1 + min(dp[j], dp[j + 1], new_dp[j])
            dp = new_dp
        total_errors += dp[len(h)]
    if total_ref_chars == 0:
        return 0.0
    return (total_errors / total_ref_chars) * 100.0
