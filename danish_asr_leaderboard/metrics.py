"""Corpus-level WER and CER, and the per-utterance counts they are built from.

Both metrics are aggregated over the whole list of (reference, hypothesis) pairs
(total edits / total reference units), i.e. true corpus WER/CER rather than a mean
of per-utterance rates. This matches the methodology of the HF Open ASR Leaderboard.

Inputs are expected to be already normalised (see ``normalizer.normalise``). Pairs
whose reference is empty after normalisation are dropped: an empty reference
carries no word/char information to score against. The dataset loaders already
skip empty raw transcripts, so this affects only degenerate punctuation-only refs.

``per_utterance_counts`` is the single primitive. The published score and the
bootstrap confidence intervals in ``scripts/compute_ci.py`` are both derived from
it, so they cannot drift apart in which pairs they drop or how they measure edits.
A corpus rate is a ratio of sums, so a bootstrap replicate is just
``sum(edits[idx]) / sum(units[idx])`` over resampled indices.
"""
from __future__ import annotations

from collections.abc import Sequence

from rapidfuzz.distance import Levenshtein


def per_utterance_counts(
    refs: list[str], hyps: list[str], *, unit: str = "word"
) -> tuple[list[int], list[int]]:
    """Return aligned ``(edits, reference_units)`` per surviving utterance.

    ``unit`` is ``"word"`` (whitespace tokens) or ``"char"``. Pairs whose
    reference is empty are dropped, so every returned unit count is >= 1 and
    ``sum(edits) / sum(units)`` is the corpus rate.
    """
    if unit not in ("word", "char"):
        raise ValueError(f"unit must be 'word' or 'char', got {unit!r}")
    words = unit == "word"
    edits: list[int] = []
    units: list[int] = []
    for ref, hyp in zip(refs, hyps):
        if not ref.strip():
            continue  # nothing to score against
        r = ref.split() if words else ref
        h = hyp.split() if words else hyp
        edits.append(Levenshtein.distance(r, h))
        units.append(len(r))
    return edits, units


def corpus_rate(edits: Sequence[float], units: Sequence[float]) -> float:
    """Total edits / total reference units, as a percentage."""
    total = sum(units)
    if total == 0:
        return 0.0
    return sum(edits) / total * 100.0


def compute_wer(refs: list[str], hyps: list[str]) -> float:
    """Corpus Word Error Rate as a percentage (lower is better)."""
    return corpus_rate(*per_utterance_counts(refs, hyps, unit="word"))


def compute_cer(refs: list[str], hyps: list[str]) -> float:
    """Corpus Character Error Rate as a percentage (lower is better)."""
    return corpus_rate(*per_utterance_counts(refs, hyps, unit="char"))
