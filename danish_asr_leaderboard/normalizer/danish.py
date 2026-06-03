"""Danish text normalisation applied to references and hypotheses before scoring.

The same normalisation is applied identically to the reference transcript and the
model hypothesis prior to computing WER/CER, so that scores reflect genuine
recognition errors rather than formatting differences.

Steps (in order):
  1. Unicode NFC
  2. Danish number canonicalisation — thousand separators stripped and decimal
     separators collapsed so that ``1.234`` and ``1,234`` (thousands) both become
     ``1234`` and ``3.14`` / ``3,14`` (decimals) both reduce to the same token.
     This prevents punctuation-only formatting differences in numerals from
     inflating WER.
  3. Lowercase
  4. Punctuation/symbol removal. Apostrophes *inside* a word (e.g. ``det's``) are
     preserved; all other punctuation and symbols are removed.
  5. Whitespace collapse.

NOTE: digit<->word equivalence (e.g. ``"4"`` vs ``"fire"``) is intentionally NOT
normalised. Both are valid transcriptions, but a model that consistently emits one
form when the reference uses the other will incur errors. This is a known
limitation shared by most public ASR leaderboards.
"""
from __future__ import annotations

import re
import unicodedata

_NUMBER_TOKEN_RE = re.compile(r"\d[\d.,]*")


def normalize_numbers_da(text: str) -> str:
    """Strip thousand separators and collapse decimal separators in numerals."""

    def _fmt(m: re.Match) -> str:
        token = m.group(0)
        # Thousand-separated integer: 1.234.567 or 1,234,567
        if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", token):
            return token.replace(".", "").replace(",", "")
        # Decimal number with a single separator: 3,14 or 3.14
        sep_count = token.count(".") + token.count(",")
        if sep_count == 1:
            left, _, right = (
                token.partition(".") if "." in token else token.partition(",")
            )
            if left.isdigit() and right.isdigit():
                return f"{left},{right}"
        # Anything else: strip separators
        return token.replace(".", "").replace(",", "")

    return _NUMBER_TOKEN_RE.sub(_fmt, text)


def normalise(text: str) -> str:
    """NFC -> number canonicalisation -> lowercase -> punctuation strip -> collapse."""
    text = unicodedata.normalize("NFC", text)
    text = normalize_numbers_da(text)
    text = text.lower()
    # Remove punctuation/symbols, keeping apostrophes between two word characters.
    result: list[str] = []
    for i, ch in enumerate(text):
        cat = unicodedata.category(ch)
        if cat.startswith("P") or cat.startswith("S"):
            if ch in ("'", "’") and 0 < i < len(text) - 1:
                if text[i - 1].isalpha() and text[i + 1].isalpha():
                    result.append("'")
            # otherwise: drop the punctuation/symbol
        else:
            result.append(ch)
    text = "".join(result)
    return " ".join(text.split())


# American-spelling alias for ergonomics.
normalize = normalise
