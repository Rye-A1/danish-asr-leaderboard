"""Danish text normalisation applied to references and hypotheses before scoring.

The same normalisation is applied identically to the reference transcript and the
model hypothesis prior to computing WER/CER, so that scores reflect genuine
recognition errors rather than formatting differences.

Steps (in order):
  1. Unicode normalisation — ``unicode_form`` selects the form. The published
     default is ``NFC`` (canonical composition). ``NFKC`` additionally folds
     compatibility characters (e.g. ligatures, full-width digits, ²→2); it is
     offered as a selectable variant for offline re-scoring experiments via
     ``scripts/rescore.py`` and is not yet the default.
  2. Danish number canonicalisation — separators *within* a numeral are removed so
     that punctuation-only formatting differences don't inflate WER. Thousands and
     decimals are both stripped: ``1.234`` and ``1,234`` -> ``1234``; ``3.14`` and
     ``3,14`` -> ``314``. The decimal separator is not preserved, so ``3,14``
     collides with the integer ``314`` — a negligible, symmetric edge case (the
     same transform is applied to reference and hypothesis).
  3. Lowercase
  4. Punctuation/symbol removal. Apostrophes *inside* a word (e.g. ``det's``) are
     preserved; all other punctuation and symbols are removed.
  5. Whitespace collapse.

NOTE: digit<->word equivalence (e.g. ``"4"`` vs ``"fire"``) is intentionally NOT
normalised. Both are valid transcriptions, but a model that consistently emits one
form when the reference uses the other will incur errors. This is a known
limitation shared by most public ASR leaderboards. A symmetric word<->digit
converter (e.g. an allo-media text2num port) is the planned fix; because raw model
outputs are now persisted at eval time, that can be added and validated offline
without re-running any model.

Because the normaliser is parameterised, ``scripts/rescore.py`` can re-derive
WER/CER from those saved raw outputs under any configuration, so changing the
normalisation strategy never requires re-running inference.
"""
from __future__ import annotations

import re
import unicodedata

_NUMBER_TOKEN_RE = re.compile(r"\d[\d.,]*")

# Unicode forms accepted by ``normalise``. NFC is the published default; NFKC is a
# selectable variant for re-scoring experiments. NFD/NFKD are decomposed forms,
# accepted for completeness but not used by the leaderboard.
_VALID_UNICODE_FORMS = {"NFC", "NFKC", "NFD", "NFKD"}


def normalize_numbers_da(text: str) -> str:
    """Remove ``.`` / ``,`` separators within a numeral (``1.234`` -> ``1234``,
    ``3,14`` -> ``314``).

    Both thousand and decimal separators are stripped, so a numeral scores
    identically regardless of formatting. (The downstream punctuation strip would
    remove these characters anyway; this step makes the numeral handling explicit
    and independent of that implementation.)
    """
    return _NUMBER_TOKEN_RE.sub(
        lambda m: m.group(0).replace(".", "").replace(",", ""), text
    )


def normalise(text: str, *, unicode_form: str = "NFC") -> str:
    """Unicode-normalise -> number canonicalisation -> lowercase -> punctuation strip -> collapse.

    ``unicode_form`` selects the Unicode normalisation form (default ``NFC``;
    ``NFKC`` is the compatibility-folding variant). The same value must be used for
    both references and hypotheses for scores to stay comparable.
    """
    if unicode_form not in _VALID_UNICODE_FORMS:
        raise ValueError(
            f"unicode_form must be one of {sorted(_VALID_UNICODE_FORMS)}, got {unicode_form!r}"
        )
    text = unicodedata.normalize(unicode_form, text)
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
