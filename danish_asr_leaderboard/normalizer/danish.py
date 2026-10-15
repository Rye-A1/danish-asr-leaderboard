r"""Danish text normalisation applied to references and hypotheses before scoring.

The same normalisation is applied identically to the reference transcript and the
model hypothesis prior to computing WER/CER, so that scores reflect genuine
recognition errors rather than formatting differences.

Steps (in order):
  1. Unicode normalisation — ``unicode_form`` selects the form. The published
     default is ``NFKC`` (compatibility composition), which folds compatibility
     characters (e.g. ligatures, full-width digits, ²→2) so that visually/semantically
     equivalent forms score identically. On the current Danish test sets it is a
     near-no-op vs ``NFC`` (±0.01pp — these chars are rare in speech transcripts), but
     it is "more correct", future-proofs against such characters in new submissions,
     and matches the de-facto Danish standard (danish-speech-eval). Pass
     ``unicode_form="NFC"`` for the older canonical-only behaviour.
  2. Danish number canonicalisation — thousands separators are removed, while
     decimal values retain a ``komma`` marker: ``1.234`` -> ``1234`` and
     ``3.14`` / ``3,14`` -> ``3komma14``. Trailing decimal zeroes are dropped,
     so ``5.0`` and ``5.00`` match. Frequency units are separated from attached
     numerals and compound suffixes (``2.4ghz`` -> ``2komma4 ghz``). A percent
     sign after a numeral and the genitive ``procents`` both become ``procent``.
  3. Lowercase
  4. Punctuation/symbol removal. Apostrophes *inside* a word (e.g. ``det's``) are
     preserved; all other punctuation and symbols are removed.
  5. Whitespace collapse.

Step 6 (``number_words``, **ON by default** — the published methodology) expands
every *standalone integer token* to its Danish cardinal words via ``num2words``
(``4`` -> ``fire``, ``24`` -> ``fireogtyve``), applied identically to reference and
hypothesis. This folds the digit<->word formatting difference (e.g. ``"4"`` vs
``"fire"``) that would otherwise count as a recognition error even though both are
valid. Measured offline over the saved raw outputs it lowers WER by ~0.2pp on
essentially every model; it is *almost* uniform but materially larger for
digit-emitting models (e.g. ElevenLabs scribe_v2 gains ~0.4pp and moves up a rank),
which is the point — those models were being penalised for formatting, not
recognition. This matches the de-facto Danish standard (Dansk-Data-Science-Community
``danish-speech-eval``, whose headline "normalised WER" is numerals -> words). Pass
``number_words=False`` to recover the older digit-preserving behaviour for a
back-comparison.

Only *standalone* integer tokens (``^\d+$`` after the punctuation strip) are
converted — digits embedded in larger tokens (decades ``1960'erne``, mixed
alphanumerics) are deliberately left untouched, because expanding them produces
num2words' canonical multi-word form which rarely matches how such spans were
actually spoken, and was measured to *hurt* WER.

Dash-separated ranges are the exception: a dash between two digits becomes a
space in the punctuation strip, so ``1-3`` scores as two numbers rather than
being glued into ``13`` -> ``tretten``. Deleting the dash produced references no
correct transcription could match (the time range ``10 00-11 00`` became ``ti
elleve nul``) and handed free matches to digit-emitting models. Colons and
slashes are *not* yet handled the same way — ``10:00`` still glues to
``ettusind`` and ``4/5`` to ``femogfyrre``; those need their own semantics
(a slash may be a fraction, a date reads differently) and are tracked separately.
Ordinals (``3.`` -> ``tredje``) were tested and rejected: most ``N.`` are
sentence-final cardinals, not true ordinals. Other units remain literal.

Step 7 (``filler_words``, **ON by default**) removes Danish hesitation fillers
(``øh``, ``hmm`` …) from reference and hypothesis alike, using the exact pattern
from ``alexandrainst/coral``. Verbatim scoring measured transcription convention
rather than accuracy: a model trained to omit disfluencies was penalised for a
style choice.

Measured over 34 models / 910k utterances: mean WER −0.09pp, no rank changes on
mean WER or Avg. Rank; only ``coral_conversation`` reorders (5 models). Not
monotone — ``whisper-tiny`` +0.52 and ``whisper-base`` +0.41, both already above
96 WER, where a shrinking reference raises the rate.

Limits: ``ah`` and ``mm`` survive (CoRal keeps them too, and a looser regex eats
``arm``); 374 filler-only utterances become empty references and are dropped by
``per_utterance_counts``.

Because the normaliser is parameterised, ``scripts/rescore.py`` can re-derive
WER/CER from those saved raw outputs under any configuration, so changing the
normalisation strategy never requires re-running inference.
"""
from __future__ import annotations

import re
import unicodedata

_NUMBER_TOKEN_RE = re.compile(r"\d[\d.,]*")
_FREQUENCY_UNITS = {
    "gigahertz": "ghz", "megahertz": "mhz", "kilohertz": "khz",
    "hertz": "hz", "ghz": "ghz", "mhz": "mhz", "khz": "khz", "hz": "hz",
}
_FREQUENCY_PATTERN = "|".join(_FREQUENCY_UNITS)
_FREQUENCY_RE = re.compile(
    rf"(?<!\w)(\d[\d.,]*)\s*({_FREQUENCY_PATTERN})\b", re.IGNORECASE
)
_FREQUENCY_NAME_RE = re.compile(
    r"\b(gigahertz|megahertz|kilohertz|hertz)\b", re.IGNORECASE
)
_FREQUENCY_COMPOUND_RE = re.compile(
    r"\b(ghz|mhz|khz|hz)[\-\u2010-\u2014](?=[^\W\d_])", re.IGNORECASE
)
_THOUSANDS_RE = re.compile(r"^\d{1,3}(?:[.,]\d{3})+$")
_GROUPED_DECIMAL_RE = re.compile(r"^(\d{1,3}(?:[.,]\d{3})+)[.,](\d{1,2})$")
_DECIMAL_RE = re.compile(r"^(\d+)[.,](\d{1,2})$")
_PERCENT_SYMBOL_RE = re.compile(r"(?<=\d)\s*%")
_PERCENT_GENITIVE_RE = re.compile(r"\bprocents\b", re.IGNORECASE)

# Unicode forms accepted by ``normalise``. NFKC is the published default; NFC is the
# canonical-only variant. NFD/NFKD are decomposed forms, accepted for completeness
# but not used by the leaderboard.
_VALID_UNICODE_FORMS = {"NFC", "NFKC", "NFD", "NFKD"}

# A standalone integer token (after punctuation stripping). Only these are expanded
# to words when ``number_words=True`` — see module docstring for why embedded digits
# are deliberately excluded.
_STANDALONE_INT_RE = re.compile(r"^\d+$")

# Danish hesitation fillers, removed when ``filler_words=True``. Same pattern as
# danish-speech-eval; text is already lower-cased at this point. Word-boundaried so
# it never bites into real words (e.g. the "hm" in "ohm" is left alone).
_FILLER_RE = re.compile(r"\b(?:eh+m*|øh+m*|h+m+|m+h+)\b")

def _spoken_to_digits_da(text: str) -> str:
    """Fold spelled-out Danish numerals to digits, ahead of ``number_words``.

    ``number_words`` can only act on tokens that are *already* digits, so it
    canonicalises one side of a digit/word pair and leaves the other as the model
    wrote it. Where the model's spelling differs from num2words' canonical form the
    pair still mismatches — and because Danish compounds are written closed, a
    spaced ``otte og tredive`` against ``otteogtredive`` costs three word errors
    rather than one. Folding words to digits first is many-to-one, so every spelling
    and spacing variant collapses to the same value before it is expanded again.
    Measured over the saved raw outputs this is near-monotone: 31 of 32 models
    improve or hold, most by ~0.1pp, and models that spell numerals out with spaces
    by up to 1.1pp.

    ``text2num`` is conservative about ambiguity — bare ``en``/``et`` stay articles
    rather than becoming ``1`` — which is what makes the pre-pass safe to apply to
    running text. A missing ``text2num`` is raised rather than skipped, for the same
    reason as ``num2words``: silently dropping it would score under the old
    methodology while still reporting the new one.
    """
    try:
        from text_to_num import alpha2digit
    except ImportError as exc:  # pragma: no cover - environment error
        raise RuntimeError(
            "text2num is required for spoken_numbers=True (the published "
            "methodology) but is not installed. Install it (`pip install "
            "text2num`) or pass spoken_numbers=False to score without the "
            "spelled-out-numeral fold."
        ) from exc
    try:
        folded = alpha2digit(text, "da")
    except Exception:
        return text  # never let a conversion failure abort scoring
    # Re-apply the numeral canonicalisation from step 2: alpha2digit can emit a
    # separator of its own ("tre komma fjorten" -> "3,14"), and that arrives after
    # step 2 has already run, so without this a *spoken* decimal would no longer
    # fold onto the *written* one it is supposed to match.
    return normalize_numbers_da(folded)


# Bounded cache so repeated numerals (years, counts) don't re-invoke num2words.
_cardinal_cache: dict[str, str] = {}


def _cardinal_words_da(token: str) -> str:
    """Render a standalone integer ``token`` as Danish cardinal words.

    A *conversion* failure (e.g. an out-of-range value num2words can't render)
    falls back to the original token, so the transform never raises mid-scoring.
    A *missing* ``num2words``, by contrast, is raised: it is a declared core
    dependency, and silently skipping the expansion would score the whole run
    under the old digit-preserving methodology while still reporting
    ``number_words=True`` — a difference of up to ~0.6pp WER that is invisible in
    the output.
    """
    cached = _cardinal_cache.get(token)
    if cached is not None:
        return cached
    try:
        from num2words import num2words
    except ImportError as exc:  # pragma: no cover - environment error
        raise RuntimeError(
            "num2words is required for number_words=True (the published "
            "methodology) but is not installed. Install it (`pip install "
            "num2words`) or pass number_words=False to score without "
            "numeral expansion."
        ) from exc
    try:
        words = num2words(int(token), lang="da")
    except Exception:
        words = token
    _cardinal_cache[token] = words
    return words


def _canonical_number_da(value: str, *, embedded: bool = False) -> str:
    """Keep decimals distinct from integers without changing mixed identifiers."""
    if embedded:
        return value.replace(".", "").replace(",", "")
    if _THOUSANDS_RE.fullmatch(value):
        return value.replace(".", "").replace(",", "")
    grouped = _GROUPED_DECIMAL_RE.fullmatch(value)
    decimal = grouped or _DECIMAL_RE.fullmatch(value)
    if decimal:
        whole = decimal.group(1).replace(".", "").replace(",", "")
        fraction = decimal.group(2).rstrip("0")
        return str(int(whole)) + (f"komma{fraction}" if fraction else "")
    return value.replace(".", "").replace(",", "")


def normalize_numbers_da(text: str) -> str:
    """Canonicalise Danish thousands, decimals and attached frequency units.

    A separator followed by three digits is treated as a thousands separator;
    one or two digits is a decimal. Bare ``802.11n`` stays an identifier rather
    than a decimal because its numeric span is embedded in a non-unit word.
    """
    text = _FREQUENCY_RE.sub(
        lambda m: f"{_canonical_number_da(m.group(1))} "
                  f"{_FREQUENCY_UNITS[m.group(2).lower()]}", text
    )
    text = _FREQUENCY_NAME_RE.sub(
        lambda m: _FREQUENCY_UNITS[m.group(1).lower()], text
    )
    text = _FREQUENCY_COMPOUND_RE.sub(r"\1 ", text)
    return _NUMBER_TOKEN_RE.sub(
        lambda m: _canonical_number_da(
            m.group(0), embedded=m.end() < len(text) and text[m.end()].isalpha()
        ), text
    )


_DASHES = ("-", "\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212")


def normalise(
    text: str,
    *,
    unicode_form: str = "NFKC",
    number_words: bool = True,
    spoken_numbers: bool = True,
    filler_words: bool = True,
) -> str:
    """Unicode-normalise -> number canonicalisation -> lowercase -> punctuation strip -> collapse.

    ``unicode_form`` selects the Unicode normalisation form (default ``NFKC``, the
    compatibility-folding form; pass ``NFC`` for canonical-only). The same value must
    be used for both references and hypotheses for scores to stay comparable.

    ``number_words`` (default ``True`` — the published methodology) expands every
    standalone integer token to its Danish cardinal words via ``num2words``
    (``4`` -> ``fire``). Pass ``False`` to recover the digit-preserving behaviour.

    ``spoken_numbers`` (default ``True`` — the published methodology) folds
    spelled-out numerals to digits *before* ``number_words`` expands them back, so
    that spelling and spacing variants (``otte og tredive``, ``hundrede``) collapse
    onto the same form as the digits they correspond to. Pass ``False`` to recover
    the num2words-only behaviour. Has no effect when ``number_words=False`` is used
    to keep digits.

    ``filler_words`` (default ``True``) removes Danish hesitation fillers
    (``øh``, ``hmm`` …). See the module docstring for rationale and measured impact.
    """
    if unicode_form not in _VALID_UNICODE_FORMS:
        raise ValueError(
            f"unicode_form must be one of {sorted(_VALID_UNICODE_FORMS)}, got {unicode_form!r}"
        )
    text = unicodedata.normalize(unicode_form, text)
    text = _PERCENT_SYMBOL_RE.sub(" procent", text)
    text = _PERCENT_GENITIVE_RE.sub("procent", text)
    text = normalize_numbers_da(text)
    text = text.lower()
    # Remove punctuation/symbols, keeping apostrophes between two word characters.
    # A dash between two digits becomes a space rather than vanishing: deleting
    # it glued ranges into a single number, so "1-3" scored as "tretten" and the
    # time range "10 00-11 00" as "ti elleve nul" -- references no correct
    # transcription could match, which also handed free matches to models that
    # emit digits. A space keeps the two numbers separate through the expansion.
    result: list[str] = []
    for i, ch in enumerate(text):
        cat = unicodedata.category(ch)
        if cat.startswith("P") or cat.startswith("S"):
            if ch in ("'", "’") and 0 < i < len(text) - 1:
                if text[i - 1].isalpha() and text[i + 1].isalpha():
                    result.append("'")
            elif ch in _DASHES and 0 < i < len(text) - 1:
                if text[i - 1].isdigit() and text[i + 1].isdigit():
                    result.append(" ")
            # otherwise: drop the punctuation/symbol
        else:
            result.append(ch)
    text = "".join(result)
    if spoken_numbers:
        text = _spoken_to_digits_da(text)
    if number_words:
        text = " ".join(
            _cardinal_words_da(tok) if _STANDALONE_INT_RE.match(tok) else tok
            for tok in text.split()
        )
    if filler_words:
        text = _FILLER_RE.sub(" ", text)
    return " ".join(text.split())


# American-spelling alias for ergonomics.
normalize = normalise
