"""Unit tests for the Danish text normaliser."""
import unicodedata

import pytest
from danish_asr_leaderboard.normalizer.danish import normalise


@pytest.mark.parametrize("text, expected", [
    # NFC unification: a + combining ring above (U+030A) → å
    ("ånd", "ånd"),

    # Number separator stripping
    ("1.234 kroner", "1234 kroner"),          # thousands dot
    ("1,234 kroner", "1234 kroner"),          # thousands comma
    ("3,14", "314"),                          # decimal comma stripped
    ("3.14", "314"),                          # decimal dot stripped
    ("12.345,67", "1234567"),                 # combined
    ("år 2025", "år 2025"),                   # non-numeric unchanged

    # Lowercase
    ("HUND", "hund"),
    ("Æble Øl Ånd", "æble øl ånd"),

    # Punctuation strip
    ("hej, verden!", "hej verden"),
    ("hej... verden", "hej verden"),
    ("hej - verden", "hej verden"),

    # Apostrophe inside word preserved
    ("det's", "det's"),
    ("o'clock", "o'clock"),
    # Apostrophe at start/end removed
    ("'hej'", "hej"),

    # Whitespace collapse
    ("  hej   verden  ", "hej verden"),
    ("hej\tverden\ngodt", "hej verden godt"),

    # Empty / whitespace-only
    ("", ""),
    ("   ", ""),

    # Curly apostrophe inside word preserved
    ("det’s godt", "det's godt"),
])
def test_normalise(text, expected):
    assert normalise(text) == expected


def test_default_form_is_nfc():
    # The published default must stay NFC: a compatibility char is left intact.
    assert normalise("ﬁsk") == normalise("ﬁsk", unicode_form="NFC")
    assert normalise("ﬁsk") == "ﬁsk"  # NFC does not split the fi-ligature


@pytest.mark.parametrize("text, expected", [
    ("ﬁsk", "fisk"),              # fi-ligature (U+FB01) → "fi"
    ("ｈｅｊ", "hej"),               # full-width latin → ascii
    ("ｈｅｊ ２０２５", "hej 2025"),    # full-width digits folded then kept
    ("m²", "m2"),                 # superscript two → "2"
    # Vulgar fractions: NFC leaves ¾ as an opaque blob (compatibility decomposition
    # only, never recomposed); NFKC folds it to digits + fraction slash, which our
    # punctuation strip then reduces to "34" — symmetric on ref and hyp.
    ("¾", "34"),
    ("2½", "212"),
    ("Ⅻ", "xii"),                 # roman numeral twelve → "xii"
])
def test_nfkc_folds_compatibility(text, expected):
    assert normalise(text, unicode_form="NFKC") == expected


def test_nfc_leaves_fraction_opaque():
    # The flip side of the case above: under the default NFC, ¾ survives whole,
    # so ref "¾" and hyp "3/4" would *not* match — the exact mismatch NFKC fixes.
    assert normalise("¾") == "¾"


def test_invalid_unicode_form_raises():
    with pytest.raises(ValueError):
        normalise("hej", unicode_form="NFKZ")
