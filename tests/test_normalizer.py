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
    # number_words=False isolates the unicode / separator / case / punctuation layers
    # (these cases assert bare-digit output; the default-on num2words path is covered
    # by test_number_words_* below).
    assert normalise(text, number_words=False) == expected


def test_default_form_is_nfkc():
    # The published default is now NFKC: it folds the fi-ligature to "fi".
    assert normalise("ﬁsk") == normalise("ﬁsk", unicode_form="NFKC")
    assert normalise("ﬁsk") == "fisk"  # NFKC splits the fi-ligature


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
    # number_words=False: these assert the folded *digit* output, before num2words.
    assert normalise(text, unicode_form="NFKC", number_words=False) == expected


def test_nfc_leaves_fraction_opaque():
    # Under the non-default NFC, ¾ survives whole, so ref "¾" and hyp "3/4" would
    # *not* match — the exact mismatch the now-default NFKC fixes (folds ¾ -> "34").
    assert normalise("¾", unicode_form="NFC", number_words=False) == "¾"
    assert normalise("¾", number_words=False) == "34"  # default NFKC folds it


def test_invalid_unicode_form_raises():
    with pytest.raises(ValueError):
        normalise("hej", unicode_form="NFKZ")


@pytest.mark.parametrize("text, expected", [
    ("jeg har 4 æbler", "jeg har fire æbler"),     # bare integer → cardinal
    ("24", "fireogtyve"),                          # vigesimal compound
    ("der var 100 mennesker", "der var ethundrede mennesker"),
    ("året 2025", "året totusinde og femogtyve"),  # year expands
    ("ingen tal her", "ingen tal her"),            # no-op when no digits
    # Symmetry with the spoken form: "4" and "fire" both canonicalise to "fire".
    ("4", "fire"),
])
def test_number_words_expands_standalone_integers(text, expected):
    assert normalise(text, number_words=True) == expected


def test_number_words_on_by_default():
    # number_words is now the published default, so a bare digit expands to words.
    assert normalise("jeg har 4 æbler") == "jeg har fire æbler"
    assert normalise("4") == "fire"


def test_number_words_opt_out_restores_digits():
    # ...and it can be turned off for a digit-preserving back-comparison.
    assert normalise("jeg har 4 æbler", number_words=False) == "jeg har 4 æbler"
    assert normalise("4", number_words=False) == "4"


def test_number_words_leaves_embedded_digits_untouched():
    # A decade like "1960'erne" normalises to the single token "1960erne", which is
    # NOT a standalone integer (^\d+$), so it is left untouched rather than expanded
    # into num2words' multi-word canonical form (measured to hurt WER).
    assert normalise("1960'erne") == "1960erne"


def test_number_words_symmetric_digit_and_word():
    # The whole point: a digit hypothesis and a spelled-out reference collapse to the
    # same string, so the formatting difference stops counting as an error.
    assert normalise("24") == normalise("fireogtyve") == "fireogtyve"


@pytest.mark.parametrize("text, expected", [
    ("øh jeg tror det", "jeg tror det"),          # leading filler
    ("det er øhm svært", "det er svært"),         # mid filler
    ("hmm ja", "ja"),                             # hmm
    ("jamen ehm altså", "jamen altså"),           # ehm
    ("nej det passer ikke", "nej det passer ikke"),  # no-op
])
def test_filler_words_removed_when_enabled(text, expected):
    assert normalise(text, filler_words=True) == expected


def test_filler_words_off_by_default():
    # Opt-in: fillers are kept unless explicitly requested.
    assert normalise("øh jeg tror det") == "øh jeg tror det"


def test_filler_words_leaves_real_words_alone():
    # Word-boundaried: the "hm" in "ohm" must not be stripped.
    assert normalise("modstanden er en ohm", filler_words=True) == "modstanden er en ohm"
