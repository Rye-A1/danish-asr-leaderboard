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
