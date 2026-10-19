"""Unit tests for the Danish text normaliser."""

import pytest
from danish_asr_leaderboard.normalizer.danish import normalise


@pytest.mark.parametrize("text, expected", [
    # NFC unification: a + combining ring above (U+030A) → å
    ("ånd", "ånd"),

    # Thousands separators are stripped; decimal value is preserved.
    ("1.234 kroner", "1234 kroner"),          # thousands dot
    ("1,234 kroner", "1234 kroner"),          # thousands comma
    ("3,14", "3komma14"),                     # decimal comma
    ("3.14", "3komma14"),                     # decimal dot
    ("12.345,67", "12345komma67"),           # grouped decimal
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


def test_filler_words_on_by_default():
    """Fillers are stripped unless verbatim scoring is asked for.

    Scoring them verbatim measured transcription convention rather than
    accuracy: a model trained to omit disfluencies was penalised for a style
    choice. The pattern matches alexandrainst/coral exactly.
    """
    assert normalise("øh jeg tror det") == "jeg tror det"
    assert normalise("øh jeg tror det", filler_words=False) == "øh jeg tror det"


def test_filler_strip_is_symmetric():
    """Applied to reference and hypothesis alike, so neither emitting nor
    omitting a filler is rewarded."""
    ref, hyp = "øh jeg tror det", "jeg tror det"
    assert normalise(ref) == normalise(hyp)


def test_filler_only_utterance_empties():
    """374 of 910k references are filler-only. Stripping empties them, and
    per_utterance_counts drops empty-reference pairs -- documented here so the
    behaviour is deliberate rather than incidental."""
    assert normalise("øh") == ""
    assert normalise("øh øhm hmm") == ""


@pytest.mark.parametrize("text", ["armen gør ondt", "modstanden er en ohm", "mit hjem"])
def test_filler_pattern_does_not_eat_real_words(text):
    """A looser pattern would: 'arm' appears 238 times in the benchmark corpus
    and a regex like a[rh]+m* swallows it."""
    assert normalise(text) == text


def test_filler_words_leaves_real_words_alone():
    # Word-boundaried: the "hm" in "ohm" must not be stripped.
    assert normalise("modstanden er en ohm", filler_words=True) == "modstanden er en ohm"


def test_missing_num2words_raises_not_silently_skipped(monkeypatch):
    """A missing num2words must fail loudly, never silently skip expansion.

    Regression test: the import error used to be swallowed by a bare
    ``except Exception``, so a venv without num2words scored the whole run under
    the old digit-preserving methodology while still reporting
    ``number_words=True`` — a silent ~0.2-0.6pp WER difference.
    """
    import builtins

    from danish_asr_leaderboard.normalizer import danish

    monkeypatch.setattr(danish, "_cardinal_cache", {})
    real_import = builtins.__import__

    def _no_num2words(name, *args, **kwargs):
        if name == "num2words":
            raise ImportError("No module named 'num2words'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_num2words)
    with pytest.raises(RuntimeError, match="num2words is required"):
        normalise("4 biler")


def test_number_words_actually_expands():
    """Guards the happy path: num2words present and wired up."""
    assert normalise("4 biler") == "fire biler"
    assert normalise("24 timer") == "fireogtyve timer"


# ── spoken_numbers: fold spelled-out numerals before num2words expands them ──

@pytest.mark.parametrize("digits, spoken", [
    ("38", "otte og tredive"),          # spaced compound — 3 tokens vs 1 without the fold
    ("38", "otteogtredive"),            # closed compound
    ("100", "hundrede"),                # num2words renders 100 as "ethundrede"
    ("24", "fireogtyve"),
    ("25", "fem og tyve"),
])
def test_spoken_numbers_folds_variants_onto_digits(digits, spoken):
    assert normalise(digits) == normalise(spoken)


def test_spoken_numbers_off_recovers_num2words_only():
    """Without the pre-pass a spaced compound no longer matches its digit form."""
    assert normalise("38", spoken_numbers=False) != normalise("otte og tredive", spoken_numbers=False)


@pytest.mark.parametrize("text", [
    "der var en mand",        # 'en' is the indefinite article, not the numeral
    "et hus i byen",
    "hun købte en kop kaffe",
])
def test_spoken_numbers_leaves_articles_alone(text):
    """Danish en/et are homographs of 'one'; folding them would rewrite articles."""
    assert normalise(text) == text


def test_spoken_numbers_leaves_ordinary_text_untouched():
    s = "det er en helt almindelig sætning uden tal"
    assert normalise(s) == s


def test_output_form_is_still_words_not_digits():
    """The pre-pass feeds num2words; the scored text stays in word form."""
    assert normalise("38") == "otteogtredive"
    assert "38" not in normalise("38")


def test_spoken_decimal_folds_onto_written_decimal():
    """alpha2digit emits a separator after step 2, so canonicalise it again."""
    assert normalise("tre komma fjorten") == normalise("3,14") == normalise("3.14")


def test_decimals_do_not_collide_with_integers_or_trailing_zeroes():
    assert normalise("3,14") != normalise("314")
    assert normalise("2,40") == normalise("2.4")
    assert normalise("5.00") == normalise("5,0") == normalise("5")


def test_frequency_units_in_issue_114_match_across_spacing_and_compounds():
    reference = "standarden 802.11n opererer på både 2.4ghz- og 5.0ghz-frekvenserne"
    brage = "Standarden 802.11n opererer på både 2,4 GHz og 5,0 GHz frekvenserne."
    edda = "Standarden 802.11n opererer på både 2,4ghz og 5,00ghz frekvenserne."
    assert normalise(reference) == normalise(brage) == normalise(edda)
    assert normalise("802.11n") == "80211n"  # model identifier, not a decimal


def test_frequency_names_match_abbreviations():
    assert normalise("2,4 gigahertz og 5 GHz") == normalise("2.4GHz og 5,0ghz")


def test_three_place_decimal_needs_ghz_context():
    assert normalise("2.400 GHz") == normalise("2,4ghz")
    assert normalise("2.400") == normalise("2400")
    assert normalise("2.400 kroner") != normalise("2,4 kroner")
    assert normalise("2.400 MHz") != normalise("2,4 MHz")


def test_clock_times_match_colon_dot_and_zero_minute_forms():
    assert normalise("klokken 06:30") == normalise("kl. 6.30")
    assert normalise("klokken 06:30 og 07:30") == normalise("kl. 6.30 og 7.30")
    assert normalise("klokken 11:35 pm") == normalise("kl. 11.35 p.m.")
    assert normalise("klokken 12:00") == normalise("kl. 12")
    assert normalise("12:00") == normalise("12")


def test_ratio_and_numeric_slash_never_glue_operands():
    assert normalise("3:2") == normalise("3 til 2")
    assert normalise("3:2") != normalise("32")
    assert normalise("4/5") == normalise("4 5")
    assert normalise("4/5") != normalise("45")


def test_percent_symbol_matches_spoken_unit_in_issue_114():
    assert normalise("100% sikkerhed") == normalise("hundrede procents sikkerhed")
    assert normalise("15 % rabat") == normalise("femten procent rabat")
    assert normalise("15% rabat") != normalise("15 rabat")


@pytest.mark.parametrize("written,spoken", [
    ("Marianergraven er næsten 11 km dyb", "Marianergraven er næsten elleve kilometer dyb"),
    ("50 cm højt", "halvtreds centimeter højt"),
    ("5 kg", "fem kilogram"),
    ("300 kr. udbetalt", "trehundrede kroner udbetalt"),
    ("ca 450 hyldemeter", "cirka 450 hyldemeter"),
    ("7 pct", "syv procent"),
    ("lige dér", "lige der"),
    ("dét var én idé", "det var en ide"),
    ("Vil De maaske være saa venlig", "Vil De måske være så venlig"),
    ("Nu skulle han passe paa", "Nu skulle han passe på"),
    ("han veed det", "han ved det"),
    ("Det skete idag", "Det skete i dag"),
    ("det skete igår", "det skete i går"),
    ("der er ialt tre", "der er i alt tre"),
    ("iøvrigt er han her", "i øvrigt er han her"),
    ("hun tog afsted", "hun tog af sted"),
    ("han var tilstede", "han var til stede"),
    ("nu går vi igang", "nu går vi i gang"),
    ("bl.a. børn", "blandt andet børn"),
    ("fx danske ord", "for eksempel danske ord"),
    ("f.eks. danske ord", "for eksempel danske ord"),
    ("osv.", "og så videre"),
    ("dvs. i morgen", "det vil sige i morgen"),
    ("m.m.", "med mere"),
    ("m.fl.", "med flere"),
    ("mv.", "med videre"),
    ("evt.", "eventuelt"),
    ("inkl.", "inklusive"),
    ("ekskl.", "eksklusive"),
    ("6 m", "seks meter"),
    ("18 mm", "atten millimeter"),
    ("7500 m²", "syvtusindefemhundrede kvadratmeter"),
    ("80 km/t", "firs kilometer i timen"),
    ("80 km/h", "firs kilometer i timen"),
    ("80km/h", "firs kilometer i timen"),
    ("5 m/s", "fem meter i sekundet"),
    ("5m/s", "fem meter i sekundet"),
    ("5 mio.", "fem millioner"),
    ("2 mia.", "to milliarder"),
])
def test_audited_transcript_spellings_score_the_same(written, spoken):
    assert normalise(written) == normalise(spoken)


def test_new_rules_leave_unrelated_words_and_notation_untouched():
    assert normalise("Aagaard") == "aagaard"
    assert normalise("ei") == "ei"
    assert normalise("Ca") == "ca"  # chemical symbol, without an abbreviation dot
    assert normalise("Köln") == "köln"  # acute-only fold retains other marks
    assert normalise("e.kr.") == "ekr"  # after Christ, not currency
    assert normalise("f.Kr.") == "fkr"  # before Christ, not currency
    assert normalise("mm") == "mm"  # bare sound/letters, not millimetres
    assert normalise("m") == "m"  # bare letter, not metres
    assert normalise("m.m.") != normalise("mm")
    assert normalise("kmm") == "kmm"  # not a unit abbreviation
    assert normalise("model fx8") == "model fx8"  # embedded acronym
    assert normalise("indag") == "indag"  # joined forms match only whole words
    assert normalise("Casper") == "casper"


def test_magnitude_abbreviations_keep_the_amount():
    assert normalise("1 mio.") == normalise("en million")
    assert normalise("5 mio.") == normalise("fem millioner")
    assert normalise("1 mia.") == normalise("en milliard")
    assert normalise("2 mia.") == normalise("to milliarder")
    assert normalise("5 mio.") != normalise("6 mio.")


@pytest.mark.parametrize("text,expected", [
    ("1-3", "et tre"),
    ("side 4-5", "side fire fem"),
    ("2020–2021", "totusinde og tyve totusinde og enogtyve"),  # en-dash
])
def test_digit_range_is_split_not_glued(text, expected):
    """A dash between digits becomes a space.

    Deleting it glued the numbers together, so "1-3" scored as "tretten" and the
    time range "10 00-11 00" as "ti elleve nul" -- references no correct
    transcription could match, which also handed free matches to digit-emitting
    models while penalising ones that spelled the range out.
    """
    assert normalise(text) == expected


def test_dash_between_letters_still_dropped():
    """Only digit-digit dashes are protected; hyphenated words keep old behaviour."""
    assert normalise("fire-fem") == "firefem"
    assert normalise("e-mail") == "email"


def test_dash_not_between_digits_is_dropped():
    assert normalise("minus -5") == "minus fem"


def test_time_range_keeps_both_times_separate():
    """The case that motivated the fix: "10 00-11 00" is a spoken time range.

    Deleting the dash produced "ti elleve nul", which no correct transcription
    could match, so a model that said "ti til elleve" was scored wrong.
    """
    assert normalise("mellem 10 00-11 00") == "mellem ti nul elleve nul"


@pytest.mark.parametrize("dash", ["-", "‐", "‑", "‒", "–", "—", "−"])
def test_all_dash_variants_split_digits(dash):
    assert normalise(f"1{dash}3") == "et tre"
