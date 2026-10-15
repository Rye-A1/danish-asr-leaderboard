"""Guard the published speaker breakdown against split or reference drift."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from build_compare_data import metadata_groups, output_slugs, preserve_examples, score_example, score_groups


def test_breakdown_requires_exact_test_rows_and_references():
    metadata = {"ftspeech": {
        "ftspeech_00000.wav": {"reference": "Dette er en prøve", "gender": "F", "speaker": "A"},
        "ftspeech_00001.wav": {"reference": "Det er en anden", "gender": "M", "speaker": "B"},
    }}
    groups = metadata_groups(metadata)
    outputs = pd.DataFrame([
        {"id": "cache/ftspeech_00000.wav", "reference": "Dette er en prøve", "hypothesis": "Dette er en prøve"},
        {"id": "cache/ftspeech_00001.wav", "reference": "Det er en anden", "hypothesis": "Det er anden"},
    ])
    scores = score_groups(outputs, "ftspeech", metadata, groups)
    assert scores["gender"]["Women"]["wer"] == 0
    assert scores["gender"]["Men"]["wer"] > 0
    assert score_groups(outputs.iloc[:1], "ftspeech", metadata, groups) is None
    changed = outputs.copy()
    changed.loc[1, "reference"] = "Different reference"
    assert score_groups(changed, "ftspeech", metadata, groups) is None
    with pytest.raises(ValueError, match="duplicate output ID"):
        score_groups(pd.concat([outputs.iloc[:1], outputs.iloc[:1]]), "ftspeech", metadata, groups)


def test_legacy_output_slug_uses_model_name(tmp_path):
    source = tmp_path / "short__model"
    source.mkdir()
    (source / "meta.json").write_text('{"model":"publisher/model"}')
    assert output_slugs(tmp_path)["short__model"] == "publisher__model"


def test_preserve_examples_keeps_clip_selection_without_stale_scores(tmp_path):
    path = tmp_path / "compare_data.json"
    path.write_text(json.dumps({"examples": {"fleurs_da": [
        {"id": "fleurs_da_00001.wav", "reference": "hej", "hypotheses": {"old": "wrong"},
         "scores": {"old": {"wer": 100}}}
    ]}}))
    assert preserve_examples(path) == {"fleurs_da": {
        "fleurs_da_00001.wav": {"id": "fleurs_da_00001.wav", "reference": "hej", "hypotheses": {}}
    }}


def test_example_scores_use_leaderboard_normalization():
    assert score_example("Jeg har 4 æbler.", "jeg har fire æbler") == {"wer": 0.0, "cer": 0.0}
    assert score_example("2.4ghz- og 5.0ghz-frekvenserne", "2,4 GHz og 5,00 GHz frekvenserne") == {"wer": 0.0, "cer": 0.0}
    assert score_example("hundrede procents sikkerhed", "100% sikkerhed") == {"wer": 0.0, "cer": 0.0}
    assert score_example("Hej verden", "") == {"wer": 100.0, "cer": 100.0}
    assert score_example("Hej verden", "Hej verden igen") == {"wer": 50.0, "cer": 50.0}


def test_fleurs_playback_matches_published_examples():
    space = Path(__file__).resolve().parent.parent / "huggingface" / "space"
    examples = json.loads((space / "compare_data.json").read_text())["examples"]["fleurs_da"]
    audio_dir = space / "audio" / "fleurs_da"
    manifest = json.loads((audio_dir / "manifest.json").read_text())["examples"]
    assert len(manifest) == len(examples)
    for example in examples:
        clip_id = example["id"]
        assert manifest[clip_id] == example["reference"]
        clip = audio_dir / clip_id.replace(".wav", ".m4a")
        assert clip.is_file() and clip.stat().st_size > 1000
        assert clip.read_bytes()[4:8] == b"ftyp"
