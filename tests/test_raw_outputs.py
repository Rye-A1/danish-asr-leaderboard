"""Round-trip tests for the raw-output persistence helpers."""
from danish_asr_leaderboard.raw_outputs import (
    outputs_root,
    read_dataset_outputs,
    read_meta,
    write_dataset_outputs,
    write_meta,
)


def test_dataset_outputs_roundtrip(tmp_path):
    records = [
        {"id": "/cache/a.wav", "reference": "fire æbler", "hypothesis": "4 æbler"},
        {"id": "/cache/b.wav", "reference": "hej", "hypothesis": "hej"},
    ]
    path = write_dataset_outputs(tmp_path, "openai/whisper-large-v3", "cv17_da", records)
    assert path.exists()
    # slugified model dir, dataset-named file
    assert path.name == "cv17_da.jsonl"
    assert path.parent == outputs_root(tmp_path, "openai/whisper-large-v3")
    assert read_dataset_outputs(path) == records


def test_meta_roundtrip(tmp_path):
    meta = {"model": "openai/whisper-large-v3", "params_b": 1.5, "unicode_form": "NFC"}
    write_meta(tmp_path, "openai/whisper-large-v3", meta)
    assert read_meta(tmp_path, "openai/whisper-large-v3") == meta


def test_read_meta_missing_returns_empty(tmp_path):
    assert read_meta(tmp_path, "does/not-exist") == {}


def test_unicode_preserved_not_escaped(tmp_path):
    records = [{"id": "x", "reference": "blåbær", "hypothesis": "blåbær"}]
    path = write_dataset_outputs(tmp_path, "m", "fleurs_da", records)
    # Stored as real UTF-8, not \uXXXX escapes.
    assert "blåbær" in path.read_text(encoding="utf-8")
