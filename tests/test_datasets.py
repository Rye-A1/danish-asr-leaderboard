"""Dataset-loading regression guards (no network / GPU / `datasets` needed).

These lock in the disk-bomb fix: ``_materialise`` must *iterate* the dataset so it
works on streaming ``IterableDataset``s and never triggers HF's generate-all-splits
path. A streaming dataset has no ``__len__`` and no ``__getitem__`` — the
``_StreamingLike`` stand-in below has neither, so any regression to ``len(ds)`` or
``ds[i]`` raises immediately. Also covers the cv17 local-manifest path
(``CV_DATA_DIR``), which is the supported way to load Common Voice.
"""
import json
from pathlib import Path

import pytest

import danish_asr_leaderboard.datasets as ds_mod
from danish_asr_leaderboard.datasets import _materialise, load_common_voice


class _StreamingLike:
    """Iterable-only dataset stand-in (mimics datasets.IterableDataset):
    no ``__len__``, no ``__getitem__``."""

    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


@pytest.fixture
def fake_transcode(monkeypatch):
    """Replace audio_bytes_to_wav with a stub that just writes the target file,
    so _materialise keeps the row without needing ffmpeg/real audio."""
    def _touch(audio, wav_path):
        Path(wav_path).write_bytes(b"RIFF\x00\x00")
    monkeypatch.setattr(ds_mod, "audio_bytes_to_wav", _touch)


def _rows(n, prefix="sætning"):
    return [{"audio": b"x", "text": f"{prefix} {i}"} for i in range(n)]


def test_materialise_iterates_streaming_dataset(tmp_path, fake_transcode):
    out = _materialise(
        slug="demo", label="Demo", audio_dir=tmp_path,
        load_ds=lambda: _StreamingLike(_rows(3)), text_keys=("text",), max_samples=0,
    )
    assert len(out) == 3
    assert out[0]["reference_text"] == "sætning 0"
    assert all(Path(r["audio_path"]).exists() for r in out)
    # full runs cache a manifest for reuse
    assert (tmp_path / "demo.manifest.jsonl").exists()


def test_materialise_respects_max_samples(tmp_path, fake_transcode):
    out = _materialise(
        slug="demo", label="Demo", audio_dir=tmp_path,
        load_ds=lambda: _StreamingLike(_rows(10)), text_keys=("text",), max_samples=4,
    )
    assert len(out) == 4
    # capped (smoke) runs must NOT write a manifest — it would cache a partial set
    assert not (tmp_path / "demo.manifest.jsonl").exists()


def test_materialise_skips_empty_text(tmp_path, fake_transcode):
    rows = [
        {"audio": b"x", "text": "hej"},
        {"audio": b"x", "text": "   "},   # whitespace-only → skipped
        {"audio": b"x", "text": "dav"},
    ]
    out = _materialise(
        slug="demo", label="Demo", audio_dir=tmp_path,
        load_ds=lambda: _StreamingLike(rows), text_keys=("text",), max_samples=0,
    )
    assert [r["reference_text"] for r in out] == ["hej", "dav"]


def test_materialise_first_matching_text_key(tmp_path, fake_transcode):
    rows = [{"audio": b"x", "sentence": "fra sentence", "text": "fra text"}]
    out = _materialise(
        slug="demo", label="Demo", audio_dir=tmp_path,
        load_ds=lambda: _StreamingLike(rows), text_keys=("sentence", "text"), max_samples=0,
    )
    assert out[0]["reference_text"] == "fra sentence"


def test_materialise_reuses_manifest_without_loading(tmp_path):
    """If a manifest exists (full run), load_ds is never called."""
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF\x00\x00")
    (tmp_path / "demo.manifest.jsonl").write_text(
        json.dumps({"audio_path": str(audio), "reference_text": "cached"}) + "\n"
    )

    def _boom():
        raise AssertionError("load_ds must not run when a manifest is cached")

    out = _materialise(
        slug="demo", label="Demo", audio_dir=tmp_path,
        load_ds=_boom, text_keys=("text",), max_samples=0,
    )
    assert out == [{"audio_path": str(audio), "reference_text": "cached"}]


def test_load_common_voice_reads_local_manifest(tmp_path, monkeypatch):
    cv = tmp_path / "cv"
    (cv / "test").mkdir(parents=True)
    a1, a2 = cv / "test" / "1.wav", cv / "test" / "2.wav"
    a1.write_bytes(b"RIFF\x00\x00")
    a2.write_bytes(b"RIFF\x00\x00")
    (cv / "test" / "test_manifest.jsonl").write_text(
        json.dumps({"audio_filepath": str(a1), "text": "én"}) + "\n"
        + json.dumps({"audio_filepath": str(a2), "text": "to"}) + "\n"
    )
    monkeypatch.setenv("CV_DATA_DIR", str(cv))
    out = load_common_voice(tmp_path, max_samples=0)
    assert out == [
        {"audio_path": str(a1), "reference_text": "én"},
        {"audio_path": str(a2), "reference_text": "to"},
    ]


def test_load_common_voice_local_manifest_respects_max_samples(tmp_path, monkeypatch):
    cv = tmp_path / "cv"
    (cv / "test").mkdir(parents=True)
    audio = cv / "test" / "1.wav"
    audio.write_bytes(b"RIFF\x00\x00")
    lines = [json.dumps({"audio_filepath": str(audio), "text": f"t{i}"}) for i in range(5)]
    (cv / "test" / "test_manifest.jsonl").write_text("\n".join(lines) + "\n")
    monkeypatch.setenv("CV_DATA_DIR", str(cv))
    out = load_common_voice(tmp_path, max_samples=2)
    assert len(out) == 2
