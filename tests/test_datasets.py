"""Dataset-loading regression guards (no network / GPU / `datasets` needed).

These lock in the disk-bomb fix: ``_materialise`` must *iterate* the dataset so it
works on streaming ``IterableDataset``s and never triggers HF's generate-all-splits
path. A streaming dataset has no ``__len__`` and no ``__getitem__`` — the
``_StreamingLike`` stand-in below has neither, so any regression to ``len(ds)`` or
``ds[i]`` raises immediately. Also covers Common Voice: both the pinned hub
dataset and a local ``CV_DATA_DIR`` copy must match the leaderboard's split.
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


def _cv_manifest(tmp_path, sentences):
    cv = tmp_path / "cv"
    (cv / "test").mkdir(parents=True)
    lines = []
    for i, text in enumerate(sentences):
        audio = cv / "test" / f"{i}.wav"
        audio.write_bytes(b"RIFF\x00\x00")
        lines.append(json.dumps({"audio_filepath": str(audio), "text": text}))
    (cv / "test" / "test_manifest.jsonl").write_text("\n".join(lines) + "\n")
    return cv


def _declare_canonical(monkeypatch, sentences):
    """Make a tiny fixture stand in for the real 2,756-row split."""
    monkeypatch.setattr(ds_mod, "CV_ROWS", len(sentences))
    monkeypatch.setattr(ds_mod, "CV_SENTENCE_SHA256", ds_mod.cv_fingerprint(sentences))


def test_load_common_voice_reads_local_manifest(tmp_path, monkeypatch):
    cv = _cv_manifest(tmp_path, ["én", "to"])
    _declare_canonical(monkeypatch, ["én", "to"])
    monkeypatch.setenv("CV_DATA_DIR", str(cv))
    out = load_common_voice(tmp_path, max_samples=0)
    assert out == [
        {"audio_path": str(cv / "test" / "0.wav"), "reference_text": "én"},
        {"audio_path": str(cv / "test" / "1.wav"), "reference_text": "to"},
    ]


def test_load_common_voice_rejects_a_different_local_split(tmp_path, monkeypatch):
    """The failure that motivated the check: a full run on another CV split.

    Two September 2026 submissions scored a 2,530-row split while the board uses
    2,756 rows; nothing stopped them, and the scores looked comparable.
    """
    cv = _cv_manifest(tmp_path, ["én", "to", "tre"])
    _declare_canonical(monkeypatch, ["én", "to"])
    monkeypatch.setenv("CV_DATA_DIR", str(cv))
    with pytest.raises(ValueError, match="not the leaderboard's"):
        load_common_voice(tmp_path, max_samples=0)


def test_load_common_voice_rejects_same_size_different_sentences(tmp_path, monkeypatch):
    cv = _cv_manifest(tmp_path, ["én", "fire"])
    _declare_canonical(monkeypatch, ["én", "to"])
    monkeypatch.setenv("CV_DATA_DIR", str(cv))
    with pytest.raises(ValueError, match="not the leaderboard's"):
        load_common_voice(tmp_path, max_samples=0)


def test_cv_fingerprint_ignores_order_and_file_names():
    """A renumbered copy of the right split must pass."""
    assert ds_mod.cv_fingerprint(["b", "a", "c"]) == ds_mod.cv_fingerprint(["c", "b", "a"])
    assert ds_mod.cv_fingerprint(["a", "b"]) != ds_mod.cv_fingerprint(["a", "b", "b"])


def test_load_common_voice_hub_path_is_checked(tmp_path, monkeypatch):
    monkeypatch.delenv("CV_DATA_DIR", raising=False)
    rows = [{"audio_path": "x.wav", "reference_text": t} for t in ("én", "to")]
    monkeypatch.setattr(ds_mod, "_materialise", lambda **kw: rows)
    _declare_canonical(monkeypatch, ["én", "to"])
    assert load_common_voice(tmp_path, max_samples=0) == rows
    _declare_canonical(monkeypatch, ["én", "tre"])
    with pytest.raises(ValueError, match="not the leaderboard's"):
        load_common_voice(tmp_path, max_samples=0)


def test_load_common_voice_smoke_runs_are_not_checked(tmp_path, monkeypatch):
    """A capped run can never match the full split, so it must not be blocked."""
    monkeypatch.delenv("CV_DATA_DIR", raising=False)
    rows = [{"audio_path": "x.wav", "reference_text": "én"}]
    monkeypatch.setattr(ds_mod, "_materialise", lambda **kw: rows)
    assert load_common_voice(tmp_path, max_samples=1) == rows


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
