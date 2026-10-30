"""Private evaluation contract, using synthetic audio and a local fake backend."""
import csv
import hashlib
import json
import os
import shutil
import stat
import sys
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from danish_asr_leaderboard import cli, private_dataset
from danish_asr_leaderboard.backends.api._base import ApiBackend
from danish_asr_leaderboard.backends.base import Backend, TranscriptionError
from danish_asr_leaderboard.custom_datasets import load_config


@pytest.fixture
def local_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    audio = tmp_path / "clip.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\x01\x00" * 160)
    path = tmp_path / "test.csv"
    path.write_text("audio_path,reference_text\nclip.wav,Hej med dig\n", encoding="utf-8")
    (tmp_path / "custom_datasets.py").write_text(
        "from danish_asr_leaderboard import DatasetConfig\n"
        "MY_DATASET = DatasetConfig(name='my-dataset', source={'test': 'test.csv'})\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(private_dataset, "audio_to_wav", lambda src, dst: bool(shutil.copy(src, dst)))
    return path


def arguments(monkeypatch, *extra, name="my-dataset"):
    monkeypatch.setattr(sys, "argv", [
        "danish-asr-eval", "--model", "example/model", "--backend", "transformers",
        "--dataset", name, *extra,
    ])
    return cli.parse_args()


def test_relative_paths_and_content_addressed_cache(local_data, tmp_path, monkeypatch):
    _, source = load_config("my-dataset")
    monkeypatch.chdir(tmp_path.parent)
    cache = tmp_path / "cache"
    rows, first = private_dataset.load_local_dataset(source, cache)
    assert Path(rows[0]["audio_path"]).is_file()
    assert rows[0]["reference_text"] == "Hej med dig"
    again, same = private_dataset.load_local_dataset(source, cache)
    assert (rows, first) == (again, same)
    audio = local_data.parent / "clip.wav"
    audio.write_bytes(audio.read_bytes()[:-2] + b"\x02\x00")
    changed, second = private_dataset.load_local_dataset(source, cache)
    assert second != first
    assert changed[0]["audio_path"] != rows[0]["audio_path"]


@pytest.mark.parametrize("content,error", [
    ("audio_path,reference_text\n", "empty"), ("audio_path,text\n", "CSV columns"),
    ("audio_path,reference_text,reference_text\n", "unique CSV columns"),
    ("audio_path,reference_text\nclip.wav, \n", "nonempty"),
    ("audio_path,reference_text\nclip.wav\n", "complete CSV row"),
    ("audio_path,reference_text\nclip.wav,hej,extra\n", "complete CSV row"),
    ("audio_path,reference_text\nmissing.wav,hej\n", "not found"),
])
def test_bad_input_fails_before_model_load(local_data, monkeypatch, content, error):
    local_data.write_text(content, encoding="utf-8")
    arguments(monkeypatch)
    loader = Mock(side_effect=AssertionError("model must not load"))
    monkeypatch.setattr(private_dataset, "load_backend", loader)
    with pytest.raises((ValueError, FileNotFoundError, csv.Error), match=error):
        cli.main()
    loader.assert_not_called()


@pytest.mark.parametrize("config_text,error", [
    (None, "define it in"), ("", "exactly one"),
    ("A = DatasetConfig('my-dataset', {})", "'test' CSV path"),
    ("A = DatasetConfig('my-dataset', {'test': 'missing.csv'})", "not found"),
])
def test_invalid_config_fails_before_model_load(local_data, monkeypatch, config_text, error):
    config_file = local_data.parent / "custom_datasets.py"
    if config_text is None:
        config_file.unlink()
    else:
        config_file.write_text("from danish_asr_leaderboard import DatasetConfig\n" + config_text,
                               encoding="utf-8")
    arguments(monkeypatch)
    loader = Mock(side_effect=AssertionError("model must not load"))
    monkeypatch.setattr(private_dataset, "load_backend", loader)
    with pytest.raises((ValueError, FileNotFoundError), match=error):
        cli.main()
    loader.assert_not_called()


def test_quotes_and_utf8_bom(local_data, tmp_path):
    local_data.write_text('audio_path,reference_text\nclip.wav,"Hej, med dig\nog dig"\n',
                          encoding="utf-8-sig", newline="")
    rows, _ = private_dataset.load_local_dataset(local_data, tmp_path / "cache")
    assert rows[0]["reference_text"] == "Hej, med dig\nog dig"


def test_failed_conversion_is_not_silently_skipped(local_data, tmp_path, monkeypatch):
    monkeypatch.setattr(private_dataset, "audio_to_wav", lambda src, dst: False)
    with pytest.raises(ValueError, match="conversion failed"):
        private_dataset.load_local_dataset(local_data, tmp_path / "cache")


def test_limit_and_empty_audio(local_data, tmp_path):
    content = local_data.read_text(encoding="utf-8")
    local_data.write_text(content + "clip.wav,Hej igen\n", encoding="utf-8")
    assert len(private_dataset.load_local_dataset(local_data, tmp_path / "cache", 1)[0]) == 1
    with pytest.raises(ValueError, match="nonnegative"):
        private_dataset.load_local_dataset(local_data, tmp_path / "cache", -1)
    with wave.open(str(tmp_path / "clip.wav"), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
    with pytest.raises(ValueError, match="empty or unreadable"):
        private_dataset.load_local_dataset(local_data, tmp_path / "cache")


@pytest.mark.parametrize("extra,limit,raw_saved", [([], 0, True),
    (["--max-samples", "1", "--outputs-dir", ""], 1, False),
])
def test_private_cli_scores_without_public_data_or_notifications(
    local_data, tmp_path, monkeypatch, extra, limit, raw_saved,
):
    public = tmp_path / "results" / "example__model.json"
    public.parent.mkdir()
    public.write_text("existing public result", encoding="utf-8")
    arguments(monkeypatch, "--model-revision", "f" * 40,
              "--elevenlabs-api-key", "never-save-this", *extra)
    backend = Mock()
    backend.transcribe.return_value = ["hej med mig"]
    monkeypatch.setattr(private_dataset, "load_backend", Mock(return_value=backend))
    monkeypatch.setattr(cli, "load_backend", Mock(side_effect=AssertionError("public path")))
    notify = Mock()
    monkeypatch.setattr(cli, "_notify", notify)
    cli.main()
    result_path, = (tmp_path / "results" / "private").rglob("*.json")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["wer"] == pytest.approx(100 / 3)
    assert result["cer"] == pytest.approx(100 / 11)
    assert result["samples"] == 1
    assert result["private_dataset"] is True
    assert result["dataset"] == "my-dataset"
    assert result["max_samples"] == limit
    assert result_path.parent.name == ("smoke-1" if limit else "full")
    assert result["model_revision"] == "f" * 40
    assert "mean_wer" not in result and "mean_cer" not in result
    assert "never-save-this" not in result_path.read_text(encoding="utf-8")
    assert public.read_text(encoding="utf-8") == "existing public result"
    assert list((tmp_path / "results").glob("*.json")) == [public]
    assert bool(list((tmp_path / "outputs").rglob("*.jsonl"))) is raw_saved
    backend.release.assert_called_once()
    notify.assert_not_called()


def test_singular_public_dataset_uses_existing_loader(local_data, monkeypatch):
    arguments(monkeypatch, name="fleurs")
    loader = Mock(return_value=[{"audio_path": "clip.wav", "reference_text": "hej"}])
    monkeypatch.setattr(cli, "DATASETS", {"fleurs": SimpleNamespace(
        loader=loader, column="fleurs", title="FLEURS")})
    monkeypatch.setattr(cli, "load_backend", Mock(side_effect=RuntimeError("public backend")))
    monkeypatch.setattr(cli, "evaluate_private", Mock(side_effect=AssertionError("private path")))
    with pytest.raises(RuntimeError, match="public backend"):
        cli.main()
    loader.assert_called_once()


def test_incomplete_transcription_fails_and_releases(local_data, tmp_path, monkeypatch):
    arguments(monkeypatch)
    backend = Mock()
    backend.transcribe.return_value = []
    monkeypatch.setattr(private_dataset, "load_backend", Mock(return_value=backend))
    with pytest.raises(ValueError, match="one transcript per recording"):
        cli.main()
    backend.release.assert_called_once()
    assert not (tmp_path / "results").exists()


def two_clips(local_data, tmp_path):
    audio = tmp_path / "second.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\x02\x00" * 160)
    with local_data.open("a", encoding="utf-8") as stream:
        stream.write("second.wav,Hej med dig\n")
    return hashlib.sha256(audio.read_bytes()).hexdigest() + ".wav"


@pytest.mark.parametrize("concurrent", [False, True])
def test_partial_transcription_failure_does_not_save_a_score(
    local_data, tmp_path, monkeypatch, concurrent,
):
    failed_name = two_clips(local_data, tmp_path)

    class Local(Backend):
        def transcribe_one(self, path):
            calls.append(path)
            if Path(path).name == failed_name:
                raise RuntimeError("failed clip")
            return "hej med dig"

    class Hosted(ApiBackend):
        concurrency = 2

        def _call(self, path):
            calls.append(path)
            if Path(path).name == failed_name:
                raise RuntimeError("failed clip")
            return "hej med dig"

    calls = []
    backend = (Hosted if concurrent else Local)(None)
    backend.release = Mock()
    arguments(monkeypatch)
    monkeypatch.setattr(private_dataset, "load_backend", lambda *args: backend)
    with pytest.raises(TranscriptionError, match="failed clip"):
        cli.main()
    backend.release.assert_called_once()
    assert len(calls) == len(set(calls))  # no paid retry after a per-clip failure
    assert not (tmp_path / "results").exists()
    assert not (tmp_path / "outputs").exists()


def test_private_allows_empty_transcripts_and_batch_fallback(local_data, tmp_path, monkeypatch):
    two_clips(local_data, tmp_path)

    class BatchFallback(Backend):
        def transcribe_batch(self, paths, *, batch_size):
            raise RuntimeError("batch failed")

        def transcribe_one(self, path):
            return "" if Path(path).name == failed_name else "hej med dig"

        def release(self):
            pass

    failed_name = hashlib.sha256((tmp_path / "second.wav").read_bytes()).hexdigest() + ".wav"
    backend = BatchFallback(None)
    arguments(monkeypatch)
    monkeypatch.setattr(private_dataset, "load_backend", lambda *args: backend)
    cli.main()
    result = json.loads(next((tmp_path / "results").rglob("*.json")).read_text())
    assert result["samples"] == 2
    assert result["wer"] == 50.0


@pytest.mark.skipif(os.name != "posix", reason="Unix permission bits")
@pytest.mark.parametrize("provider", ["mistral", "syv"])
def test_private_files_and_existing_cache_are_owner_only(
    local_data, tmp_path, monkeypatch, provider,
):
    for parent in ("eval_audio_cache/private", "results/private", "outputs/private"):
        path = tmp_path / parent
        path.mkdir(parents=True)
        path.chmod(0o755)
    cache = tmp_path / "provider-cache.jsonl"
    cache.write_text("", encoding="utf-8")
    cache.chmod(0o644)
    arguments(monkeypatch, "--backend", provider, f"--{provider}-cache", str(cache))
    backend = Mock()
    backend.transcribe.return_value = ["hej med dig"]
    monkeypatch.setattr(private_dataset, "load_backend", lambda *args: backend)
    old_umask = os.umask(0o022)
    try:
        cli.main()
    finally:
        os.umask(old_umask)
    for parent in ("eval_audio_cache/private", "results/private", "outputs/private"):
        assert stat.S_IMODE((tmp_path / parent).stat().st_mode) == 0o700
    for path in [cache, *tmp_path.rglob("*.jsonl"), *tmp_path.rglob("*.json"),
                 *(tmp_path / "eval_audio_cache/private").glob("*.wav")]:
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
