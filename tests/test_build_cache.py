"""Incremental builds must reuse only identical, fully published inputs."""

import sys
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from build_cache import output_fingerprints, reusable
from build_compare_data import DATASETS, cache_matches
import build_compare_data as compare
import compute_ci as ci


def test_output_fingerprints_use_lfs_content_not_dataset_revision():
    siblings = [SimpleNamespace(rfilename="outputs/a.parquet",
                                lfs=SimpleNamespace(sha256="abc"), blob_id="old-commit"),
                SimpleNamespace(rfilename="outputs/b.parquet", lfs=None, blob_id="new-commit"),
                SimpleNamespace(rfilename="data/ci.json", lfs=None, blob_id="ignored")]
    assert output_fingerprints(siblings) == {
        "outputs/a.parquet": "abc", "outputs/b.parquet": None,
    }


def test_ci_reuse_requires_matching_signature_hash_and_published_entry():
    manifest = {"signature": "scoring-v1", "outputs": {"outputs/a.parquet": "abc"}}
    published = {"a": {"mean_wer_ci": [1, 2]}}
    assert reusable("a", "outputs/a.parquet", "abc", "scoring-v1", manifest, published)
    assert not reusable("a", "outputs/a.parquet", "changed", "scoring-v1", manifest, published)
    assert not reusable("a", "outputs/a.parquet", "abc", "scoring-v2", manifest, published)
    assert not reusable("a", "outputs/a.parquet", None, "scoring-v1", manifest, published)
    assert not reusable("b", "outputs/a.parquet", "abc", "scoring-v1", manifest, published)


def test_ci_main_skips_download_and_upload_for_unchanged_model(monkeypatch):
    sibling = SimpleNamespace(rfilename="outputs/a.parquet",
                              lfs=SimpleNamespace(sha256="abc"))
    class FakeApi:
        def repo_info(self, *args, **kwargs):
            return SimpleNamespace(sha="new-dataset-commit", siblings=[sibling])
        def create_commit(self, *args, **kwargs):
            raise AssertionError("uploaded unchanged intervals")
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(ci, "source_signature", lambda *a, **kw: "scoring-v1")
    monkeypatch.setattr(ci, "_fetch_published_ci", lambda: {
        "a": {"mean_wer_ci": [1, 2], "mean_cer_ci": [1, 2]}})
    monkeypatch.setattr(ci, "_fetch_published_manifest", lambda: {
        "signature": "scoring-v1", "outputs": {"outputs/a.parquet": "abc"}})
    monkeypatch.setattr(pd, "read_parquet",
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("downloaded")))
    monkeypatch.setattr(sys, "argv", ["compute_ci.py"])
    ci.main()


def test_compare_cache_invalidates_outputs_metadata_or_example_selection():
    selected = {dataset: {"clip.wav": {"id": "clip.wav", "reference": "hej"}}
                for dataset in DATASETS}
    old = {
        "metadata_revisions": {"ftspeech": "source-v1"},
        "cache": {"signature": "scoring-v1", "outputs": {"outputs/a.parquet": "abc"},
                  "aliases": {}},
        "breakdown": {"a": {"ftspeech": {}}},
        "examples": {dataset: [{"id": "clip.wav", "reference": "hej"}]
                     for dataset in DATASETS},
    }
    fingerprints = {"outputs/a.parquet": "abc"}
    assert cache_matches(old, fingerprints, "scoring-v1", {"ftspeech": "source-v1"},
                         selected, {})
    assert not cache_matches(old, {"outputs/a.parquet": "new"}, "scoring-v1",
                             {"ftspeech": "source-v1"}, selected, {})
    assert not cache_matches(old, fingerprints, "scoring-v1",
                             {"ftspeech": "source-v2"}, selected, {})
    assert not cache_matches(old, fingerprints, "scoring-v1",
                             {"ftspeech": "source-v1"}, selected, {"a": "renamed"})
    selected["ftspeech"]["clip.wav"]["reference"] = "changed"
    assert not cache_matches(old, fingerprints, "scoring-v1",
                             {"ftspeech": "source-v1"}, selected, {})


def test_compare_build_reuses_unchanged_model_and_reads_changed_one(tmp_path, monkeypatch):
    examples = {dataset: [{"id": "clip.wav", "reference": "hej"}]
                for dataset in DATASETS}
    selection = tmp_path / "previous.json"
    selection.write_text(json.dumps({"examples": examples}))
    cached = {
        "metadata_revisions": {},
        "cache": {"signature": compare.compare_signature(50),
                  "outputs": {"outputs/a.parquet": "abc"}, "aliases": {}},
        "breakdown": {"a": {"ftspeech": {"gender": {"Women": {"wer": 1}}}}},
        "examples": {dataset: [{"id": "clip.wav", "reference": "hej",
                               "hypotheses": {"a": "hej"}}]
                     for dataset in DATASETS},
    }
    siblings = [SimpleNamespace(rfilename="outputs/a.parquet",
                                lfs=SimpleNamespace(sha256="abc")),
                SimpleNamespace(rfilename="outputs/b.parquet",
                                lfs=SimpleNamespace(sha256="new"))]
    class FakeApi:
        def repo_info(self, *args, **kwargs):
            return SimpleNamespace(sha="new-dataset-commit", siblings=siblings)
    monkeypatch.setattr(compare, "load_metadata", lambda _: ({}, {}))
    monkeypatch.setattr(compare, "output_slugs", lambda _: {})
    monkeypatch.setattr(compare, "HfApi", FakeApi)
    downloads = []
    def fake_download(_, file, **kwargs):
        downloads.append(file)
        return tmp_path / "b.parquet"
    monkeypatch.setattr(compare, "hf_hub_download", fake_download)
    monkeypatch.setattr(compare.pd, "read_parquet", lambda *a, **kw: pd.DataFrame([
        {"dataset": dataset, "id": "clip.wav", "reference": "hej", "hypothesis": "ny"}
        for dataset in DATASETS
    ]))
    monkeypatch.setattr(compare, "normalise", lambda text, **kw: text)
    monkeypatch.setattr(compare, "score_example", lambda ref, hyp: {"wer": 0, "cer": 0})
    data = compare.build(tmp_path / "unused", tmp_path, tmp_path, 50, selection, cached)
    assert downloads == ["outputs/b.parquet"]
    assert data["breakdown"] == cached["breakdown"]
    assert data["examples"]["ftspeech"][0]["hypotheses"] == {"a": "hej", "b": "ny"}
    assert data["cache"]["outputs"] == {"outputs/a.parquet": "abc",
                                           "outputs/b.parquet": "new"}
