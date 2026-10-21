"""Incremental builds must reuse only identical, fully published inputs."""

import sys
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

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
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("downloaded")))
    monkeypatch.setattr(ci, "source_signature", lambda *a, **kw: "scoring-v1")
    monkeypatch.setattr(ci, "_fetch_published_ci", lambda: {
        "a": {"mean_wer_ci": [1, 2], "mean_cer_ci": [1, 2]}})
    monkeypatch.setattr(ci, "_fetch_published_manifest", lambda: {
        "signature": "scoring-v1", "outputs": {"outputs/a.parquet": "abc"}})
    monkeypatch.setattr(pd, "read_parquet",
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("downloaded")))
    monkeypatch.setattr(sys, "argv", ["compute_ci.py"])
    ci.main()


@pytest.mark.parametrize("updating", [False, True], ids=["new-model", "updated-model"])
def test_ci_recomputes_only_new_or_updated_model(tmp_path, monkeypatch, updating):
    a_hash = "new-a" if updating else "old-a"
    siblings = [SimpleNamespace(rfilename="outputs/a.parquet",
                                lfs=SimpleNamespace(sha256=a_hash)),
                SimpleNamespace(rfilename="outputs/b.parquet",
                                lfs=SimpleNamespace(sha256="old-b" if updating else "new-b"))]
    class FakeApi:
        def repo_info(self, *args, **kwargs):
            return SimpleNamespace(sha="dataset-revision", siblings=siblings)
        def create_commit(self, *args, **kwargs):
            raise AssertionError("--no-upload must not publish")
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    downloads = []
    def fake_download(_, file, **kwargs):
        downloads.append(file)
        return file
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(ci, "source_signature", lambda *a, **kw: "scoring-v1")
    old_entry = {"mean_wer_ci": [0, 0], "mean_cer_ci": [0, 0],
                 "mean_wer_point": 0, "mean_cer_point": 0}
    published = {"a": old_entry.copy()}
    if updating:
        published["b"] = old_entry.copy()
    monkeypatch.setattr(ci, "_fetch_published_ci", lambda: published)
    monkeypatch.setattr(ci, "_fetch_published_manifest", lambda: {
        "signature": "scoring-v1",
        "outputs": {"outputs/a.parquet": "old-a",
                    **({"outputs/b.parquet": "old-b"} if updating else {})}})
    reads = []
    def fake_read(url):
        reads.append(url)
        return pd.DataFrame([{"dataset": "cv17_da", "reference": "hej", "hypothesis": "nej"}])
    monkeypatch.setattr(pd, "read_parquet", fake_read)
    monkeypatch.setattr(ci, "OUT_PATH", tmp_path / "ci.json")
    monkeypatch.setattr(ci, "MANIFEST_PATH", tmp_path / "ci_manifest.json")
    monkeypatch.setattr(sys, "argv", ["compute_ci.py", "-B", "20", "--no-upload"])
    ci.main()
    changed = "a" if updating else "b"
    unchanged = "b" if updating else "a"
    assert downloads == [f"outputs/{changed}.parquet"]
    assert reads == downloads
    result = json.loads((tmp_path / "ci.json").read_text())
    manifest = json.loads((tmp_path / "ci_manifest.json").read_text())
    assert result[unchanged] == old_entry
    assert result[changed]["mean_wer_point"] == 100
    assert manifest["outputs"][f"outputs/{changed}.parquet"] == siblings[0 if updating else 1].lfs.sha256
    assert manifest["outputs"][f"outputs/{unchanged}.parquet"] == ("old-b" if updating else "old-a")


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


@pytest.mark.parametrize("updating", [False, True], ids=["new-model", "updated-model"])
def test_compare_build_reuses_unchanged_model_and_reads_changed_one(tmp_path, monkeypatch,
                                                                     updating):
    examples = {dataset: [{"id": "clip.wav", "reference": "hej"}]
                for dataset in DATASETS}
    selection = tmp_path / "previous.json"
    selection.write_text(json.dumps({"examples": examples}))
    cached = {
        "metadata_revisions": {"ftspeech": "rev1"},
        "cache": {"signature": compare.compare_signature(50),
                  "outputs": {"outputs/a.parquet": "abc"}, "aliases": {}},
        "breakdown": {"a": {"ftspeech": {"gender": {"Women": {"wer": 0}}}}},
        "examples": {dataset: [{"id": "clip.wav", "reference": "hej",
                               "hypotheses": {"a": "hej"}}]
                     for dataset in DATASETS},
    }
    siblings = [SimpleNamespace(rfilename="outputs/a.parquet",
                                lfs=SimpleNamespace(sha256="new" if updating else "abc"))]
    if not updating:
        siblings.append(SimpleNamespace(rfilename="outputs/b.parquet",
                                        lfs=SimpleNamespace(sha256="new")))
    class FakeApi:
        def repo_info(self, *args, **kwargs):
            return SimpleNamespace(sha="new-dataset-commit", siblings=siblings)
    monkeypatch.setattr(compare, "load_metadata", lambda _: ({"ftspeech": {
        "clip.wav": {"reference": "hej", "gender": "F", "speaker": "s"}}},
        {"ftspeech": "rev1"}))
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
    monkeypatch.setattr(compare, "score_groups", lambda part, dataset, metadata, groups: {
        "gender": {"Women": {"wer": 100, "cer": 100}}})
    data = compare.build(tmp_path / "unused", tmp_path, tmp_path, 50, selection, cached)
    assert downloads == ["outputs/a.parquet" if updating else "outputs/b.parquet"]
    assert data["breakdown"]["a"]["ftspeech"]["gender"]["Women"]["wer"] == (
        100 if updating else 0)
    assert data["examples"]["ftspeech"][0]["hypotheses"] == (
        {"a": "ny"} if updating else {"a": "hej", "b": "ny"})
    assert data["cache"]["outputs"][downloads[0]] == "new"
