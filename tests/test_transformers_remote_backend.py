"""The remote-code backend must load the reviewed Hub revision."""

import sys
from types import SimpleNamespace

import pytest

from danish_asr_leaderboard.backends import LoadOptions
from danish_asr_leaderboard.backends import transformers_remote_backend


REVISION = "fb0de9b0fd76386b2f475dee12eb65da25994060"


@pytest.mark.parametrize("revision", [None, "main", "fb0de9b0"])
def test_remote_model_requires_full_commit_sha(revision):
    with pytest.raises(ValueError, match="full 40-character commit SHA"):
        transformers_remote_backend.load(
            "danish-foundation-models/edda-v0.2-duo",
            LoadOptions(model_revision=revision),
        )


def test_same_revision_pins_model_and_processor(monkeypatch):
    calls = []

    class Model:
        dtype = "float32"

        def to(self, device):
            return self

        def eval(self):
            return self

    def load_model(ref, **kwargs):
        calls.append(("model", ref, kwargs))
        return Model()

    def load_processor(ref, **kwargs):
        calls.append(("processor", ref, kwargs))
        return object()

    monkeypatch.setattr(transformers_remote_backend, "cuda_ok", lambda device: False)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32="float32"))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForSpeechSeq2Seq=SimpleNamespace(from_pretrained=load_model),
        AutoProcessor=SimpleNamespace(from_pretrained=load_processor),
    ))

    transformers_remote_backend.load(
        "danish-foundation-models/edda-v0.2-duo",
        LoadOptions(device="cpu", model_revision=REVISION),
    )

    assert [call[0] for call in calls] == ["model", "processor"]
    assert all(call[2]["revision"] == REVISION for call in calls)
    assert all(call[2]["trust_remote_code"] is True for call in calls)
