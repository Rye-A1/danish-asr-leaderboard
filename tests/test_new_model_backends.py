from contextlib import nullcontext
import json
import sys
from types import SimpleNamespace

import pytest

from danish_asr_leaderboard.backends import LoadOptions, available_backends
from danish_asr_leaderboard.backends import lfm3_audio_backend as liquid
from danish_asr_leaderboard.backends import nemo_vendor_backend as vendor


def test_liquid_uses_danish_prompt_and_only_decodes_generated_tokens(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))

    class Inputs(dict):
        def to(self, device):
            assert device == "cuda:0"
            return self

    class Sequences:
        def __getitem__(self, key):
            assert key == (slice(None), slice(3, None))
            return "generated"

    class Processor:
        def apply_chat_template(self, conversations, **kwargs):
            assert conversations == [[
                {"role": "system", "content": "Perform ASR in Danish."},
                {"role": "user", "content": [{"type": "audio", "audio": "clip.wav"}]},
            ]]
            assert kwargs["add_generation_prompt"] and kwargs["tokenize"]
            return Inputs(input_ids=SimpleNamespace(shape=(1, 3)))

        def batch_decode(self, tokens, **kwargs):
            assert tokens == "generated"
            assert kwargs == {"skip_special_tokens": True}
            return [" transcript "]

    class Model:
        device = "cuda:0"

        def generate(self, **kwargs):
            assert not kwargs["do_sample"]
            assert kwargs["suppress_mixed"] and kwargs["suppress_audio_out"]
            assert kwargs["max_new_tokens"] == 1024
            return SimpleNamespace(sequences=Sequences())

    assert liquid.Lfm3AudioBackend(Model(), Processor()).transcribe_batch(
        ["clip.wav"], batch_size=1
    ) == ["transcript"]


def test_nemotron_manifest_sets_danish_prompt_and_guards_every_batch(monkeypatch):
    monkeypatch.setitem(sys.modules, "soundfile", SimpleNamespace(info=lambda _: SimpleNamespace(duration=2.5)))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        inference_mode=nullcontext, all=lambda value: SimpleNamespace(item=lambda: value)))
    monkeypatch.setattr(vendor, "_snapshot", lambda *_: __import__("pathlib").Path("/models"))

    class Model:
        cfg = SimpleNamespace(decoding=SimpleNamespace(strategy="greedy_batch"),
                              model_defaults=SimpleNamespace(prompt_dictionary={"da": 25, "auto": 0}))

        def _transcribe_forward(self, batch, cfg):
            return "ok"

        def transcribe(self, manifest, **kwargs):
            assert kwargs == {"batch_size": 1, "verbose": False, "target_lang": "da"}
            with open(manifest[0]) as stream:
                rows = [json.loads(line) for line in stream]
            assert rows == [{"audio_filepath": "clip.wav", "duration": 2.5,
                             "text": "", "lang": "da", "prompt_mode": "langID"}]
            assert self._transcribe_forward([None, None, None, None, 25],
                                            SimpleNamespace(target_lang="da")) == "ok"
            return [SimpleNamespace(text="hej")]

    monkeypatch.setattr(vendor, "_restore", lambda *_: Model())
    backend = vendor.load_nemotron("nvidia/model", LoadOptions())
    assert backend.transcribe(["clip.wav"], batch_size=1) == ["hej"]
    with pytest.raises(RuntimeError, match="did not use the Danish prompt"):
        backend.model._transcribe_forward([None, None, None, None, 0],
                                          SimpleNamespace(target_lang="da"))
    with pytest.raises(RuntimeError, match="did not use the Danish prompt"):
        backend.model._transcribe_forward([None, None, None, None, 25],
                                          SimpleNamespace(target_lang="auto"))


def test_transcribepsy_fails_if_any_adapter_did_not_execute(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    model = SimpleNamespace(transcribe=lambda *_, **__: [SimpleNamespace(text="hej")])
    backend = vendor.DanishNemoBackend(model, options=LoadOptions(), adapter_counts=[1, 1, 0, 1, 1, 1])
    with pytest.raises(RuntimeError, match="adapters did not run"):
        backend.transcribe(["clip.wav"], batch_size=1)


def test_pinned_snapshot_includes_checkpoint_and_adapter_files(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        snapshot_download=lambda ref, **kwargs: calls.append((ref, kwargs)) or "/cache"))
    assert str(vendor._snapshot("qvac/model", LoadOptions(model_revision="abc"),
                                ["model.nemo", "inference.py", "adapters.safetensors"])) == "/cache"
    assert calls == [("qvac/model", {"revision": "abc", "allow_patterns":
                                     ["model.nemo", "inference.py", "adapters.safetensors"]})]
    assert {"lfm3-audio", "nemotron-unified-da", "transcribepsy-da"} <= set(available_backends())
