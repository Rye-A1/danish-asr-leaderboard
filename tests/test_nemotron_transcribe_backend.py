from contextlib import nullcontext
from types import SimpleNamespace
import sys

import pytest

from danish_asr_leaderboard.backends import LoadOptions, available_backends
from danish_asr_leaderboard.backends import nemotron_transcribe_backend as backend_module


def test_transcription_batches_and_decodes_only_generated_tokens(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setattr(backend_module, "load_audio_array", lambda path: path)

    class Inputs(dict):
        input_ids = SimpleNamespace(shape=(2, 3))

        def to(self, device):
            assert device == "cuda:0"
            return self

    class Output:
        def __getitem__(self, key):
            assert key == (slice(None), slice(3, None))
            return "generated tokens"

    class Processor:
        def apply_transcription_request(self, audio, **kwargs):
            assert kwargs == {"prompt": backend_module.TRANSCRIPTION_PROMPT}
            self.audio = audio
            return Inputs()

        def batch_decode(self, output, **kwargs):
            assert output == "generated tokens"
            assert kwargs == {"skip_special_tokens": True}
            return [f" {item} " for item in self.audio]

    class Model:
        device = "cuda:0"

        def generate(self, **kwargs):
            assert kwargs == {"max_new_tokens": 1024, "do_sample": False}
            return Output()

    assert "nemotron-transcribe" in available_backends()
    backend = backend_module.NemotronTranscribeBackend(Model(), Processor())
    assert backend.transcribe_batch(["a", "b", "c"], batch_size=2) == ["a", "b", "c"]
    assert backend.fail_on_error


@pytest.mark.parametrize("with_device_map", [True, False])
def test_model_and_processor_use_the_same_revision(monkeypatch, with_device_map):
    calls = []

    class Model:
        device = "cuda:0"

        def eval(self):
            return self

    def model(ref, **kwargs):
        calls.append(("model", ref, kwargs))
        instance = Model()
        if with_device_map:
            instance.hf_device_map = {"": 0}
        return instance

    def processor(ref, **kwargs):
        calls.append(("processor", ref, kwargs))
        return object()

    monkeypatch.setattr(backend_module, "cuda_ok", lambda _: True)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForMultimodalLM=SimpleNamespace(from_pretrained=model),
        AutoProcessor=SimpleNamespace(from_pretrained=processor),
    ))
    backend_module.load("model", LoadOptions(model_revision="abc"))
    assert calls == [
        ("processor", "model", {"revision": "abc"}),
        ("model", "model", {"revision": "abc", "device_map": "auto", "dtype": "auto"}),
    ]


def test_wrong_number_of_transcripts_fails(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setattr(backend_module, "load_audio_array", lambda path: path)

    class Inputs(dict):
        input_ids = SimpleNamespace(shape=(1, 1))
        def to(self, _device):
            return self

    class Output:
        def __getitem__(self, _key):
            return []

    model = SimpleNamespace(device="cpu", generate=lambda **_: Output())
    processor = SimpleNamespace(apply_transcription_request=lambda *_, **__: Inputs(),
                                batch_decode=lambda *_, **__: [])
    backend = backend_module.NemotronTranscribeBackend(model, processor)
    with pytest.raises(RuntimeError, match="0 transcripts for 1 inputs"):
        backend.transcribe_batch(["a"], batch_size=1)
