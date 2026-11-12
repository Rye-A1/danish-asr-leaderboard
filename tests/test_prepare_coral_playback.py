"""Check clip identity and failure behavior before preparing CoRal playback."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
import wave
from array import array
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_coral_playback.py"
spec = importlib.util.spec_from_file_location("prepare_coral_playback", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PrepareCoRalPlaybackTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source, self.manifests, self.outputs = (self.root / name for name in ("source", "manifests", "outputs"))
        self.manifests.mkdir()
        self.outputs.mkdir()
        self.mapping = self.root / "map.jsonl"
        self.license = self.root / "LICENSE"
        self.license.write_text("LICENSED MATERIAL\nDistribution and Redistribution\n")
        self.destination = self.root / "bundle"
        self.original_counts = module.COUNTS
        module.COUNTS = {"coral_conversation": 1, "coral_read_aloud": 1}
        mapping = []
        for dataset in module.COUNTS:
            audio = self.source / dataset / "00000.wav"
            audio.parent.mkdir(parents=True)
            with wave.open(str(audio), "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                wav.writeframes(array("h", [100, -100, 0] * 100).tobytes())
            name = f"{dataset}_00000.wav"
            ref = f"Reference for {dataset}"
            (self.manifests / f"{dataset}.manifest.jsonl").write_text(json.dumps({"audio_path": f"/original/{name}", "reference_text": ref}) + "\n")
            (self.outputs / f"{dataset}.jsonl").write_text(json.dumps({"id": name, "reference": ref}) + "\n")
            mapping.append([f"/original/{name}", f"/remote/{dataset}/00000.wav", audio.stat().st_size, hashlib.sha256(audio.read_bytes()).hexdigest()])
        self.mapping.write_text("".join(json.dumps(item) + "\n" for item in mapping))

    def tearDown(self):
        module.COUNTS = self.original_counts
        self.tmp.cleanup()

    def prepare(self):
        module.prepare(self.source, self.manifests, self.mapping, self.outputs, self.license, self.destination)

    def test_prepares_both_verified_datasets(self):
        self.prepare()
        for dataset in module.COUNTS:
            folder = self.destination / "audio" / dataset
            name = f"{dataset}_00000.wav"
            with wave.open(str(folder / name), "rb") as wav:
                samples = array("h", wav.readframes(wav.getnframes()))
            self.assertGreater(max(samples), 100)
            self.assertLessEqual(max(samples), 32767)
            self.assertEqual(json.loads((folder / "manifest.json").read_text())["examples"][name], f"Reference for {dataset}")
            self.assertEqual((folder / "LICENSE").read_text(), self.license.read_text())
            self.assertIn(self.license.read_bytes(), (folder / name).read_bytes())
            self.assertIn(b"converted to mono 16 kHz PCM", (folder / name).read_bytes())

    def test_silent_audio_stays_silent(self):
        source = self.root / "silent.wav"
        target = self.root / "silent-copy.wav"
        with wave.open(str(source), "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            wav.writeframes(b"\0\0" * 100)
        self.assertFalse(module.prepare_audio(source, target))
        self.assertEqual(source.read_bytes(), target.read_bytes())

    def test_already_loud_audio_is_unchanged(self):
        source = self.root / "loud.wav"
        target = self.root / "loud-copy.wav"
        with wave.open(str(source), "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            wav.writeframes(array("h", [20000, -20000] * 100).tobytes())
        self.assertFalse(module.prepare_audio(source, target))
        self.assertEqual(source.read_bytes(), target.read_bytes())

    def test_rejects_audio_mismatch_without_creating_bundle(self):
        (self.source / "coral_conversation/00000.wav").write_bytes(b"wrong")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.prepare()
        self.assertFalse(self.destination.exists())

    def test_rejects_reference_mismatch_without_creating_bundle(self):
        p = self.outputs / "coral_conversation.jsonl"
        p.write_text(json.dumps({"id": "coral_conversation_00000.wav", "reference": "wrong"}) + "\n")
        with self.assertRaisesRegex(ValueError, "Reference mismatch"):
            self.prepare()
        self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main()
