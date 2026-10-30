"""Offline export checks using a miniature benchmark and synthetic transcripts."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation import soniox_report as report
from evaluation.checkpoints import PINS, sha256
from evaluation.results import NORMALIZATION, fingerprint, score, write_json


class ReportTests(unittest.TestCase):
    def setUp(self):
        cache = Path(__file__).resolve().parents[1] / ".cache"
        cache.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=cache)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        row = {
            "id": "fleurs:0",
            "reference_text": "hej verden",
            "audio_sha256": "audio",
            "duration": 1,
        }
        self.rows = [row]
        settings = {
            "model": "stt-async-v5",
            "language_hints": ["da"],
            "language_hints_strict": False,
            "enable_speaker_diarization": False,
            "enable_language_identification": False,
        }
        model = {"provider": "soniox", "id": "stt-async-v5", "revision": None}
        self.config = {
            "max_samples": 0,
            "model": model,
            "harness_revision": PINS["harness"],
            "normalization": NORMALIZATION,
            "translation": None,
            "context": None,
            "request_settings": settings,
            "code_sha256": {},
            "base_url": "endpoint",
            "baseline_sha256": sha256(Path(report.__file__).with_name("benchmark.json")),
        }
        self.record = {
            "id": row["id"],
            "reference": row["reference_text"],
            "hypothesis": "hej verden",
            "audio_seconds": 1,
            "batch_seconds_per_clip": 2,
            "transcription_id": "job",
            "requested_at": "2026-10-23T10:00:00+00:00",
            "completed_at": "2026-10-23T10:00:02+00:00",
        }
        metrics = {**score([self.record], 1), "audio_seconds": 1, "inference_seconds": 2, "rtf": 2}
        self.summary = {
            "model": model,
            "smoke_test": False,
            "datasets": {"fleurs": metrics},
            "core5_mean": {"wer": 0, "cer": 0},
        }
        job = {
            "audio_sha256": "audio",
            "settings": settings,
            "base_url": "endpoint",
            "remote_deleted": True,
            "transcription_id": "job",
            "elapsed_seconds": 2,
            "transcript": {"text": "hej verden"},
            "status_response": {
                "status": "completed",
                "model": "stt-async-v5",
                "id": "job",
                "language_hints": ["da"],
                "enable_speaker_diarization": False,
                "enable_language_identification": False,
            },
            "started_at": self.record["requested_at"],
            "completed_at": self.record["completed_at"],
        }
        self.job_path = self.folder / "jobs/fleurs/0.json"
        self.job_path.parent.mkdir(parents=True)
        write_json(self.job_path, job)
        (self.folder / "fleurs.jsonl").write_text(json.dumps(self.record) + "\n")
        spec = {**PINS["datasets"]["fleurs"], "expected_rows": 1}
        self.specs = {"fleurs": spec}
        self.data_info = {**spec, "samples": 1, "manifest_sha256": fingerprint(self.rows)}
        write_json(self.folder / "fleurs.data.json", self.data_info)
        write_json(self.folder / "api-preflight.json", {})

    def audit(self, partial=False):
        write_json(self.folder / "config.json", self.config)
        write_json(self.folder / "summary.json", self.summary)
        original_read = report.read
        with (
            patch.object(report, "PINS", {"harness": PINS["harness"], "datasets": self.specs}),
            patch.object(report, "load_rows", return_value=self.rows),
            patch.object(
                report,
                "read",
                side_effect=lambda p: {"datasets": {"fleurs": self.data_info}}
                if p.name == "benchmark.json"
                else original_read(p),
            ),
        ):
            return report.audit(self.folder, self.folder, allow_partial=partial)

    def test_valid_audit_recomputes_metrics_and_hashes(self):
        result = self.audit()
        self.assertTrue(result["complete"])
        model = result["models"]["soniox-stt-async-v5"]
        self.assertEqual(model["core5_mean"], {"wer": 0, "cer": 0})
        self.assertEqual(model["transcript_sha256"]["fleurs"], sha256(self.folder / "fleurs.jsonl"))

    def test_smoke_cannot_be_exported(self):
        self.summary["smoke_test"] = True
        with self.assertRaisesRegex(ValueError, "smoke"):
            self.audit()

    def test_incomplete_requires_explicit_partial_audit_and_has_no_mean(self):
        self.specs["missing"] = {"expected_rows": 1}
        self.summary["core5_mean"] = None
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.audit()
        result = self.audit(partial=True)
        self.assertFalse(result["complete"])
        self.assertIsNone(result["models"]["soniox-stt-async-v5"]["core5_mean"])

    def test_raw_transcript_tampering_is_rejected(self):
        job = report.read(self.job_path)
        job["transcript"]["text"] = "altered"
        write_json(self.job_path, job)
        with self.assertRaisesRegex(ValueError, "text mismatch"):
            self.audit()

    def test_summary_tampering_is_rejected(self):
        self.summary["datasets"]["fleurs"]["wer"] = 12
        with self.assertRaisesRegex(ValueError, "summary differs"):
            self.audit()

    def test_server_settings_mismatch_is_rejected(self):
        job = report.read(self.job_path)
        job["status_response"]["language_hints"] = ["en"]
        write_json(self.job_path, job)
        with self.assertRaisesRegex(ValueError, "server settings differ"):
            self.audit()


if __name__ == "__main__":
    unittest.main()
