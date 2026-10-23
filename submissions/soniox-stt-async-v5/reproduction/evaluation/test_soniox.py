"""Offline checks: never call Soniox or download audio."""

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import requests

from evaluation import soniox_evaluation as runner
from evaluation.soniox import Soniox


class SonioxTests(unittest.TestCase):
    def setUp(self):
        cache = Path(__file__).resolve().parents[1] / ".cache"
        cache.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=cache)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.audio = self.root / "clip.wav"
        self.audio.write_bytes(b"fake wav for mocked transport")
        self.state_path = self.root / "job.json"
        key = patch.dict(os.environ, {"SONIOX_API_KEY": "offline-secret"})
        key.start()
        self.addCleanup(key.stop)
        self.client = Soniox(poll_seconds=0)

    def responses(self, status="completed"):
        return [
            {"id": "file-1"},
            {"id": "job-1"},
            {"status": status},
            {"text": "Hej verden", "tokens": []},
            {},
            {},
        ]

    def test_upload_transcript_cleanup_and_cached_resume(self):
        with patch.object(self.client, "request", side_effect=self.responses()) as request:
            result = self.client.transcribe(self.audio, self.state_path)
            self.assertEqual(result["transcript"]["text"], "Hej verden")
            calls = request.call_args_list
            self.assertIn(b'name="file"', calls[0].args[2])
            payload = calls[1].args[2]
            self.assertEqual(payload["language_hints"], ["da"])
            self.assertNotIn("translation", payload)
            self.assertNotIn("context", payload)
            self.assertEqual(
                [c.args[0] for c in calls], ["POST", "POST", "GET", "GET", "DELETE", "DELETE"]
            )
        with patch.object(self.client, "request") as request:
            self.assertEqual(self.client.transcribe(self.audio, self.state_path), result)
            request.assert_not_called()
        self.assertNotIn("offline-secret", self.state_path.read_text())

    def test_poll_timeout_resumes_known_job_without_paid_post(self):
        self.client.timeout = 0
        with patch.object(self.client, "request", side_effect=self.responses("processing")[:3]):
            with self.assertRaises(TimeoutError):
                self.client.transcribe(self.audio, self.state_path)
        with patch.object(self.client, "request", side_effect=self.responses()[2:]) as request:
            self.client.transcribe(self.audio, self.state_path)
            self.assertTrue(all(c.args[0] != "POST" for c in request.call_args_list))

    def test_uncertain_create_never_automatically_repeats(self):
        with patch.object(self.client, "request", side_effect=[{"id": "file-1"}, OSError()]):
            with self.assertRaises(OSError):
                self.client.transcribe(self.audio, self.state_path)
        with patch.object(self.client, "request") as request:
            with self.assertRaisesRegex(RuntimeError, "Uncertain POST"):
                self.client.transcribe(self.audio, self.state_path)
            request.assert_not_called()

    def test_error_job_is_not_scored_as_empty_text(self):
        with patch.object(self.client, "request", side_effect=self.responses("error")[:3]):
            with self.assertRaisesRegex(RuntimeError, "job failed"):
                self.client.transcribe(self.audio, self.state_path)
        self.assertNotIn("transcript", json.loads(self.state_path.read_text()))

    def test_changed_audio_rejected(self):
        with patch.object(self.client, "request", side_effect=self.responses()):
            self.client.transcribe(self.audio, self.state_path)
        self.audio.write_bytes(b"different audio")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.client.transcribe(self.audio, self.state_path)

    def test_cleanup_failure_preserves_transcript_for_retry(self):
        with patch.object(self.client, "request", side_effect=self.responses()[:4] + [OSError()]):
            with self.assertRaises(OSError):
                self.client.transcribe(self.audio, self.state_path)
        with patch.object(self.client, "request", side_effect=[{}, {}]) as request:
            result = self.client.transcribe(self.audio, self.state_path)
            self.assertTrue(result["remote_deleted"])
            self.assertEqual([c.args[0] for c in request.call_args_list], ["DELETE", "DELETE"])

    def test_missing_key_fails_before_loading_data(self):
        with patch.dict(os.environ, {"SONIOX_API_KEY": ""}):
            with self.assertRaisesRegex(ValueError, "SONIOX_API_KEY"):
                Soniox()

    def test_connection_is_reused_and_post_network_errors_are_not_retried(self):
        with patch("evaluation.soniox.requests.Session") as session:
            transport = session.return_value
            response = transport.request.return_value.__enter__.return_value
            response.status_code = 200
            response.content = b"{}"
            response.json.return_value = {}
            self.client.request("GET", "/models")
            self.client.request("GET", "/models")
            session.assert_called_once()
            self.assertEqual(transport.request.call_count, 2)
            transport.request.reset_mock()
            transport.request.side_effect = requests.ConnectionError("offline-secret")
            with self.assertRaisesRegex(RuntimeError, "network failure") as error:
                self.client.request("POST", "/transcriptions", {})
            self.assertNotIn("offline-secret", str(error.exception))
            transport.request.assert_called_once()

    def test_transient_windows_save_lock_is_retried_without_api_requests(self):
        from evaluation.soniox import write_json

        with (
            patch("evaluation.soniox._write_json", side_effect=[PermissionError(), None]) as save,
            patch("evaluation.soniox.time.sleep"),
        ):
            write_json(self.state_path, {"test": True})
            self.assertEqual(save.call_count, 2)

    def test_rate_limit_has_shared_cooldown_and_does_not_compound_per_thread(self):
        self.client._rate_limit("1", {"error_type": "limit_exceeded", "message": "rate"})
        first_deadline = self.client._cooldown_until
        first_rate = self.client._requests_per_second
        self.client._rate_limit("1", {"error_type": "limit_exceeded", "message": "rate"})
        self.assertEqual(self.client._cooldown_until, first_deadline)
        self.assertEqual(self.client._requests_per_second, first_rate)
        self.assertLess(first_rate, 12)

    def test_rolling_workers_refill_before_slow_clip_finishes(self):
        released = threading.Event()
        args = SimpleNamespace(
            model="stt-async-v5",
            region="us",
            timeout=10,
            workers=2,
            output=self.root,
            cache=self.root,
            cv_dir=None,
            max_samples=3,
            datasets=["fleurs"],
        )
        rows = [
            {
                "id": f"fleurs:{i}",
                "audio_path": str(i),
                "reference_text": "hej",
                "duration": 1,
                "audio_sha256": str(i),
            }
            for i in range(3)
        ]

        def transcribe(audio_path, state_path):
            if audio_path == "1":
                if not released.wait(3):
                    raise AssertionError("Worker window stalled behind a slow clip")
            if audio_path == "2":
                released.set()
            return {
                "transcript": {"text": "hej"},
                "elapsed_seconds": 1,
                "transcription_id": audio_path,
                "started_at": "start",
                "completed_at": "end",
            }

        with (
            patch.object(runner, "load_rows", return_value=rows),
            patch.object(Soniox, "transcribe", side_effect=transcribe),
            patch.object(runner.platform, "platform", return_value="offline-platform"),
            patch.object(runner.subprocess, "check_output", return_value=b"ffmpeg offline"),
        ):
            folder = runner.main(args)
        records = [json.loads(line) for line in (folder / "fleurs.jsonl").read_text().splitlines()]
        self.assertTrue(released.is_set())
        self.assertEqual([r["id"] for r in records], [r["id"] for r in rows])

    def test_runner_reuses_rye_scores_and_resumes(self):
        args = SimpleNamespace(
            model="stt-async-v5",
            region="us",
            timeout=10,
            output=self.root,
            cache=self.root,
            cv_dir=None,
            max_samples=1,
            datasets=["fleurs"],
        )
        rows = [
            {
                "id": "fleurs:0",
                "audio_path": str(self.audio),
                "reference_text": "Hej verden",
                "duration": 1,
                "audio_sha256": "mocked",
            }
        ]
        job = {
            "transcript": {"text": "Hej verden"},
            "elapsed_seconds": 2,
            "transcription_id": "job-1",
            "started_at": "start",
            "completed_at": "end",
        }
        with (
            patch.object(runner, "load_rows", return_value=rows),
            patch.object(Soniox, "transcribe", return_value=job) as transcribe,
            patch.object(runner.platform, "platform", return_value="offline-platform"),
            patch.object(runner.subprocess, "check_output", return_value=b"ffmpeg offline"),
        ):
            folder = runner.main(args)
            runner.main(args)
            transcribe.assert_called_once()
            summary = json.loads((folder / "summary.json").read_text())
            self.assertTrue(summary["smoke_test"])
            self.assertIsNone(summary["core5_mean"])
            self.assertEqual(summary["datasets"]["fleurs"]["wer"], 0)
            args.region = "eu"
            # A settings mismatch within a run must reject reuse.
            with self.assertRaisesRegex(ValueError, "changed"):
                runner.unchanged(folder / "config.json", {"different": True})


if __name__ == "__main__":
    unittest.main()
