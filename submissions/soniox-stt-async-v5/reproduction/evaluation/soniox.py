"""Small Soniox REST adapter with durable job state; no automatic paid retries."""

import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests

from evaluation.checkpoints import sha256
from evaluation.results import write_json as _write_json


def write_json(path, value):
    """Windows readers/antivirus can briefly block an otherwise atomic replace."""
    for attempt in range(6):
        try:
            return _write_json(path, value)
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.01 * 2**attempt)


class Soniox:
    def __init__(self, model="stt-async-v5", region="us", timeout=600, poll_seconds=2):
        self.key = os.environ.get("SONIOX_API_KEY", "").strip()
        if not self.key:
            raise ValueError("Set SONIOX_API_KEY locally before running Soniox evaluation")
        self.base_url = {
            "us": "https://api.soniox.com/v1",
            "eu": "https://api.eu.soniox.com/v1",
        }[region]
        self.timeout = timeout
        self.poll_seconds = poll_seconds
        self._thread = threading.local()
        self._sessions = []
        self._rate_lock = threading.Lock()
        self._requests_per_second = 12.0
        self._next_request = 0.0
        self._cooldown_until = 0.0
        self.settings = {
            "model": model,
            "language_hints": ["da"],
            "language_hints_strict": False,
            "enable_speaker_diarization": False,
            "enable_language_identification": False,
        }  # No translation or reference-derived context is sent.

    def request(self, method, route, data=None, content_type="application/json"):
        if isinstance(data, dict):
            data = json.dumps(data).encode()
        if not hasattr(self._thread, "session"):
            self._thread.session = requests.Session()
            self._sessions.append(self._thread.session)
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": content_type}
        for attempt in range(6):
            self._pace()
            try:
                with self._thread.session.request(
                    method,
                    self.base_url + route,
                    data=data,
                    headers=headers,
                    timeout=(10, 60),
                    allow_redirects=False,
                ) as response:
                    code = response.status_code
                    if method == "DELETE" and code == 404:
                        return {}
                    if 200 <= code < 300:
                        return response.json() if response.content else {}
                    retryable = code == 429 or (method != "POST" and code >= 500)
                    if not retryable or attempt == 5:
                        raise RuntimeError(f"Soniox {method} {route}: HTTP {code}")
                    delay = response.headers.get("Retry-After", "")
                    if code == 429:
                        try:
                            error = response.json()
                        except ValueError:
                            error = {}
                        self._rate_limit(delay, error)
                time.sleep(min(60, float(delay)) if delay.isdigit() else 2**attempt)
            except requests.RequestException:
                if method == "POST" or attempt == 5:
                    raise RuntimeError(f"Soniox {method} {route}: network failure") from None
                time.sleep(2**attempt)

    def _pace(self):
        while True:
            with self._rate_lock:
                now = time.monotonic()
                delay = max(self._next_request, self._cooldown_until) - now
                if delay <= 0:
                    self._next_request = now + 1 / self._requests_per_second
                    return
            time.sleep(min(delay, 0.25))

    def _rate_limit(self, retry_after, error):
        with self._rate_lock:
            now = time.monotonic()
            if now < self._cooldown_until:
                return
            seconds = max(60, float(retry_after)) if retry_after.isdigit() else 60
            self._cooldown_until = now + seconds
            self._requests_per_second = max(1, self._requests_per_second * 0.8)
            event = {
                "event": "soniox_rate_limit",
                "at": datetime.now(timezone.utc).isoformat(),
                "cooldown_seconds": seconds,
                "requests_per_second": self._requests_per_second,
                "error_type": error.get("error_type"),
                "message": str(error.get("message", error.get("error_message", ""))).replace(
                    self.key, "[REDACTED]"
                )[:512],
            }
        print(json.dumps(event), flush=True)

    def close(self):
        for session in self._sessions:
            session.close()

    def create(self, state, state_path, field, route, data, content_type="application/json"):
        # A lost response can mean a successful paid request. Never silently repeat it.
        state["pending_request"] = route
        write_json(state_path, state)
        response = self.request("POST", route, data, content_type)
        state[field] = response["id"]
        del state["pending_request"]
        write_json(state_path, state)

    def transcribe(self, audio_path, state_path):
        state_path = Path(state_path)
        identity = {
            "audio_sha256": sha256(audio_path),
            "settings": self.settings,
            "base_url": self.base_url,
        }
        state = (
            json.loads(state_path.read_text(encoding="utf-8"))
            if state_path.exists()
            else {
                **identity,
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        if any(state.get(k) != v for k, v in identity.items()):
            raise ValueError("Soniox job audio/settings changed; use another output directory")
        if state.get("pending_request"):
            raise RuntimeError(
                f"Uncertain POST outcome in {state_path}. Inspect Soniox Console before retrying; "
                "see evaluation/SONIOX.md for recovery."
            )
        state_path.parent.mkdir(parents=True, exist_ok=True)
        began = time.monotonic()
        previous_elapsed = state.get("elapsed_seconds", 0)
        if "file_id" not in state:
            boundary = uuid.uuid4().hex
            body = (
                (
                    f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
                    'filename="clip.wav"\r\nContent-Type: audio/wav\r\n\r\n'
                ).encode()
                + Path(audio_path).read_bytes()
                + f"\r\n--{boundary}--\r\n".encode()
            )
            self.create(
                state,
                state_path,
                "file_id",
                "/files",
                body,
                f"multipart/form-data; boundary={boundary}",
            )
        if "transcription_id" not in state:
            self.create(
                state,
                state_path,
                "transcription_id",
                "/transcriptions",
                {**self.settings, "file_id": state["file_id"]},
            )
        route = f"/transcriptions/{state['transcription_id']}"
        if "transcript" not in state:
            while True:
                status = self.request("GET", route)
                state["status_response"] = status
                state["elapsed_seconds"] = previous_elapsed + time.monotonic() - began
                write_json(state_path, state)
                if status["status"] == "error":
                    raise RuntimeError(f"Soniox job failed; inspect {state_path}")
                if status["status"] == "completed":
                    break
                if time.monotonic() - began >= self.timeout:
                    raise TimeoutError(f"Soniox job timed out; rerun to resume {state_path}")
                time.sleep(self.poll_seconds)
            transcript = self.request("GET", route + "/transcript")
            if not isinstance(transcript.get("text"), str):
                raise ValueError("Soniox returned no transcript text")
            state.update(
                transcript=transcript,
                completed_at=datetime.now(timezone.utc).isoformat(),
                elapsed_seconds=previous_elapsed + time.monotonic() - began,
            )
            write_json(state_path, state)  # Save raw output before deleting remote resources.
        if not state.get("remote_deleted"):
            self.request("DELETE", route)
            self.request("DELETE", f"/files/{state['file_id']}")
            state["remote_deleted"] = True
            write_json(state_path, state)
        return state
