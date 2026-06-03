"""Google Cloud Speech-to-Text v2 (Chirp 3) backend."""
from __future__ import annotations

import importlib
import os

from danish_asr_leaderboard.backends.base import Backend, LoadOptions, register


class GoogleChirpBackend(Backend):
    name = "google-chirp"

    def __init__(self, client, project_id, model_id, *, options=None):
        super().__init__(client, options=options)
        self.project_id = project_id
        self.model_id = model_id

    def transcribe_one(self, audio_path: str) -> str:
        cloud_speech = importlib.import_module("google.cloud.speech_v2.types.cloud_speech")
        with open(audio_path, "rb") as f:
            audio_content = f.read()
        config = cloud_speech.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDecodingConfig(),
            language_codes=["da-DK"],
            model=self.model_id,
        )
        request = cloud_speech.RecognizeRequest(
            recognizer=f"projects/{self.project_id}/locations/global/recognizers/_",
            config=config,
            content=audio_content,
        )
        response = self.model.recognize(request=request)
        texts = [alt.transcript for res in response.results for alt in res.alternatives[:1]]
        return " ".join(texts).strip()


@register("google-chirp")
def load(model_ref: str, options: LoadOptions) -> Backend:
    try:
        speech_v2_mod = importlib.import_module("google.cloud.speech_v2")
        SpeechClient = speech_v2_mod.SpeechClient
    except Exception as exc:
        raise RuntimeError(
            "google-cloud-speech is not installed. Install: pip install google-cloud-speech"
        ) from exc

    project_id = options.google_cloud_project or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if not project_id:
        raise ValueError(
            "Google Cloud project ID required: pass --google-cloud-project or set GOOGLE_CLOUD_PROJECT"
        )
    creds_file = options.google_credentials_file or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if creds_file:
        service_account_mod = importlib.import_module("google.oauth2.service_account")
        credentials = service_account_mod.Credentials.from_service_account_file(creds_file)
        client = SpeechClient(credentials=credentials)
    else:
        client = SpeechClient()
    print(
        f"  Google Cloud STT v2 client ready (project={project_id}, model={options.google_chirp_model_id}) "
        f"[API — speed is network-bound]"
    )
    return GoogleChirpBackend(client, project_id, options.google_chirp_model_id, options=options)
