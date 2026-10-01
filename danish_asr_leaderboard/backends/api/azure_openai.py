"""Azure OpenAI audio transcription backend."""
from __future__ import annotations

import importlib
import os
from urllib.parse import urlsplit, urlunsplit

from danish_asr_leaderboard.backends.base import Backend, LoadOptions, register
from danish_asr_leaderboard.backends.api._base import ApiBackend


class AzureOpenAIBackend(ApiBackend):
    name = "azure-openai"

    def __init__(self, client, deployment, *, options=None):
        super().__init__(client, options=options)
        self.deployment = deployment

    def _call(self, audio_path: str) -> str:
        # GPT-Transcribe uses the plural language hint; earlier transcription
        # models use the singular field. Never send both in the same request.
        language_hint = (
            {"languages": ["da"]}
            if self.deployment == "gpt-transcribe"
            else {"language": "da"}
        )
        with open(audio_path, "rb") as f:
            resp = self.model.audio.transcriptions.create(
                model=self.deployment, file=f, **language_hint
            )
        return (resp.text or "").strip()


def _resource_endpoint(endpoint: str) -> str:
    """AzureOpenAI adds its own /openai/deployments path to a resource URL."""
    parsed = urlsplit(endpoint)
    if parsed.hostname and parsed.hostname.endswith(
        (".openai.azure.com", ".cognitiveservices.azure.com")
    ):
        return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    return endpoint


@register("azure-openai")
def load(model_ref: str, options: LoadOptions) -> Backend:
    openai_mod = importlib.import_module("openai")
    api_key = options.azure_openai_api_key or os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint = options.azure_openai_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
    if not api_key or not endpoint:
        raise ValueError(
            "Azure OpenAI requires --azure-openai-api-key and --azure-openai-endpoint "
            "(or AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT)"
        )
    client = openai_mod.AzureOpenAI(
        api_key=api_key,
        azure_endpoint=_resource_endpoint(endpoint),
        api_version=options.azure_openai_api_version,
    )
    print(f"  Azure OpenAI client ready (deployment={model_ref}) [API — speed is network-bound]")
    return AzureOpenAIBackend(client, model_ref, options=options)
