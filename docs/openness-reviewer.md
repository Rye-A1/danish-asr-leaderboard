# Openness reviewer draft

This branch keeps the reviewer agent separate from the leaderboard UI in PR #104.
It reviews **only** the model licence, training data, training code, and model
card for an exact Hugging Face checkpoint. The leaderboard derives downloadable
weights from repository access; a checkpoint-specific paper remains a manual
annotation.

Run a local draft with `python scripts/openness_reviewer_agent.py --model
OWNER/MODEL`. Set `OPENROUTER_API_KEY` in the environment or enter it at the
hidden terminal prompt. The tool fetches public, revision-pinned model files and
public source links, then asks the configured OpenRouter model for evidence
quotes. It accepts only quotes found in the fetched text. The result is saved
under ignored `eval_audio_cache/openness-reviewer/suggestions.json`.

The draft does not edit `scripts/model_metadata.json`, score models, or publish
results. Review the effective licence, inherited terms, data completeness, and
checkpoint-specific code before changing any annotation. Free model responses
can vary or fail under rate limits. This branch does not include a scheduled
workflow; that should be added only after the narrowed reviewer is tested
against the existing manual annotations.
