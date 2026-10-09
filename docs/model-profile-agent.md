# Model profile review agent

The weekly [profile refresh workflow](../.github/workflows/refresh-model-profiles.yml)
first identifies new or changed public model sources. If the repository has an
`OPENROUTER_API_KEY` Actions secret, it also asks
`nvidia/nemotron-3-super-120b-a12b:free` for field-by-field **review
suggestions**. The default model can be changed with `OPENROUTER_MODEL`.

The agent reads the exact revision of a Hugging Face model card, checkpoint
license files, linked dataset cards and training files, and base-model card and
license when `base_model` metadata identifies one. It checks these questions:

| Field | Evidence needed for a suggested Yes |
|---|---|
| License | Effective checkpoint and inherited terms allow commercial use and redistribution without separate permission. |
| Training data | The complete training corpus is identified, accessible, and reusable. |
| Training code | Public executable training and preprocessing code plus the released checkpoint's run configuration. |
| Model card | Substantive documentation for this checkpoint, including provenance, evaluation, and limitations. |
| Formatting | Cased and punctuated transcripts on this model's path. |
| Timestamps | Danish word or segment timestamps. |
| Speakers | Speaker-labeled transcription. |
| Streaming | Incremental transcription while live audio arrives. |

For each proposed `yes`, `partial`, or `no`, it must provide exact source quotes,
including multiple quotes when a decision depends on checkpoint and base terms.
The script checks every quote and source ID before writing
`scripts/model_profile_agent_suggestions.json`.
Unsupported claims become `unknown`. The output records source URLs and text
hashes so reviewers can recheck what the agent saw. Gated or missing sources
remain for manual review. Hosted APIs are skipped until their provider docs can
be collected with endpoint-specific rules. Only public source text is sent to
OpenRouter.

**Suggestions never alter leaderboard scores.** A reviewer must confirm the
claim applies to the evaluated checkpoint, inspect effective and inherited
license terms, and then edit `scripts/model_profiles.json`. In particular, a
Hub license tag, linked dataset, parent paper, or generic fine-tuning script
cannot earn a positive point on its own. Run the agent against the existing
human-reviewed profiles before considering any automatic approval of narrow
cases. Free model availability and rate limits may change.

The workflow's review PR also requires the repository Actions setting **Allow
GitHub Actions to create and approve pull requests**. Without the key, source
collection still runs and the agent step is skipped.
