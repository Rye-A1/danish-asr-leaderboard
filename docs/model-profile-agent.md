# Model profile review agent

The weekly [profile refresh workflow](../.github/workflows/refresh-model-profiles.yml)
checks all current model sources. If the repository has an
`OPENROUTER_API_KEY` Actions secret, it also asks
`nvidia/nemotron-3-super-120b-a12b:free` for field-by-field **review suggestions**.
It requests a structured tool call and validates the returned quotes locally.
Requests are pinned to Nvidia with fallback disabled. Only public documents
are sent; no private benchmark data is included. The
model and provider can be changed with `OPENROUTER_MODEL` and
`OPENROUTER_PROVIDER` together.

For open models the agent reads the exact revision of a Hugging Face model card,
checkpoint license files, linked dataset cards and training files, and the
base-model card and license when `base_model` metadata identifies one. For API
models it reads the official pages in `scripts/provider_profile_sources.json`;
generic provider capabilities do not establish support for the scored model.
It checks these questions:

| Field | Evidence needed for a suggested Yes |
|---|---|
| License | Effective checkpoint and inherited terms allow commercial use and redistribution without separate permission. Published responsible-use conditions do not by themselves lower this leaderboard factor. |
| Training data | The full checkpoint training or fine-tuning mix is named and consists of publicly obtainable datasets. Published access conditions such as CoRal's are acceptable; an ordinary filtered copy need not be redistributed. A newly generated pseudo-labeled, synthetic, or teacher-labeled corpus needs its own direct access link. A source-audio or teacher-model link alone is insufficient. |
| Training code | Public executable training and preprocessing code plus the released checkpoint's run configuration. |
| Model card | Substantive documentation for this checkpoint, including provenance, evaluation, and limitations. |
| Formatting | Exact checkpoint card, official examples, or saved raw outputs show cased and punctuated transcripts. |
| Timestamps | Model-specific API option and word/segment timestamp response for Danish. |
| Speakers | This model's response includes speaker IDs; a different diarization model does not count. |
| Streaming | This model accepts audio incrementally and emits interim text; streamed text after full upload is partial. |

For each proposed `yes`, `partial`, or `no`, it must provide exact source quotes,
including multiple quotes when a decision depends on checkpoint and base terms.
The script checks every quote and source ID before writing
`scripts/model_profile_agent_suggestions.json`.
Unsupported claims become `unknown`. The output records source URLs and text
hashes so reviewers can recheck what the agent saw. Gated or missing sources
remain for manual review. Hosted API openness factors are `no` by leaderboard
policy; only their model-specific features are reviewed from official API docs.
Only public source text is sent to OpenRouter.

The first live pilot on Edda v0.1 yielded source-validated license and data
suggestions from Nemotron Super. That pilot used the former, stricter data
criterion; the updated public-dataset rule has not yet had a live pilot.
A speculative negative timestamp claim is now rejected, and other claims
without verbatim quotes stay `unknown`. The Gemma 4 free
endpoint returned provider rate limits in the same pilot. These observations
do not establish general model accuracy; human review remains required.

**Suggestions never alter leaderboard scores.** A reviewer must confirm the
claim applies to the evaluated checkpoint, inspect effective and inherited
license terms, and then edit `scripts/model_profiles.json`. In particular, a
Hub license tag, linked dataset, parent paper, or generic fine-tuning script
cannot earn a positive point on its own. Run the agent against the existing
human-reviewed profiles before considering any automatic approval of narrow
cases. Free model availability and rate limits may change.

The [pre-merge PR check](../.github/workflows/review-model-profile-pr.yml)
collects public leads for changed result files without the OpenRouter secret.
Running PR-controlled code with that secret is intentionally excluded. The
weekly agent runs from trusted main-branch code. Its review PR also requires the repository Actions setting **Allow
GitHub Actions to create and approve pull requests**. Without the key, source
collection still runs and the agent step is skipped.

For a live one-model check, dispatch the workflow on the PR branch with the
exact model ID and `pilot_only=true`. Optionally choose a different
`reviewer_model` and `reviewer_provider` to compare endpoints on the same
sources. The draft is retained as a workflow artifact; the pilot does not
create a review PR or change scores.
