# Local profile agent pilot — 10 October 2026

The local pilot ran `nvidia/nemotron-3-super-120b-a12b:free` against the
current public source bundles for Edda v0.2 and Ekko v1 Tiny. The model saw
pinned checkpoint cards, accessible effective terms and training-source leads,
plus positive formatting counts from committed raw benchmark output. Its
suggestions were checked for source quotes before comparison with the
human-reviewed [`model_profiles.json`](../scripts/model_profiles.json). Neither
the pilot nor the comparison changed leaderboard scores.

| Model | Field | Human | Validated agent |
|---|---|---|---|
| Edda v0.2 | License, data, model card, formatting | yes | yes |
| Edda v0.2 | Training code | partial | partial |
| Edda v0.2 | Timestamps, diarization, streaming | unknown | unknown |
| Ekko v1 Tiny | License, timestamps | yes | yes |
| Ekko v1 Tiny | Training data | partial | partial |
| Ekko v1 Tiny | Training code, diarization, streaming | unknown | unknown |
| Ekko v1 Tiny | Formatting | no | no |
| Ekko v1 Tiny | Model card | yes | unknown |

**Result: 15/16 exact field-state matches.** Ekko's model-card suggestion was
`yes`, but one purported quote joined separate card bullet lines; the validator
discarded the claim. Edda's raw suggestion inferred `no` timestamps from the
`<|notimestamps|>` training target; validation kept `unknown` because that
target alone does not prove output support is impossible. The final prompt asks
for one short contiguous passage or bullet per quote.

**Repeat with that final prompt: 14/16 exact matches.** Ekko's model-card and
Edda's training-data suggestions were both `yes`, but the free model again
provided unsupported combined or empty quotes; validation kept both `unknown`.
It also repeated the unsupported Edda timestamp `no`, which validation rejected.
The change from 15/16 to 14/16 on the same two models shows run-to-run
variability despite temperature zero. These small, selected pilots test the
workflow and guards, not general reviewer accuracy. Gated cards still need
manual review, and no agent draft is applied to leaderboard scores.

**Offline recheck of that saved final response: 16/16.** We improved the
validator to split an oversized stitched excerpt into individual, exact,
source-ordered lines. Every substantive line must match the fetched card; a
fabricated line still invalidates the field. For training-data reviews, an
explicit complete-public-corpora statement in the checkpoint card can remain
as evidence when the agent also supplies unmatched ancillary dataset-card
license snippets. Those snippets are discarded and the draft receives a
visible warning to inspect every linked dataset. The recheck fetched the
same pinned model-card URLs and current public dataset-card URLs; it did not
call the language model again. The incorrect Edda timestamp `no` still fails
the explicit-negative-evidence guard and stays `unknown`. Agreement on these
two selected models does not establish accuracy across the other 42 models.

**Another live repeat: 15/16 before, 16/16 after an offline recheck.** The
new model response again suggested Ekko's model card as `yes`, but used a
standalone `...` line to mark omitted card text. The validator initially
rejected that stitched quote. It now treats only standalone ellipsis lines as
omission markers and still verifies every substantive line against the pinned
card in source order. Revalidating this saved response against the public
sources yields 16/16. Edda's unsupported timestamp `no` is still rejected;
one unmatched ancillary dataset-card quote is discarded with a review warning.
