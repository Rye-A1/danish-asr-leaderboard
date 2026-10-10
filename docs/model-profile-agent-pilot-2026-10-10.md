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
