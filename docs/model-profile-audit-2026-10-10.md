# Model profile audit — 10 October 2026

All 44 current result models have a reviewed profile in
[`scripts/model_profiles.json`](../scripts/model_profiles.json). That file is the
field-by-field record of states, explanations, and evidence links. The earlier
[40-model audit](model-profile-audit-2026-09-27.md) records the original source
review; its historical scores should not be read as the current scores.

This pass rechecked the available cards and metadata for all 37 downloadable
model entries, including the four gated cards through an accepted Hugging Face
login, and reviewed the seven hosted entries against model-specific provider
information and saved raw outputs. It completed the three missing profiles and
checked earlier decisions against the current license, data, and capability
rules. Source silence remains `unknown`, rather than `no`. API entries receive
0/5 openness by leaderboard policy; their feature decisions still need evidence.

| Newly completed model | Openness | Features | Key evidence and limit |
|---|---:|---:|---|
| [Brage v1](https://huggingface.co/Harmonium/brage-v1) | 4/5 | 1/4 | The gated card names all six gold-transcript training corpora. Its actual [license](https://huggingface.co/Harmonium/brage-v1/blob/main/LICENSE) permits commercial use and redistribution subject to its published conditions. The card gives a training recipe and inference decoder, but no released executable training run. Saved transcripts confirm formatting; timestamps, speakers, and live streaming remain unverified. |
| [Edda v0.2](https://huggingface.co/danish-foundation-models/edda-v0.2) | 4/5 | 1/4 | Apache 2.0 checkpoint, complete public six-corpus fine-tuning mix, detailed recipe without an executable run, and confirmed cased and punctuated output. A no-timestamps training token does not prove timestamp output is impossible. |
| Saga 2 L | 0/5 | 1/4 | Hosted API policy sets openness to zero. [Saved raw output](https://huggingface.co/datasets/RyeAI/danish-asr-leaderboard/blob/main/outputs/saga-2-l.parquet) confirms formatting; available official material does not establish model-specific timestamp, speaker, or live-streaming responses. |

The reviewed rubric now treats a complete mix of named, publicly obtainable
training corpora as open data even when ordinary access conditions apply. A
newly generated pseudo-labeled or synthetic corpus needs a direct link to that
corpus; links to its source audio or teacher do not establish its availability.
Commercial redistribution under published responsible-use terms can score
`yes` for the license factor, but a Hub license tag alone cannot. The exact
checkpoint's license and inherited base terms must be checked. Separate
punctuation restorers, forced aligners, diarizers, and generic provider API
features do not transfer capabilities to a scored ASR checkpoint.

The automated agent remains a **review aid**. Its source-quoted suggestions are
compared with these human decisions by `scripts/compare_model_profile_agent.py`;
neither agreement nor a validated quote changes leaderboard scores without
review. Gated or inaccessible cards remain manual work for the public-source
agent.
