# Model profile evidence audit — 2026-09-27

Scope: the 40 distinct models in `results/*.json` at `origin/main` commit `05cfdca`, merged into the `codex/openness-features` review branch. Every Hugging Face model card and its metadata were fetched; gated cards were read with an existing accepted Hugging Face login. Public API descriptions, linked licenses, publisher GitHub repositories, and relevant model output examples were checked. The per-field decision, detail, and direct evidence URL are in [`scripts/model_profiles.json`](../scripts/model_profiles.json).

## Decision rules

- **Openness (out of five):** downloadable weights; effective license allowing commercial use and redistribution without separate permission; identified and accessible training data; executable training/preprocessing code for the released checkpoint, with its run configuration; and a substantive checkpoint-specific model card. A detailed recipe or reusable fine-tuning script without the released run is **partial** for code. Generic inference code is not training code. Papers remain unscored citations.

- **Features (out of four):** cased and punctuated text (enabled by default or selectable); Danish word/segment timestamps; speaker-labeled transcription; and incremental transcription while live audio arrives. File-output deltas after uploading a completed recording and offline chunking are **partial** or **no** for streaming.

- **States:** `Y` confirmed, `P` partial, `N` documented absent or restrictive, `?` not established. Only `Y` earns a point. A Hub license tag never overrides narrower card, gated-access, or LICENSE-file terms.

## All 40 models

| Model and primary source | W | L | D | C | M | Open | F | T | S | R | Feat |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| [3dio-ai/svale-110M](https://huggingface.co/3dio-ai/svale-110M) | Y | P | P | Y | P | 2/5 | N | ? | ? | ? | 0/4 |
| [3dio-ai/svale-600M](https://huggingface.co/3dio-ai/svale-600M) | Y | P | P | Y | P | 2/5 | N | ? | ? | ? | 0/4 |
| [capacit-ai/saga](https://huggingface.co/capacit-ai/saga) | Y | P | P | ? | P | 1/5 | Y | ? | ? | P | 1/4 |
| [capacit-ai/saga-2-m](https://huggingface.co/capacit-ai/saga-2-m) | Y | N | P | P | Y | 2/5 | Y | P | N | N | 1/4 |
| [CoRal-project/roest-v2-wav2vec2-1B](https://huggingface.co/CoRal-project/roest-v2-wav2vec2-1B) | Y | P | P | Y | Y | 3/5 | N | ? | ? | ? | 0/4 |
| [CoRal-project/roest-v2-wav2vec2-2B](https://huggingface.co/CoRal-project/roest-v2-wav2vec2-2B) | Y | P | P | Y | Y | 3/5 | N | ? | ? | ? | 0/4 |
| [CoRal-project/roest-v3-wav2vec2-315m](https://huggingface.co/CoRal-project/roest-v3-wav2vec2-315m) | Y | P | P | Y | P | 2/5 | N | ? | ? | ? | 0/4 |
| [CoRal-project/roest-v3-whisper-1.5b](https://huggingface.co/CoRal-project/roest-v3-whisper-1.5b) | Y | P | P | Y | P | 2/5 | Y | ? | ? | ? | 1/4 |
| [danish-foundation-models/edda-v0.1](https://huggingface.co/danish-foundation-models/edda-v0.1) | Y | Y | P | P | Y | 3/5 | Y | ? | ? | N | 1/4 |
| [facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all) | Y | N | ? | P | P | 1/5 | N | ? | ? | ? | 0/4 |
| [facebook/seamless-m4t-v2-large](https://huggingface.co/facebook/seamless-m4t-v2-large) | Y | N | ? | P | P | 1/5 | Y | ? | ? | ? | 1/4 |
| [gpt-4o-mini-transcribe](https://developers.openai.com/api/docs/guides/speech-to-text) | N | N | ? | ? | ? | 0/5 | Y | N | N | P | 1/4 |
| [gpt-4o-transcribe](https://developers.openai.com/api/docs/guides/speech-to-text) | N | N | ? | ? | ? | 0/5 | Y | N | N | P | 1/4 |
| [gpt-transcribe](https://developers.openai.com/api/docs/guides/speech-to-text) | N | N | ? | ? | ? | 0/5 | Y | N | N | P | 1/4 |
| [hinge/danstral-v1](https://huggingface.co/hinge/danstral-v1) | Y | ? | P | Y | Y | 3/5 | N | ? | ? | ? | 0/4 |
| [MediaCatch/xls-r-300m-danish-mc-v2](https://huggingface.co/MediaCatch/xls-r-300m-danish-mc-v2) | Y | ? | ? | P | P | 1/5 | N | ? | ? | ? | 0/4 |
| [mistralai/Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) | Y | Y | ? | ? | P | 2/5 | Y | ? | ? | ? | 1/4 |
| [mistralai/Voxtral-Small-24B-2507](https://huggingface.co/mistralai/Voxtral-Small-24B-2507) | Y | Y | ? | ? | P | 2/5 | Y | ? | ? | ? | 1/4 |
| [nvidia/canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2) | Y | Y | P | Y | Y | 4/5 | Y | Y | ? | ? | 2/4 |
| [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | Y | Y | P | P | Y | 3/5 | Y | ? | ? | Y | 2/4 |
| [nvidia/parakeet-rnnt-110m-da-dk](https://huggingface.co/nvidia/parakeet-rnnt-110m-da-dk) | Y | P | P | P | Y | 2/5 | N | ? | ? | ? | 0/4 |
| [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | Y | Y | P | Y | Y | 4/5 | Y | Y | ? | Y | 3/4 |
| [openai/whisper-base](https://huggingface.co/openai/whisper-base) | Y | Y | N | ? | Y | 3/5 | Y | Y | ? | N | 2/4 |
| [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | Y | Y | N | ? | Y | 3/5 | Y | Y | ? | N | 2/4 |
| [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | Y | Y | N | ? | Y | 3/5 | Y | Y | ? | N | 2/4 |
| [openai/whisper-small](https://huggingface.co/openai/whisper-small) | Y | Y | N | ? | Y | 3/5 | Y | Y | ? | N | 2/4 |
| [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) | Y | Y | N | ? | Y | 3/5 | Y | Y | ? | N | 2/4 |
| [ordbogen/whisper](https://odincore.ai/docs/models/ordbogen-whisper) | N | N | ? | ? | ? | 0/5 | Y | ? | ? | ? | 1/4 |
| [pluttodk/milo-asr](https://huggingface.co/pluttodk/milo-asr) | Y | ? | P | P | P | 1/5 | Y | P | ? | Y | 2/4 |
| [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | Y | Y | ? | P | P | 2/5 | Y | N | ? | Y | 2/4 |
| [scribe_v2](https://elevenlabs.io/docs/overview/capabilities/speech-to-text) | N | N | ? | ? | ? | 0/5 | Y | Y | Y | N | 3/4 |
| [syv-transcribe](https://syv.ai/) | N | N | ? | ? | ? | 0/5 | Y | ? | ? | ? | 1/4 |
| [syvai/hviske-v3-conversation](https://huggingface.co/syvai/hviske-v3-conversation) | Y | N | ? | ? | P | 1/5 | Y | ? | ? | ? | 1/4 |
| [syvai/hviske-v5](https://huggingface.co/syvai/hviske-v5) | Y | N | P | P | P | 1/5 | Y | ? | ? | ? | 1/4 |
| [syvai/hviske-v5-tiny](https://huggingface.co/syvai/hviske-v5-tiny) | Y | N | ? | ? | Y | 2/5 | Y | N | N | N | 1/4 |
| [syvai/hviske-v5.1](https://huggingface.co/syvai/hviske-v5.1) | Y | N | P | P | P | 1/5 | Y | ? | ? | ? | 1/4 |
| [syvai/hviske-v5.3](https://huggingface.co/syvai/hviske-v5.3) | Y | N | P | P | P | 1/5 | Y | ? | ? | ? | 1/4 |
| [syvai/hviske-v6](https://huggingface.co/syvai/hviske-v6) | Y | N | P | P | Y | 2/5 | Y | N | N | N | 1/4 |
| [thorhojhus/whisper-large-v3-turbo-danish](https://huggingface.co/thorhojhus/whisper-large-v3-turbo-danish) | Y | Y | P | P | Y | 3/5 | Y | P | ? | ? | 1/4 |
| [thorhojhus/whisper-small-danish](https://huggingface.co/thorhojhus/whisper-small-danish) | Y | Y | P | P | Y | 3/5 | Y | P | ? | ? | 1/4 |

`W/L/D/C/M` = weights/license/data/code/model card. `F/T/S/R` = formatting/timestamps/speakers/streaming.

## Evidence that changed the profiles

- **Svale 110M and 600M:** The publisher's [Svale repository](https://github.com/3dio-aps/svale) provides [`modal_finetune.py`](https://github.com/3dio-aps/svale/blob/main/modal_finetune.py), data preparation, and separate [110M](https://github.com/3dio-aps/svale/blob/main/models/svale-110M/run.json) and [600M](https://github.com/3dio-aps/svale/blob/main/models/svale-600M/run.json) run configurations. Both training-code decisions are now `Y`. The cards apply [NVIDIA model terms](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) plus CoRal use restrictions, so license remains `P`, even though commercial use is generally possible.
- **Røst family:** The [CoRal repository](https://github.com/alexandrainst/coral) contains `src/coral/finetune.py`, an ASR fine-tuning entry point, and model configurations. The four cards identify the training command or framework. CoRal data require access acceptance, and its OpenRAIL-derived model terms restrict speech synthesis and biometric identification; these are `P` for data and license, not unrestricted `Y`.
- **Saga 2 M and Hviske:** Saga 2 M's [gated card](https://huggingface.co/capacit-ai/saga-2-m) and [LICENSE](https://huggingface.co/capacit-ai/saga-2-m/blob/main/LICENSE) explicitly say CC BY-NC 4.0 from 2026-09-24; older Apache revisions do not govern the current weights. [Hviske v5](https://huggingface.co/syvai/hviske-v5), [v5.1](https://huggingface.co/syvai/hviske-v5.1), [v5.3](https://huggingface.co/syvai/hviske-v5.3), [v5 tiny](https://huggingface.co/syvai/hviske-v5-tiny), and [v6](https://huggingface.co/syvai/hviske-v6) carry noncommercial terms. The previously ambiguous [Hviske v3 LICENSE](https://huggingface.co/syvai/hviske-v3-conversation/blob/main/LICENSE) also requires separate contact for commercial use. All six Hviske license decisions are `N`.
- **Hviske training code:** Syv's [public training framework](https://github.com/syv-ai/hviske/blob/main/src/scripts/finetune_asr_model.py) has reusable model and dataset presets, but no run files tying it to v5, v5.1, or v5.3. Those code states are `P`. The v3 card and the distinct v5 tiny/v6 architectures do not establish equivalent training code.
- **Edda:** The new [checkpoint card](https://huggingface.co/danish-foundation-models/edda-v0.1) gives the base, exact corpus splits and counts, training hyperparameters, evaluation, limitations, and Apache 2.0 weight terms. It links no executable training script, so code is `P`; CoRal access keeps data `P`. Its saved raw outputs contain casing and punctuation together in 2,297 of 26,780 transcriptions.
- **Nemotron 3.5 ASR:** The new [card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) documents native incremental streaming and formatted text; 22,547 of 26,780 saved outputs contain both casing and punctuation. Its training blend includes private internal data, so data is `P`. [OpenMDW 1.1](https://openmdw.ai/license/1-1/) allows commercial use and redistribution with notice retention, so license is `Y`. NeMo inference and fine-tuning tools do not establish the released checkpoint's training run, so code is `P`.
- **Qwen and Milo:** [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning) publishes a reusable fine-tuning script, but not the original 1.7B pretraining run (`P` for code). [Milo's card](https://huggingface.co/pluttodk/milo-asr) describes its two-stage fine-tune but does not link its run script (`P`). The Qwen forced aligner does not list Danish; Milo's Danish timestamp support therefore remains `P`, rather than a confirmed point.
- **Whisper:** The [public repository](https://github.com/openai/whisper) contains inference code but does not establish original checkpoint training scripts (`?` for code). The five cards say the training audio is not released (`N` for data) and that Whisper cannot transcribe live audio in real time out of the box (`N` for native streaming). Their model cards and timestamp output remain `Y`.
- **Meta and Mistral:** [MMS](https://huggingface.co/facebook/mms-1b-all) and [SeamlessM4T v2](https://huggingface.co/facebook/seamless-m4t-v2-large) have CC BY-NC 4.0 terms (`N` license); their linked code covers the research toolkit or fine-tuning, not an exact public training run. [Voxtral Mini](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) and [Small](https://huggingface.co/mistralai/Voxtral-Small-24B-2507) have Apache 2.0 weights, but their cards do not publish the training corpus or checkpoint training code.
- **API models:** [Official OpenAI speech-to-text documentation](https://developers.openai.com/api/docs/guides/speech-to-text) directs timestamp requests to `whisper-1`, speaker labels to a separate diarization model, and live microphone transcription to Realtime; file-output streaming for the three scored API models is `P`. [ElevenLabs documentation](https://elevenlabs.io/docs/overview/capabilities/speech-to-text) confirms Scribe v2 word timestamps and up to 32 speakers; its realtime service is a separate model.

## Decisions needing better publisher evidence

1. [Danstral v1](https://huggingface.co/hinge/danstral-v1) links its [training script](https://github.com/ChristianHinge/danstral/blob/main/src/train.py), but states no effective checkpoint license. Its Voxtral parent license does not automatically resolve the fine-tune's terms.
2. [MediaCatch XLS-R](https://huggingface.co/MediaCatch/xls-r-300m-danish-mc-v2) names only a “Preprocessed Dataset”; the exact source and terms are missing. Its Apache Hub tag lacks a clear derivative-license statement.
3. [Milo](https://huggingface.co/pluttodk/milo-asr) labels the checkpoint OpenRAIL without spelling out commercial/redistribution restrictions. Its timestamp example uses a forced aligner whose documented languages exclude Danish.
4. [Capacit Saga](https://huggingface.co/capacit-ai/saga) has an Apache Hub tag, while the CoRal-derived training-data conditions are not reconciled in the card. License stays `P` pending explicit publisher clarification.
5. [Ordbogen](https://odincore.ai/docs/models/ordbogen-whisper) serves a near-empty JavaScript documentation shell to a text reader; [Syv Transcribe](https://syv.ai/) has only a general company page. Their timestamp, speaker, and native-streaming capabilities are not established by accessible model-specific documentation.
6. [Edda](https://huggingface.co/danish-foundation-models/edda-v0.1) trains with `<|notimestamps|>` but inherits Whisper architecture; reliable timestamp output is not documented. [Nemotron](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) documents streaming text but not Danish timestamp or speaker-ID output. [Svale](https://huggingface.co/3dio-ai/svale-110M) does not document native incremental inference. These remain `?` rather than inferred `N` or `Y`.

An unknown state means the public material does not settle the field. It is not a claim that the model lacks the capability.
