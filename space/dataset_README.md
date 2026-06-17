---
language:
- da
license: mit
task_categories:
- automatic-speech-recognition
tags:
- benchmark
- danish
- asr
- evaluation
- leaderboard
pretty_name: Open Danish ASR Leaderboard — Results
size_categories:
- n<1K
---

# Open Danish ASR Leaderboard — Results

Benchmark results backing the **[Open Danish ASR Leaderboard](https://huggingface.co/spaces/RyeAI/danish-asr-leaderboard)** — a reproducible, open evaluation of Danish automatic speech recognition models across five independent public test sets.

Each row is one evaluated model. Scores are WER / CER (%) — lower is better.

## Test sets

| Column prefix | Dataset | Split | Domain |
|---|---|---|---|
| `coral_conversation` | [CoRal-project/coral-v3](https://huggingface.co/datasets/CoRal-project/coral-v3) — conversation | test | Spontaneous conversation |
| `coral_read_aloud` | [CoRal-project/coral-v3](https://huggingface.co/datasets/CoRal-project/coral-v3) — read_aloud | test | Read-aloud speech |
| `ftspeech` | [alexandrainst/ftspeech](https://huggingface.co/datasets/alexandrainst/ftspeech) | test_balanced | Parliamentary / broadcast |
| `cv17_da` | [mozilla-foundation/common_voice_17_0](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) — da | test | Crowd-sourced read speech |
| `fleurs_da` | [google/fleurs](https://huggingface.co/datasets/google/fleurs) — da_dk | test | Read speech |

Two Alvenir subsets (`alvenir_oss`, `alvenir_wiki`) can be included in runs but are excluded from the core means.

## Schema

| Column | Type | Description |
|---|---|---|
| `model` | string | Markdown link: `[org/name](https://huggingface.co/org/name)` for HF models, plain name for hosted APIs |
| `params_b` | float | Parameter count in billions from safetensors metadata; `NaN` for API models |
| `access` | string | `open` = open weights; `proprietary` = hosted or closed model |
| `mean_wer` | float | Macro-averaged WER (%) across the five core test sets |
| `mean_cer` | float | Macro-averaged CER (%) across the five core test sets |
| `coral_conversation_wer` | float\|null | WER on CoRal v3 conversation |
| `coral_read_aloud_wer` | float\|null | WER on CoRal v3 read-aloud |
| `ftspeech_wer` | float\|null | WER on FTSpeech |
| `cv17_da_wer` | float\|null | WER on Common Voice 17 (Danish) |
| `fleurs_da_wer` | float\|null | WER on FLEURS (Danish) |
| `alvenir_oss_wer` | float\|null | WER on Alvenir OSS (excluded from mean) |
| `alvenir_wiki_wer` | float\|null | WER on Alvenir Wiki (excluded from mean) |
| `coral_conversation_cer` | float\|null | CER on CoRal v3 conversation |
| `coral_read_aloud_cer` | float\|null | CER on CoRal v3 read-aloud |
| `ftspeech_cer` | float\|null | CER on FTSpeech |
| `cv17_da_cer` | float\|null | CER on Common Voice 17 (Danish) |
| `fleurs_da_cer` | float\|null | CER on FLEURS (Danish) |
| `alvenir_oss_cer` | float\|null | CER on Alvenir OSS (excluded from mean) |
| `alvenir_wiki_cer` | float\|null | CER on Alvenir Wiki (excluded from mean) |
| `speed_x` | float\|null | Audio seconds / wall-clock second (higher = faster). Measured on one RTX Pro 5000 Blackwell; network-bound for API models. `NaN` if not measured. |
| `submitted` | string | ISO 8601 date the result was submitted (`YYYY-MM-DD`) |

## Text normalisation

Applied identically to hypothesis and reference before scoring:

1. Unicode NFC
2. Danish number formatting — digit separators stripped (`1.234` → `1234`, `3,14` → `314`)
3. Lowercase
4. Strip punctuation (apostrophes inside words preserved)
5. Collapse whitespace

Digit–word equivalence (`"4"` vs `"fire"`) is intentionally **not** normalised. Danish orthographic variants (`aa`↔`å`, `oe`↔`ø`, `ae`↔`æ`) are **not** normalised either — the digraphs occur legitimately as letter sequences. See the [evaluation harness](https://github.com/Rye-A1/danish-asr-leaderboard) for full details.

## Submitting a result

Run the evaluation harness from the [GitHub repo](https://github.com/Rye-A1/danish-asr-leaderboard) and open a pull request with the result JSON, or start a [discussion on the Space](https://huggingface.co/spaces/RyeAI/danish-asr-leaderboard/discussions).

## License

MIT — see [LICENSE](https://github.com/Rye-A1/danish-asr-leaderboard/blob/main/LICENSE).
