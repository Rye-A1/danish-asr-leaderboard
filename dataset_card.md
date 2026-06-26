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

<!-- This card is generated from dataset_card.md in the GitHub repo. Edit it there
     (not on the Hub) — the `configs:` block is injected automatically on deploy by
     scripts/update_dataset_card.py, and any direct Hub edits are overwritten. -->

# Open Danish ASR Leaderboard — Results

Benchmark results backing the **[Open Danish ASR Leaderboard](https://huggingface.co/spaces/RyeAI/danish-asr-leaderboard)** — a reproducible, open evaluation of Danish automatic speech recognition models across five independent public test sets.

The `results` config (shown by default) has one row per evaluated model. Scores are WER / CER (%) — lower is better. Each model also has its own config exposing the raw, un-normalised transcriptions (`reference` vs `hypothesis` per utterance) for GPU-free re-scoring and error analysis.

## Test sets

| Column prefix | Dataset | Split | Domain |
|---|---|---|---|
| `coral_conversation` | [CoRal-project/coral-v3](https://huggingface.co/datasets/CoRal-project/coral-v3) — conversation | test | Spontaneous conversation |
| `coral_read_aloud` | [CoRal-project/coral-v3](https://huggingface.co/datasets/CoRal-project/coral-v3) — read_aloud | test | Read-aloud speech |
| `ftspeech` | [alexandrainst/ftspeech](https://huggingface.co/datasets/alexandrainst/ftspeech) | test_balanced | Parliamentary / broadcast |
| `cv17_da` | [mozilla-foundation/common_voice_17_0](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) — da | test | Crowd-sourced read speech |
| `fleurs_da` | [google/fleurs](https://huggingface.co/datasets/google/fleurs) — da_dk | test | Read speech |

## Schema

`results` config — one row per model:

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
| `coral_conversation_cer` | float\|null | CER on CoRal v3 conversation |
| `coral_read_aloud_cer` | float\|null | CER on CoRal v3 read-aloud |
| `ftspeech_cer` | float\|null | CER on FTSpeech |
| `cv17_da_cer` | float\|null | CER on Common Voice 17 (Danish) |
| `fleurs_da_cer` | float\|null | CER on FLEURS (Danish) |
| `speed_x` | float\|null | Audio seconds / wall-clock second (higher = faster). Measured on one RTX Pro 5000 Blackwell; network-bound for API models. `NaN` if not measured. |
| `submitted` | string | ISO 8601 date the result was submitted (`YYYY-MM-DD`) |

Per-model configs (`outputs/<model-slug>`) — one row per utterance: `dataset`, `id`, `reference`, `hypothesis` (raw, un-normalised).

## Text normalisation

Applied identically to hypothesis and reference before scoring, so WER/CER reflect recognition errors rather than formatting:

1. **Unicode NFKC** — compatibility composition (folds ligatures, full-width digits, `²`→`2`, …). A near-no-op on Danish speech text, adopted for correctness and consistency with the Danish standard.
2. **Danish number canonicalisation** — separators within a numeral are stripped (`1.234` → `1234`, `3,14` → `314`).
3. **Lowercase.**
4. **Punctuation / symbol removal** — apostrophes inside a word (`det's`) are preserved; all other punctuation and symbols are removed.
5. **Whitespace collapse.**
6. **Numerals → words** — every standalone integer token is expanded to its Danish cardinal words via `num2words` (`4` → `fire`, `24` → `fireogtyve`), so digit-vs-word formatting (`"4"` vs `"fire"`) is not counted as an error. Only standalone integers are converted; digits embedded in larger tokens (decades like `1960'erne`, ranges like `1-3`) are left untouched. Ordinals (`3.` → `tredje`) and symbol/unit expansion (`%` → `procent`) were tested and rejected as net-neutral-to-harmful.

An optional filler-word strip (`øh`, `hmm`, …) is available in the harness but **off** by default, since its effect concentrates on spontaneous-speech sets and can shift that column's relative order.

Danish orthographic variants (`aa`↔`å`, `oe`↔`ø`, `ae`↔`æ`) are **not** normalised — the digraphs occur legitimately as letter sequences. Because the normaliser is parameterised, [`scripts/rescore.py`](https://github.com/Rye-A1/danish-asr-leaderboard) can re-derive WER/CER from the saved raw outputs under any configuration without re-running inference.

## Adding a model

There are two paths, depending on whether you have run the evaluation yourself:

- **Request a model (we run it):** open a [GitHub issue](https://github.com/Rye-A1/danish-asr-leaderboard/issues) with the model id, backend, and where to find it — we'll run it through the harness and add it.
- **Submit a score (you ran it):** run the harness from the [GitHub repo](https://github.com/Rye-A1/danish-asr-leaderboard) and open a pull request with `results/<model-slug>.json` plus the raw `outputs/<model-slug>/` transcriptions. On merge, CI publishes both here and updates the leaderboard automatically.

Whichever path a model arrives by, we re-evaluate it independently on our own hardware before publishing — to confirm the scores reproduce and catch configuration differences. Do **not** modify the normalisation or metrics; run the harness as-is so results stay comparable.

## License

MIT — see [LICENSE](https://github.com/Rye-A1/danish-asr-leaderboard/blob/main/LICENSE).
