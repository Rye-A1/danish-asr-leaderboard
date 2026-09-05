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

Benchmark results backing the **[Open Danish ASR Leaderboard](https://huggingface.co/spaces/RyeAI/danish-asr-leaderboard)** — an open, reproducible comparison of **Danish speech-to-text** models.

Every model is transcribed and scored identically on the same five independent public Danish test sets, so the numbers compare directly: Word Error Rate (WER) and Character Error Rate (CER) — lower is better — plus speed. Open-weight Danish speech recognition models you can run yourself and hosted transcription APIs are both included.

## Dansk tale-til-tekst benchmark

**Åben benchmark for dansk tale til tekst.** Sammenlign hvor præcist forskellige modeller til dansk talegenkendelse omsætter tale til tekst.

Alle modeller måles ens på de samme fem offentlige danske testsæt, så tallene kan sammenlignes direkte: ordfejlrate (WER) og tegnfejlrate (CER) — lavere er bedre — samt hastighed. Både åbne modeller, du selv kan køre lokalt, og kommercielle tale-til-tekst-API'er indgår.

Den aktuelle rangliste findes på [leaderboardet](https://huggingface.co/spaces/RyeAI/danish-asr-leaderboard).

## Using this dataset

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
| `speed_x` | float\|null | Audio seconds / wall-clock second (higher = faster). Measured on one NVIDIA A100 80&nbsp;GB at batch size 16; network-bound for API models. `NaN` if not measured. |
| `submitted` | string | ISO 8601 date the result was submitted (`YYYY-MM-DD`) |

Per-model configs (`outputs/<model-slug>`) — one row per utterance: `dataset`, `id`, `reference`, `hypothesis` (raw, un-normalised).

## Text normalisation

Applied identically to hypothesis and reference before scoring, so WER/CER reflect recognition errors rather than formatting:

1. **Unicode NFKC** — compatibility composition (folds ligatures, full-width digits, `²`→`2`, …). A near-no-op on Danish speech text, adopted for correctness and consistency with the Danish standard.
2. **Danish number canonicalisation** — separators within a numeral are stripped (`1.234` → `1234`, `3,14` → `314`).
3. **Lowercase.**
4. **Punctuation / symbol removal** — apostrophes inside a word (`det's`) are preserved; all other punctuation and symbols are removed.
5. **Spelled-out numerals → digits** — words are folded to their digit form with `text2num`, so spelling and spacing variants collapse onto one value before step 6 expands them back (`otte og tredive` and `otteogtredive` → `38`; `hundrede` → `100`). Bare `en` / `et` are left as indefinite articles.
6. **Numerals → words** — every standalone integer token is expanded to its Danish cardinal words via `num2words` (`4` → `fire`, `24` → `fireogtyve`), so digit-vs-word formatting (`"4"` vs `"fire"`) is not counted as an error. Only standalone integers are converted, so decades such as `1960'erne` keep their digits. Standalone symbol/unit expansion (`%` → `procent`) was tested and rejected as net-neutral-to-harmful.
7. **Whitespace collapse.**

A dash between two digits becomes a space in step 4 rather than being deleted, so `1-3` scores as two numbers. Deleting it glued the range into `13` → `tretten`, and the time range `10 00-11 00` into `ti elleve nul` — references no correct transcription could match, which also handed free matches to models that emit digits while penalising ones that spell the range out.

**Known limitation.** Colons and slashes are not yet treated the same way: `10:00` still collapses to `ettusind` and `4/5` to `femogfyrre`. Both need their own reading rule — a slash may be a fraction, a date is spoken differently again — so they are tracked separately rather than folded into the dash rule. Ordinals written with a full stop are also left alone (`3. plads` → `tre plads`); converting them to `tredje` was measured and rejected as net-harmful, because most `N.` in these references are sentence-final cardinals rather than true ordinals.

An optional filler-word strip (`øh`, `hmm`, …) is available in the harness but **off** by default, since its effect concentrates on spontaneous-speech sets and can shift that column's relative order.

Danish orthographic variants (`aa`↔`å`, `oe`↔`ø`, `ae`↔`æ`) are **not** normalised — the digraphs occur legitimately as letter sequences. Because the normaliser is parameterised, [`scripts/rescore.py`](https://github.com/Rye-A1/danish-asr-leaderboard) can re-derive WER/CER from the saved raw outputs under any configuration without re-running inference.

## Adding a model

There are two paths, depending on whether you have run the evaluation yourself:

- **Request a model (we run it):** open a [GitHub issue](https://github.com/Rye-A1/danish-asr-leaderboard/issues) with the model id, backend, and where to find it — we'll run it through the harness and add it.
- **Submit a score (you ran it):** run the harness from the [GitHub repo](https://github.com/Rye-A1/danish-asr-leaderboard) and open a pull request with `results/<model-slug>.json` plus the raw `outputs/<model-slug>/` transcriptions. On merge, CI publishes both here and updates the leaderboard automatically.

Whichever path a model arrives by, we re-evaluate it independently on our own hardware before publishing — to confirm the scores reproduce and catch configuration differences. Do **not** modify the normalisation or metrics; run the harness as-is so results stay comparable.

## License

MIT — see [LICENSE](https://github.com/Rye-A1/danish-asr-leaderboard/blob/main/LICENSE).
