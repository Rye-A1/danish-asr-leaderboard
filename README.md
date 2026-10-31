# Danish Speech-to-Text Leaderboard

An open, reproducible benchmark for **Danish speech-to-text**: every model is
transcribed and scored the same way on the same five public Danish test sets, so
WER, CER and speed compare directly. Open-weight Danish speech recognition models
and hosted transcription APIs are both included.

*Åben benchmark for dansk tale til tekst — sammenlign hvor præcist danske
talegenkendelsesmodeller omsætter tale til tekst.*

![Danish ASR Leaderboard cover](https://huggingface.co/spaces/RyeAI/danish-asr-leaderboard/resolve/main/cover.jpeg)

Reproducible benchmark and open leaderboard for **Danish automatic speech
recognition**, scored across five independent public test sets.

- **Leaderboard (Space):** https://huggingface.co/spaces/RyeAI/danish-asr-leaderboard
- **Results (dataset):** https://huggingface.co/datasets/RyeAI/danish-asr-leaderboard

One command evaluates a model on every test set, writes a result JSON, and the
push script publishes it to the leaderboard. The same harness covers local
models (transformers, NeMo, faster-whisper, …) and hosted APIs (ElevenLabs,
Azure OpenAI, Google Chirp, Soniox).

## Test sets

The leaderboard reports a per-dataset and a macro-averaged score over five
**core** test sets:

| Column | Dataset | Split | Domain |
|--------|---------|-------|--------|
| `coral_conversation` | [CoRal-project/coral-v3](https://huggingface.co/datasets/CoRal-project/coral-v3) — conversation | test | Spontaneous conversation |
| `coral_read_aloud` | [CoRal-project/coral-v3](https://huggingface.co/datasets/CoRal-project/coral-v3) — read_aloud | test | Read-aloud speech |
| `cv17_da` | [Common Voice 27.0](https://mozilladatacollective.com/datasets/cmu5unqiw008jnq079btgdfgm) — da, 2,756 clips (the CV 25.0 test split) | test | Crowd-sourced read speech |
| `fleurs_da` | [google/fleurs](https://huggingface.co/datasets/google/fleurs) — da_dk | test | Read speech |
| `ftspeech` | [alexandrainst/ftspeech](https://huggingface.co/datasets/alexandrainst/ftspeech) | test_balanced | Parliamentary / broadcast |

## Installation

System dependency (audio decoding):

```bash
apt install ffmpeg     # macOS: brew install ffmpeg
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already, then install the core library and whichever backend(s) you need:

```bash
uv pip install -e ".[transformers]"    # Whisper, Røst, hviske, …
uv pip install -e ".[transformers-remote]"  # models shipping their own code (trust_remote_code)
uv pip install -e ".[nemo]"            # Canary / Parakeet / SALM
uv pip install -e ".[nemotron-asr]"    # Nemotron 3.5 ASR (transformers, no NeMo)
uv pip install -e ".[nemotron-transcribe]"  # Nemotron 3.5 Transcribe
uv pip install -e ".[faster-whisper]"
uv pip install -e ".[qwen-asr]"
uv pip install -e ".[saga2]"           # capacit-ai/saga-2-*
uv pip install -e ".[brage]"           # Harmonium/brage-v1
uv pip install -e ".[voxtral]"
uv pip install -e ".[elevenlabs]"      # API backends
uv pip install -e ".[azure-openai]"
uv pip install -e ".[google-chirp]"
uv pip install -e ".[soniox]"
uv pip install -e ".[mistral]"        # hosted Voxtral Transcribe 3
```

Available extras match the `--backend` names: `transformers`, `transformers-remote`, `wav2vec2`,
`faster-whisper`, `qwen-asr`, `saga2`, `nemo`, `nemotron-asr`, `nemotron-transcribe`, `voxtral`, `danstral`, `seamless`, `cohere-asr`,
`vibevoice`, `brage`, `elevenlabs`, `azure-openai`, `google-chirp`, `soniox`, `syv`, `mistral`.

> **NeMo note:** install `nemo_toolkit[asr]` *first* to avoid
> dependency-resolver conflicts: `uv pip install "nemo_toolkit[asr]"` then
> `uv pip install -e ".[nemo]"`.
>
> On Python ≥3.10 the resolver may backtrack to an ancient `numba` (0.53.1) whose
> `llvmlite` (0.36) refuses to build ("only versions >=3.6,<3.10 are supported").
> Pin modern versions explicitly:
> `uv pip install -e ".[nemo]" "numba==0.64.0" "llvmlite==0.46.0"`.

> **Common Voice note:** the `cv17_da` column is scored on 2,756 fixed clips:
> the Danish test split of **Common Voice 25.0**, not 17 (the key is kept for
> compatibility). Mozilla no longer serves 25.0 and forbids re-hosting, but
> Common Voice 27.0's test split contains the same clips byte-identical, plus 5
> newer ones that the fetch script drops. Accept the terms on the
> [dataset page](https://mozilladatacollective.com/datasets/cmu5unqiw008jnq079btgdfgm), then:
> ```bash
> export MOZILLA_API_KEY=...        # mozilladatacollective.com
> python scripts/fetch_common_voice_da.py --output-dir cv_da
> export CV_DATA_DIR=$PWD/cv_da     # load_common_voice reads cv_da/test/test_manifest.jsonl
> ```
> A full run refuses any other Common Voice set (row count and sentence fingerprint).

Log in for pushing results (read access is anonymous):

```bash
huggingface-cli login   # RyeAI org write token
```

## Running an evaluation

```bash
danish-asr-eval --model <model> --backend <backend> [options]
# equivalently: python run_eval.py --model ... --backend ...
```

Examples:

```bash
# Whisper / transformers (also Røst, hviske, and other seq2seq HF models)
danish-asr-eval --model openai/whisper-large-v3 --backend transformers

# NeMo Canary (HF or local .nemo)
danish-asr-eval --model nvidia/canary-1b-v2 --backend nemo --nemo-model-type canary
danish-asr-eval --model /path/to/best.nemo --model-id RyeAI/canary-1b-v2-da \
  --backend nemo --nemo-model-type canary --params-b 1.0

# NeMo Parakeet
danish-asr-eval --model nvidia/parakeet-tdt-0.6b-v3 --backend nemo --nemo-model-type parakeet

# Nemotron 3.5 ASR (Danish language prompt, native transformers)
danish-asr-eval --model nvidia/nemotron-3.5-asr-streaming-0.6b --backend nemotron-asr

# Nemotron 3.5 Transcribe (BF16, requires an 80 GB GPU)
danish-asr-eval --model nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B \
  --backend nemotron-transcribe --batch-size 4 --params-b 33.698 \
  --model-revision c17667641642c4265ff2b7037dd101b7d4567634

# Qwen3-ASR / fine-tunes
danish-asr-eval --model Qwen/Qwen3-ASR-1.7B --backend qwen-asr

# Open-weight Voxtral (local inference)
danish-asr-eval --model mistralai/Voxtral-Mini-3B-2507 --backend voxtral

# Saga 2 (inference package is loaded from the model repo itself)
danish-asr-eval --model capacit-ai/saga-2-m --backend saga2

# Brage (runs the decoding module shipped in the gated model repo: accept its terms on the Hub
# and run `hf auth login` first; the 14.5 GB neural LM it rescores with is downloaded on
# first use). Needs about 15 GiB of free GPU memory and 22 GB of RAM: if its "brage
# settings" line lists batch size 1, the run fell back to one clip at a time and is not the
# reference decode. The full inference system is 8.785B neural parameters:
# 1.543B in Brage plus 7.242B in the Munin rescoring model.
danish-asr-eval --model Harmonium/brage-v1 --backend brage --params-b 8.785

# Models whose decoding code ships in their own repo (trust_remote_code). This executes model-repo
# Python, so use a reviewed commit SHA and run without unnecessary credentials. The repo's config
# picks the dtype; the model must expose model.transcribe(processor=, language=, audio_arrays=, sample_rates=).
danish-asr-eval --model danish-foundation-models/edda-v0.2-duo --backend transformers-remote \
  --model-revision fb0de9b0fd76386b2f475dee12eb65da25994060

# API backends (params not applicable → defaults to 0.0)
danish-asr-eval --model chirp_3 --backend google-chirp --google-cloud-project my-gcp-project
danish-asr-eval --model soniox-v1 --backend soniox --soniox-api-key "$SONIOX_API_KEY"

# Hosted Voxtral 3: export MISTRAL_API_KEY privately in the shell first.
# The cache is required; an interrupted run reuses completed transcripts.
danish-asr-eval --model voxtral-mini-transcribe-3-0 --backend mistral \
  --mistral-cache eval_audio_cache/mistral-mini-3.jsonl \
  --max-samples 20 --out-dir eval_audio_cache/mistral-pilot-results \
  --outputs-dir eval_audio_cache/mistral-pilot-outputs

# Quick smoke test (cap samples per dataset)
danish-asr-eval --model openai/whisper-large-v3 --backend transformers --max-samples 100

# Subset of test sets
danish-asr-eval --model openai/whisper-large-v3 --backend transformers \
  --datasets cv17,fleurs
```

To run the whole board in one go (one process per model, GPU freed between each):

```bash
bash scripts/run_benchmark.sh                 # all open-weight models
MAX_SAMPLES=50 bash scripts/run_benchmark.sh  # quick smoke
RUN_API=1 bash scripts/run_benchmark.sh       # also hosted APIs (needs keys)
```

The sweep posts progress to a webhook if `MATTERMOST_WEBHOOK_URL` is set (in the
environment or `.env`). Mattermost and Slack share the `{"text": …}` webhook
payload, so either works out of the box; other services need a one-line tweak in
`danish_asr_leaderboard/notify.py`. Copy [.env.example](.env.example) to
`.env` to configure this and the other optional integrations (Mozilla CV key,
HF token). Notifications are a silent
no-op when unconfigured.

Run `danish-asr-eval --help` for all options (device, batch size, beam/KenLM,
per-API credentials, `--access open|proprietary`, …). Available backends:

`transformers`, `transformers-remote`, `wav2vec2`, `faster-whisper`, `qwen-asr`, `saga2`, `nemo`, `nemo-salm`, `nemotron-asr`, `nemotron-transcribe`,
`voxtral`, `danstral`, `seamless`, `cohere-asr`, `vibevoice`, `brage`, `elevenlabs`,
`azure-openai`, `google-chirp`, `soniox`, `syv`, `mistral`.

Each run writes `results/<model-slug>.json` with per-dataset WER/CER, the core
means, speed, and metadata. It also persists the **raw, un-normalised** per-sample
model output under `outputs/<model-slug>/<dataset>.jsonl` (one `{id, reference,
hypothesis}` per line, plus a `meta.json`). Because the raw output is saved, any
normalisation change can be evaluated **offline** — see *Re-scoring* below — without
re-running the model. Disable with `--outputs-dir ""`.

### Private datasets

Put the private audio and a UTF-8 `test.csv` with `audio_path` and `reference_text`
columns in `private-data/`, which Git ignores, or outside the repository. Restrict
access to that input directory (for example, `chmod 700 private-data` on Unix).
Audio paths are relative to the CSV. Register it in `custom_datasets.py` in the
directory where you run the command:

```python
from danish_asr_leaderboard import DatasetConfig

MY_DATASET = DatasetConfig(name="my-dataset", source={"test": "private-data/test.csv"})
```

```bash
danish-asr-eval --model openai/whisper-small --backend transformers --dataset my-dataset
```

The CSV and audio are checked before loading the model. Scores and transcripts
are saved under `results/private/` and `outputs/private/`, separate from public
leaderboard results. Generated private directories and files are owner-only on
Unix; check access controls on other systems. If using `--syv-cache` or
`--mistral-cache`, place the cache in `private-data/` or outside the repository.
The same applies to custom output directories passed through command-line options.
A failed transcription stops the run without writing a score. Hosted backends
send the audio to their provider; use them only when that is appropriate for the data.

## Re-scoring (offline normalisation experiments)

The normaliser is parameterised, and `scripts/rescore.py` recomputes WER/CER
from saved raw outputs under any configuration — no GPU or re-inference needed:

```bash
# Re-score every model under NFKC and diff mean_wer against the published results
python scripts/rescore.py --unicode-form NFKC --compare results

# Re-score one model into a separate dir
python scripts/rescore.py --model openai/whisper-large-v3 \
  --unicode-form NFKC --out-dir results_nfkc
```

The published default is **NFKC**, spoken-number folding, number-word
normalisation, and filler removal. `scripts/rescore.py` exposes the corresponding
`--no-*` options for comparison experiments; published scores must use the shared
defaults.

## Publishing to the leaderboard

```bash
python scripts/push_results.py
```

Pulls existing results from the HF dataset, merges your local JSONs (local wins
on conflict), rebuilds `data/results.parquet`, and uploads everything. Safe on a
fresh clone — nothing is lost.

To also publish the **raw per-sample outputs** (so others can re-score under their
own normalisation), roll them into one Parquet per model and push under the
`outputs/` prefix of the same dataset repo:

```bash
python scripts/push_outputs.py                 # all models under outputs/
python scripts/push_outputs.py --model openai/whisper-large-v3   # just one
```

Each file is `outputs/<model-slug>.parquet` with columns
`dataset, id, reference, hypothesis` — incremental, so re-running one model only
re-uploads that model. Use `--no-upload` to build the Parquet locally without pushing.

To redeploy the static Space after editing `huggingface/space/index.html`:

```bash
HF_TOKEN=hf_... python scripts/update_space.py
```

## Methodology

### Text normalisation
The same transform is applied to references and hypotheses:

1. Unicode NFKC compatibility folding; optional acute accents (`dér` / `der`) are ignored.
2. Exact historical and joined spellings (`saa` / `så`; `idag`, `igår`, `ialt`, `iøvrigt`, `igang`, `afsted`, `tilstede` and their spaced forms) are unified. Attested abbreviations (`bl.a.`, `ca`, `dvs.`, `evt.`, `f.eks.`, `fx`, `inkl.`, `ekskl.`, `m.fl.`, `m.m.`, `mv.`, `osv.`, `pct.`) and units (`km`, `cm`, `kg`, `kr`, numeric `m`, `mm`, `m²`, `km/h`, `km/t`, `m/s`) match their spoken forms. Numeric `mio.` and `mia.` match the corresponding number of millions or billions.
3. Thousands separators are removed (`1.234` → `1234`); decimals retain a marker (`3,14` / `3.14` → `3komma14`). Trailing decimal zeroes, frequency-unit spacing (including `2.400 GHz` → `2.4 GHz`), `%` / `procent`, clock times (`kl. 06:30` = `klokken 6.30`), and ratios (`3:2` = `3 til 2`) are normalized. Numeric slashes separate their operands instead of joining them.
4. Lowercasing and punctuation/symbol removal, retaining apostrophes within words.
5. Spoken Danish numerals → digits, then standalone integers → canonical Danish words.
6. Danish hesitation filler removal and whitespace collapse.

This treats digit-versus-word formatting as equivalent. General `aa` / `å`
substitution is avoided; only a small explicit list of historical words is folded. Bare `2.400` still means
`2400`; the decimal reading needs GHz context. Slashes are not assumed to be
fractions or dates, and ordinals remain literal because their spoken forms are
ambiguous. Bare `mm` stays literal because it can be a hesitation sound; dotted
`m.m.` is handled separately as `med mere`.

### Metrics
Corpus-level **WER** and **CER** (%), lower is better, computed with RapidFuzz
Levenshtein distance:

```
WER = (substitutions + deletions + insertions) / reference_words   × 100
CER = (char sub + del + ins)                   / reference_chars   × 100
```

`mean_wer` / `mean_cer` are macro-averages across the five core test sets.
References that normalise to empty are dropped from scoring.

### Speed
`speed_x` = total audio duration / transcription time. Model load is excluded;
hardware, runtime, batching, and API latency all affect it, so it is context
rather than a pure model-quality metric.

## Repository layout

```
danish_asr_leaderboard/
  cli.py            # argument parsing + evaluation driver
  datasets.py       # test-set loaders + registry
  normalizer/       # Danish text normalisation
  metrics.py        # WER / CER
  scoring.py        # transcribe + time a dataset (returns raw outputs too)
  results.py        # EvalResult, slug, params lookup, JSON writer
  raw_outputs.py    # persist/load raw per-sample outputs for offline re-scoring
  audio.py          # ffmpeg transcode + duration helpers
  backends/         # one module per backend, self-registered via @register
    base.py         # Backend ABC + LoadOptions + registry
    api/            # hosted-API backends
run_eval.py         # thin CLI entry point
scripts/            # run_benchmark.sh, fetch_common_voice_da.py, push_results.py, push_outputs.py, update_space.py, rescore.py
  model_metadata.json # hand-maintained per-model metadata: release dates, openness
outputs/            # raw per-sample model outputs (git-ignored)
results/            # generated result JSONs (git-ignored)
huggingface/
  space/            # Static HTML leaderboard (deployed to the HF Space)
  dataset_card.md   # Source for the published HF dataset README
  hf_downloads.json # Fallback seed for Hub download history
```

## Adding a backend

Create `danish_asr_leaderboard/backends/<name>_backend.py`, subclass `Backend`,
implement `transcribe_one` (and optionally `transcribe_batch`), and decorate the
loader with `@register("<name>")`. Import it from `backends/__init__.py` so it
registers on package import, add a `<name>` extra to `pyproject.toml`
`[project.optional-dependencies]`, and it becomes a valid `--backend` choice automatically.

## Contributing

Want your model on the leaderboard? Run the eval and open a PR with the result
JSON, or open a [discussion](https://huggingface.co/spaces/RyeAI/danish-asr-leaderboard/discussions)
on the Space. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE).
