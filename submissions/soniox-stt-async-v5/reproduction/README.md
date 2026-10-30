# Danish Open Whisper

Open research into Danish speech recognition. The first experiment evaluates
all four OWSM v4 checkpoints using RyeAI's Danish benchmark normalization and metrics.
A [minimal training wrapper](training/README.md) uses ESPnet with YAML configs
and W&B logging for all four OWSM checkpoints.
Define dataset, training and evaluation settings in an [experiment folder](experiments/README.md).
Select it in [experiments/active.yaml](experiments/active.yaml) and run [run.py](run.py).
Its `experiment.md`, `config.yaml`, `results.json` and `results.md` stay together.
The current [corrected Nordic experiments](experiments/danish-nordic-balanced-small-v2/experiment.md)
retain all Danish and add source-balanced Swedish/Norwegian data, including
RixVox v2, NPSC 2.0 and NorGovPCC. Their order lives in `experiments/queue.yaml`.
The [recorded workflow check](research/workflow-smoke.md) passed dataset mixing, GPU
training and evaluation of the exported model on all five test domains.

The first full fine-tuning run is defined in
[danish-small-1epoch](experiments/danish-small-1epoch/experiment.md): all accepted
labelled Danish training clips, one OWSM small epoch on three GPUs, complete
Rye evaluation and public Hugging Face upload.
The sources follow [the training data mix](data/README.md),
supported by a [32-model leaderboard data audit](research/leaderboard-data.md)
and [research on Scandinavian ASR transfer](research/scandinavian-transfer.md).
The [filtering and mixing literature review](research/data-methods.md) supports
a [small data-selection experiment plan](data/filtering-mixing.md).
A [dataset preparation pipeline](training/datasets/README.md) provides basic
filtering and duration-weighted mixing. Training prepares the configured mix
automatically; [prepare_data.py](prepare_data.py) runs just that stage.
Full-corpus downloads and preparation are resumable and configured in the
experiment. Embedding selection has not run yet.

See [Danish benchmark results](evaluation/RESULTS.md) for all 20 completed and
audited runs. [Machine-readable results](evaluation/benchmark.json) include
full-precision scores, pinned revisions and audit provenance.

The completed [Soniox API evaluation](evaluation/SONIOX_RESULTS.md) has a separate
[leaderboard submission checklist](evaluation/SONIOX_SUBMISSION.md) and
[draft PR description](evaluation/SONIOX_PR_DESCRIPTION.md), including current
upstream rescoring of its saved predictions.

Repository code is MIT licensed. Models, datasets and upstream dependencies
retain their own licenses; the MIT license does not relicense them.

## Setup

Python 3.11, [uv](https://docs.astral.sh/uv/), Git and FFmpeg are required.
The lockfile targets PyTorch 2.9.1 with CUDA 12.8 on Linux (including B200)
and CUDA 12.6 on Windows. [Multi-GPU training](training/README.md#multiple-gpus)
uses one Linux host and is selected in the experiment config.
CPU inference is available with `--device cpu`, but will be slow.

```sh
uv sync --locked
git clone https://github.com/Rye-A1/danish-asr-leaderboard .cache/rye
git -C .cache/rye checkout dff9607fe1a84a7f1a64a0bb2ec75e7885568ebc
```

The same Rye revision is installed as a library by `uv sync`. The clone gives
you its Common Voice preparation script and the original harness to inspect.
Model and audio downloads live in `.cache/`; they are excluded from Git.
If large Hugging Face downloads stall on Windows, retry with the environment
variables `HF_HUB_DISABLE_XET=1` and `HF_HUB_DOWNLOAD_TIMEOUT=120`.

## Test data

The five test sets are CoRal-v3 conversation, CoRal-v3 read-aloud, Common Voice
Danish, FLEURS Danish and FTSpeech `test_balanced`. Exact revisions are recorded
in `evaluation/pins.json`. Test audio is converted using Rye's FFmpeg helper to
16 kHz mono WAV. Missing audio or unexpected full split sizes stop the run.
Only test Parquet files are downloaded, including for a capped smoke run;
training splits are never fetched by this evaluator.

For gated data, request access on the
[CoRal dataset page](https://huggingface.co/datasets/CoRal-project/coral-v3)
and sign in locally with `uv run huggingface-cli login`.

Download [Common Voice 27.0 Danish](https://mozilladatacollective.com/datasets/cmu5unqiw008jnq079btgdfgm)
through your Mozilla account, then prepare it with the pinned Rye script:

```sh
uv run python -X utf8 .cache/rye/scripts/fetch_common_voice_da.py --tarball PATH_TO_ARCHIVE --output-dir .cache/common_voice
```

Rye selects the **2,756 fixed CV25 test clips** from CV27. Its legacy column name
is `cv17_da`; the command selector is `cv17`. A full run validates the reference
fingerprint. Using all 2,761 CV27 test clips would be a different benchmark.
Keep downloaded data private; dataset licenses and access terms still apply.

## Evaluate

Check installation on four clips per model:

```sh
uv run python -X utf8 -m evaluation.evaluation --datasets fleurs --max-samples 4
```

Evaluate all four models on all five complete test sets:

```sh
uv run python -X utf8 -m evaluation.evaluation
```

Evaluate a single model, or an exported fine-tuned checkpoint:

```sh
uv run python -X utf8 -m evaluation.evaluation --models ctc --datasets fleurs cv17
uv run python -X utf8 -m evaluation.evaluation --models checkpoints/my-model --kind seq2seq
```

A local export contains `model.pth`, `config.yaml`, and the BPE/normalization
files referenced by that config (relative paths are resolved within the export).
It must use an architecture supported by ESPnet's S2T inference classes.
You can also supply a Hugging Face ESPnet export ID and `--revision COMMIT`.
The four known OWSM IDs use the same pins as their short aliases. For other
CTC exports, pass `--kind ctc` explicitly.

## Method and outputs

For hosted Soniox transcription on the same pinned test sets, see the
[Soniox adapter instructions](evaluation/SONIOX.md). Its separate command defaults
to a four-clip FLEURS smoke test and requires a local `SONIOX_API_KEY`.

- Language is forced to `<dan>`, task `<asr>`, with float32 inference and seed 0.
- Encoder-decoder models use greedy decoding by default (`--beam-size 1`), no
  CTC fusion, and `maxlenratio=1.0`. Recordings above 30 seconds use consecutive
  non-overlapping windows, joined with spaces. This is a baseline; boundaries
  can hurt recognition. Short clips use ESPnet's normal padding.
- CTC uses ESPnet's greedy buffered decoding with 4 seconds of context.
  `--batch-size 8` controls CTC batching; seq2seq currently decodes one clip at a time.
- Rye normalizes references and predictions with NFKC, number words and filler
  removal. WER/CER are corpus edit rates in percent, not mean per-clip rates.
  Rye excludes pairs whose reference becomes empty after normalization. Those
  clips are still transcribed; each score records `empty_normalized_references`.
- The five-set mean is the unweighted mean of the five corpus scores. It is
  only emitted after all five complete sets finish. Smoke runs never get it.

Results are saved to `evaluation/results/MODEL/{full,smoke-N}/`: `config.json`
records model hashes, decoding settings and environment; each `.data.json`
records dataset provenance and an audio/text fingerprint; `.jsonl` files hold
references and predictions; `summary.json` contains WER, CER and inference timing.
`evaluation/results/summary.md` updates after each completed full dataset with a
readable table across models. It excludes partial runs and smoke tests.
Timing excludes checkpoint loading and dataset preparation, and includes reading
cached audio during inference.

Repeat the same command to resume saved transcripts. A changed model, dataset or
setting is rejected: select a new `--output` directory for a new experiment.
Save raw transcripts locally so results can be audited and re-scored; review
dataset terms before publishing references or transcripts.

After all 20 baseline runs finish, audit the complete matrix and export the
full-precision numbers (including pins, settings and transcript hashes):

```sh
uv run python -X utf8 -m evaluation.report
```

This rechecks audio/reference fingerprints, re-scores every prediction and
verifies the five-set means before writing `evaluation/benchmark.json` and
replacing `evaluation/RESULTS.md` with the final tables and method. It rejects
incomplete runs or inconsistent summaries.

## Code map

`evaluation.py` runs experiments; `checkpoints.py` resolves model files;
`owsm.py` calls ESPnet; `data.py` prepares test audio; `results.py` scores and
stores outputs; `report.py` audits the final baseline. Handwritten files stay
below 300 lines. `uv.lock` is generated.

```sh
uv run ruff check .
```

Upstream: [RyeAI harness](https://github.com/Rye-A1/danish-asr-leaderboard),
[OWSM collection](https://huggingface.co/collections/pyf98/open-whisper-style-speech-models-owsm),
[ESPnet](https://github.com/espnet/espnet).
Model/data provenance must be audited separately before claiming a completely
open training pipeline; evaluating a checkpoint does not establish that claim.
