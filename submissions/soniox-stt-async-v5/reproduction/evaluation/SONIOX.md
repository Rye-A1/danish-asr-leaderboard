# Soniox Danish evaluation

This runner sends the same pinned benchmark WAV clips to Soniox's async REST
API and scores Danish transcripts with the existing pinned RyeAI normalizer and
corpus WER/CER functions. It does not send references, prompts or translation
requests. Danish is a language hint (`da`), not a hard language restriction.
No GPU or additional Python dependencies are needed beyond the project setup.

## Start with four clips

For the completed run and upstream PR format, see
[submission readiness](SONIOX_SUBMISSION.md) and the
[draft PR description](SONIOX_PR_DESCRIPTION.md). The upstream submission uses
separately recorded current-harness rescoring; original audited scores stay intact.

Create a project API key at <https://console.soniox.com>. Set `SONIOX_API_KEY`
in your local environment; never put it in a tracked file or chat. In PowerShell
you can enter it without displaying it or writing it into command history:

```powershell
$env:SONIOX_API_KEY = [System.Net.NetworkCredential]::new('', (Read-Host 'Soniox API key' -AsSecureString)).Password
uv run python -X utf8 -m evaluation.soniox_evaluation
```

The default is **four FLEURS Danish test clips**, clearly marked as a smoke test.
The runner processes up to eight independent clips concurrently (`--workers 8`),
using a rolling window and a separate reusable HTTPS session per worker. Each
benchmark clip remains an independent transcription job. Soniox does not expose
the server's GPU batch size through this async request.
Once your key is available in the process environment, the command makes paid
API requests. It does not launch a full benchmark automatically. For a project
configured in Soniox's EU region, append `--region eu`; the default region is US.

Results are under `evaluation/results/soniox-stt-async-v5-us/smoke-4/` (or `eu`).
Read `summary.json` for WER/CER. Repeat the command to resume; completed clips
and known remote job IDs are reused. Stop on errors; failed clips never become
empty successful transcripts. Do not run two processes in the same output folder.

## Full benchmark (explicit opt-in)

After checking the smoke results and dataset terms for external processing:

```powershell
uv run python -X utf8 -m evaluation.soniox_evaluation --max-samples 0 --datasets fleurs coral_conversation coral_read_aloud ftspeech cv17
```

The existing gated CoRal and fixed Common Voice preparation requirements in the
root README apply. `--cache`, `--cv-dir` and `--output` work as in the local
evaluator. A full run uses a separate `full/` folder. Only all five complete test
sets produce a five-set mean. Smoke results never enter the full-run overview.
The OWSM-only baseline audit/export command (`evaluation.report`) is unchanged;
Soniox results appear in their own summary and the completed evaluations overview.

## Provenance and recovery

`config.json` records the model ID, endpoint, request settings, normalization,
harness revision, environment and source hashes. Each `.data.json` records dataset
revisions and fingerprints. Each `.jsonl` stores references and predictions.
`jobs/DATASET/INDEX.json` stores remote IDs, raw status/transcript responses,
timestamps and cleanup state. All these files stay in the ignored results folder.
Successful raw transcripts are saved locally before their remote job and uploaded
file are deleted. Cleanup errors stop the run; rerunning retries cleanup.
Failed/timed-out jobs stay remote so they can be inspected/resumed.

Soniox does not expose a weight hash through this adapter. `revision` is therefore
explicitly null: a model name is not a guarantee of an immutable hosted backend.
The runner records evaluation dates and retains raw responses for auditing.
API timing includes network transfers and queue/poll waits, and is not comparable
to GPU inference speed. With multiple workers it sums individual clip latencies;
it is not the total elapsed duration of the run. Saved polling time accumulates across attempts;
interrupted network calls and time between attempts are excluded.

Rate-limit rejections (HTTP 429) and transient GET/DELETE failures are retried with
bounded backoff. All workers share an initial ceiling of 12 HTTP requests/second.
A 429 pauses them together for at least a minute and lowers that ceiling. Rate
events are printed to the run log; increasing workers does not bypass this ceiling.
Ambiguous POST/network failures are never automatically retried.
If a POST fails or is interrupted before
its returned ID is saved, `pending_request` blocks further submissions to avoid
duplicate charges. Inspect the project's Soniox Console and the saved job file.
If a resource was created, fill in its `file_id` or `transcription_id` and remove
`pending_request`. Remove the marker alone only after confirming the request did
not create a resource. Preserve the rest of the job file. Known job failures
require investigation rather than automatic resubmission. Never delete a file
belonging to a still-processing transcription.

Full runs validate every dataset fingerprint against `evaluation/benchmark.json`
before submitting that set's audio. The run's `source/` snapshot must contain the
four files listed in `config.json`'s `code_sha256` for the final audit. The current
evaluation also saves the model catalog and concurrency-limit response in
`api-preflight.json` (no credentials).

After all five test sets finish, independently verify predictions against raw
responses, job/audio hashes, original references, settings, scores and the mean:

```powershell
uv run python -X utf8 -m evaluation.soniox_report
```

This writes `audit.json` and `audit.md` in the run folder, then exports
`evaluation/soniox-benchmark.json` and `evaluation/SONIOX_RESULTS.md` only when
the whole benchmark passes. `--allow-partial` audits only fully completed sets,
leaving the mean null and writing no full benchmark export. API transcripts
and audio stay local. The OWSM baseline export is preserved.

For a full run with automatic source/API snapshots, final audit and export:

```powershell
uv run python -X utf8 -m evaluation.soniox_run --workers 32
```

`run-status.json` records the current phase and any failure. Changing only the
worker count archives the original configuration and records the change in
`execution-history.json`; recognition settings must remain identical. Resume
only after the previous process has stopped. A recorded failure must be resolved
before retrying. Model IDs are not immutable weight revisions.

`--upgrade-client` archives the previous client source, configuration, saved
transcript prefixes and job hashes before a transport/scheduling update. It
requires unchanged data preparation, scoring, model settings and baseline pins.
This is used to preserve the current run's initial connection/scheduling versions;
existing predictions and known remote job IDs are reused.

Offline validation (no account, downloads or paid requests):

```powershell
uv run python -m unittest evaluation.test_soniox
uv run ruff check evaluation/soniox.py evaluation/soniox_evaluation.py evaluation/test_soniox.py
```

API references: [async transcription](https://soniox.com/docs/stt/async/async-transcription),
[request fields](https://soniox.com/docs/api-reference/stt/transcriptions/create_transcription),
[models](https://soniox.com/docs/stt/models).
