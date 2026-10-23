# Add Soniox stt-async-v5 Danish ASR results

Adds a proprietary hosted System entry for Soniox `stt-async-v5`, evaluated on
all five Danish test sets: 26,780 independent clips, 41.8868 audio hours.
Includes original per-clip reference/hypothesis outputs and model metadata.

| Test set | Clips | WER (%) | CER (%) |
| --- | ---: | ---: | ---: |
| FLEURS | 930 | 6.40 | 2.50 |
| CoRal conversation | 8,438 | 32.92 | 21.25 |
| CoRal read-aloud | 9,122 | 21.82 | 9.04 |
| FTSpeech test_balanced | 5,534 | 11.26 | 6.03 |
| Common Voice fixed Rye subset | 2,756 | 8.65 | 3.09 |
| Five-set mean | | 16.21 | 8.38 |

Inference ran on 2026-10-23 through a custom resumable REST client using the
pinned Rye harness `dff9607fe1a84a7f1a64a0bb2ec75e7885568ebc` for data/scoring.
Full audio/reference fingerprints match the original benchmark. The final
pipeline invocation from the reproduction project root was:

```powershell
.venv\Scripts\python.exe -X utf8 -m evaluation.soniox_run --upgrade-client --workers 32
```

Earlier saved clips are retained; execution history and original source/config
archives record the concurrency and transport changes. Every clip is a separate
transcription request. Settings: US endpoint `https://api.soniox.com/v1`,
`model=stt-async-v5`, `language_hints=["da"]`, strict=false, language
identification=false, diarization=false, no context and no translation. The
client shares an initial 12 HTTP requests/second ceiling with coordinated 429
backoff. Hosted decoding, hardware and immutable weight revision are unavailable.

The original pinned audit gives 16.48% mean WER / 8.53% mean CER. This PR's scores
were recomputed from those same unnormalized outputs using the **unmodified**
current upstream normalizer/metrics at
`1de3f8ad2724c363c1c93349f48337792d15bf12`:

```powershell
.venv\Scripts\python.exe -X utf8 .cache/soniox-upstream-review/scripts/rescore.py --outputs-dir .cache/soniox-pr-submission/outputs --model soniox-stt-async-v5 --out-dir .cache/soniox-pr-submission/results
```

Both scoring revisions and their provenance are retained. No new inference was
performed for rescoring. Output IDs match upstream loader basenames; original
references/hypotheses and all 26,780 clips are preserved. The Common Voice column
contains the fixed Rye CV25 subset sourced from CV27, not all CV27 test clips.

The local API client ran on Windows, Python 3.11.7; no local GPU inference.
`speed_x` is null because saved timings sum concurrent per-clip network/queue
latencies and do not represent comparable batch wall time. `params_b=0.0` is the
existing API sentinel; the parameter count is unknown.

Validation: the independent full-run audit verified full coverage, data/audio
fingerprints, raw-response/transcript consistency, API settings, source/config
hashes, and recomputed original scores. Submission preparation checks prediction
hashes and unchanged text, then runs upstream offline rescoring. Credentials,
audio and raw remote resource IDs are excluded. Maintainer reproduction remains
required before publication under CONTRIBUTING.md.
