# Soniox leaderboard submission readiness

Checked 2026-10-23 against Rye-A1/danish-asr-leaderboard commit
`1de3f8ad2724c363c1c93349f48337792d15bf12`.
Source: [contribution instructions](https://github.com/Rye-A1/danish-asr-leaderboard/blob/1de3f8ad2724c363c1c93349f48337792d15bf12/CONTRIBUTING.md),
[result schema](https://github.com/Rye-A1/danish-asr-leaderboard/blob/1de3f8ad2724c363c1c93349f48337792d15bf12/danish_asr_leaderboard/results.py),
[raw output format](https://github.com/Rye-A1/danish-asr-leaderboard/blob/1de3f8ad2724c363c1c93349f48337792d15bf12/danish_asr_leaderboard/raw_outputs.py).

## What the PR needs

| Requirement | Prepared evidence |
| --- | --- |
| All five complete test sets | 26,780 predictions; 41.8868 audio hours; full independent audit |
| Leaderboard result JSON | `.cache/soniox-pr-submission/results/soniox-stt-async-v5.json` |
| Original, unnormalized reference/hypothesis pairs | Five JSONL files under bundle `outputs/soniox-stt-async-v5/` |
| Output metadata | `outputs/soniox-stt-async-v5/meta.json` |
| Model metadata entry | Bundle `model_metadata.entry.json`; merge its entry into upstream `scripts/model_metadata.json` |
| Exact commands, settings, dates, hardware | PR description, original audit, source/lockfile snapshots |
| Reproducible scoring | Unmodified current upstream `scripts/rescore.py`; scoring receipt and source hashes |
| Hosted revision limitation | Weight revision explicitly null; provider/model ID and request dates retained |
| Public-safe output | No audio, credentials, personal local paths, or remote job IDs in output JSONL |

This is a **proprietary System** entry, labelled `Hosted async API; Danish hint`.
The [provider changelog](https://soniox.com/docs/stt/models#june-11-2026) dates
`stt-async-v5` to 2026-06-11. Metadata uses `plot_by: submitted` for the hosted
service, null training-code/data/paper annotations, and the model-specific
changelog as documentation evidence. `params_b: 0.0` is upstream's API sentinel,
not a claim that the model has zero parameters. `speed_x` is null: our timings
sum concurrent per-clip request latency and cannot support a comparable speed.
The hosted server hardware is undisclosed; the client used Windows/Python 3.11.7
and 32 concurrent workers with no local GPU inference.

## Two scoring revisions, one set of predictions

Original experiment/audit files remain unchanged. They use Rye commit
`dff9607fe1a84a7f1a64a0bb2ec75e7885568ebc`, including its original normalization
and metric behavior. Current upstream has additional number, spelling,
abbreviation and punctuation normalization. We ran its unmodified rescore script
on the same raw predictions; this does not call Soniox or alter clip coverage.

| Dataset | Clips | Original WER / CER (%) | Current upstream WER / CER (%) |
| --- | ---: | ---: | ---: |
| FLEURS | 930 | 6.78 / 2.77 | 6.40 / 2.50 |
| CoRal conversation | 8,438 | 33.19 / 21.37 | 32.92 / 21.25 |
| CoRal read-aloud | 9,122 | 21.96 / 9.13 | 21.82 / 9.04 |
| FTSpeech test_balanced | 5,534 | 11.48 / 6.17 | 11.26 / 6.03 |
| Common Voice fixed Rye subset | 2,756 | 8.99 / 3.21 | 8.65 / 3.09 |
| Five-set mean | 26,780 | 16.48 / 8.53 | 16.21 / 8.38 |

Submit the **current upstream** result JSON. Keep the original audited result
as provenance, not as a second row or a historical hosted-model evaluation:
these are different scoring revisions of the same inference run.
Upstream rounds each dataset to two decimals before taking its macro mean.

Dataset revisions, reference/audio fingerprints and original normalization are
in `evaluation/soniox-benchmark.json`; none of the baseline pins changed.
Common Voice is the fixed 2,756-clip CV25 test set extracted from CV27, not the
complete 2,761-clip CV27 test set. Eleven CoRal conversation references normalize
to empty under the original harness; their predictions are retained and the
official metrics determine exclusion.

For upstream Compare/speaker matching, non-CV sample IDs are exported as the
official loader's `{column}_{index:05d}.wav` basenames. CV IDs preserve the
original `common_voice_da_*.wav` basenames from the fixed manifest. A complete
old/new ID map is in the bundle provenance directory. References and hypotheses
are copied verbatim, with order and all clips preserved.

## Rebuild and verify locally

The review checkout is a clean, separate upstream clone at the commit above.
The original pinned package and environment have not been upgraded.
From the project root, using the existing Python 3.11 environment:

```powershell
.venv\Scripts\python.exe -X utf8 -m evaluation.soniox_submission
```

The exporter refuses smoke/incomplete runs, changed audit inputs/predictions,
a different or modified upstream checkout, and an existing destination. Use
`--destination .cache/soniox-pr-submission-v2` to create another preserved bundle.
No API key or network access is needed. It invokes:

```powershell
.venv\Scripts\python.exe -X utf8 .cache/soniox-upstream-review/scripts/rescore.py --outputs-dir .cache/soniox-pr-submission/outputs --model soniox-stt-async-v5 --out-dir .cache/soniox-pr-submission/results
```

The bundle includes file checksums, original audit, sample-ID map, current scoring
source hashes/package versions, and a `reproduction/` tree containing the custom
runner, tests, pins, baseline fingerprints, dependency lockfile and documentation.
The runner files come from the archived run's `source/` snapshot, with their
recorded hashes checked, rather than assuming the current workspace matches.
During this review, the working `evaluation/soniox.py` lacked the archived
rate limiter and Windows write retry; two of its 18 offline tests failed.
That file was preserved. The submission reproduction copy uses the actual
audited client and its offline suite passes all 18 tests.
Inference reproduction requires gated dataset access, local CV preparation,
FFmpeg and a locally configured Soniox API key; it incurs API charges. Follow
`reproduction/evaluation/SONIOX.md` for setup and the exact custom runner. The
upstream repository already registers a Soniox backend, but its older SDK API
must not be represented as the command used for this REST evaluation.

## PR assembly and remaining maintainer steps

1. Start a branch in a fork of Rye-A1/danish-asr-leaderboard at current main.
   This workspace has no Git remote or tracked base commit; it is not the
   upstream PR checkout. Do not commit this entire research workspace.
2. Copy bundle `results/` and `outputs/` into the upstream tree. Merge the single
   entry from `model_metadata.entry.json` into `scripts/model_metadata.json`.
3. Include the small provenance/reproduction bundle in an agreed submissions
   documentation directory, or provide an immutable public source URL before
   dropping it. A hash alone cannot replace unavailable reproduction source.
4. Use `SONIOX_PR_DESCRIPTION.md`, keeping the custom-runner and normalization
   disclosures. Review the text-containing output files as the intended public
   submission; audio and local raw API job archives remain excluded.
5. Upstream maintainers require their own reproduction before publication.
   Their review may request an integrated modern Soniox backend or an additional
   hosted inference run. No new paid run is needed just to rescore these outputs.
6. On merge, upstream CI publishes scores/transcripts and computes derived
   intervals/Compare assets. Do not invoke the publication scripts manually.

The submission artifacts and PR description are prepared locally. No fork,
branch, commit, PR, or external publication was created during this check.

Local validation is recorded in bundle
`provenance/submission-validation.json`: all 26,780 exported reference/hypothesis
pairs match verbatim, all original raw-job hashes still match the audit, audited
source hashes match the reproduction copy, the result schema/means agree, and
the original baseline/export remain unchanged. The reproduction suite passed
18/18 offline tests. The bundle's `SHA256SUMS.json` covers every prepared file;
checks found no API-key values, audio or model weights in it.
