# Soniox stt-async-v5 submission

Completed evaluation of Soniox model (soniox-stt-async-v5)

See [submission methodology](reproduction/evaluation/SONIOX_SUBMISSION.md),
[original independent audit](provenance/original-audit.json),
[current scoring comparison](provenance/scoring-comparison.json), and
[validation receipt](provenance/submission-validation.json).

The inference client in reproduction/evaluation/ comes from the audited run snapshot.
Run it from the reproduction project root. The original benchmark and lockfile
are included for exact data/scoring provenance. No audio, API credentials or
raw remote job IDs are included. Per-clip text is in ../../outputs/soniox-stt-async-v5/.

The original pinned scores are 16.48% mean WER / 8.53% mean CER.
The submission uses current upstream offline rescoring: 16.21% / 8.38%.
