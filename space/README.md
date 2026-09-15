---
title: Danish Speech-to-Text Leaderboard
emoji: 🏆
colorFrom: red
colorTo: gray
sdk: static
app_file: index.html
thumbnail: https://huggingface.co/spaces/RyeAI/danish-asr-leaderboard/resolve/main/cover.jpeg
pinned: true
license: mit
short_description: "Open Danish speech-to-text benchmark: WER, CER and speed."
tags:
  - speech-to-text
  - leaderboard
  - automatic-speech-recognition
  - speech-recognition
  - danish
  - evaluation
  - benchmark
  - static
---

# Danish ASR Leaderboard

Static leaderboard for the [RyeAI/danish-asr-leaderboard](https://huggingface.co/datasets/RyeAI/danish-asr-leaderboard)
results dataset. WER and CER rankings across five Danish test sets.

Source code and evaluation harness:
[github.com/Rye-A1/danish-asr-leaderboard](https://github.com/Rye-A1/danish-asr-leaderboard)

## Deploying updates

Run the deploy script after pushing new results to the dataset:

```bash
export HF_TOKEN=hf_...
python scripts/update_space.py
```

This bakes `leaderboard.json` from the parquet (resolving provider logos and
formatting sizes server-side) and uploads the static files to the Space.

The Popularity sparklines read daily rolling-download snapshots from
`data/hf_downloads.json` in the results dataset. The scheduled GitHub workflow
updates that dataset file directly, so protected-branch rules do not block the
daily refresh.

`leaderboard.json` is generated, not tracked. To preview the page locally, fetch
the deployed copy first — the page fetches it at runtime, so without it the table
renders empty:

```bash
curl -sL https://huggingface.co/spaces/RyeAI/danish-asr-leaderboard/raw/main/leaderboard.json \
  -o space/leaderboard.json
python -m http.server 7860 --directory space
```
