---
title: Danish ASR Leaderboard
emoji: 🏆
colorFrom: red
colorTo: gray
sdk: static
app_file: index.html
thumbnail: https://huggingface.co/spaces/RyeAI/danish-asr-leaderboard/resolve/main/cover.jpeg
pinned: false
license: mit
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
