---
title: Danish ASR Leaderboard
emoji: 🇩🇰
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
---

# Danish ASR Leaderboard

Gradio app for the [RyeAI/danish-asr-leaderboard](https://huggingface.co/datasets/RyeAI/danish-asr-leaderboard)
results dataset. It reads `data/results.parquet` from that dataset and renders
WER and CER leaderboards over five Danish test sets.

Source code and evaluation harness:
[github.com/Rye-A1/danish-asr-leaderboard](https://github.com/Rye-A1/danish-asr-leaderboard)

## Local development

```bash
pip install -r requirements.txt
python app.py
```

The app pulls results from the public HF dataset at startup; no token is
required to view it.
