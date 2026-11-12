# Preparing CoRal playback

`scripts/prepare_coral_playback.py` builds a local bundle from the audited public
benchmark audio cache. It checks all test-set clip identities, references,
counts and audio checksums against the benchmark output and source mapping. It
amplifies quiet playback copies by up to 40 dB without changing the benchmark
inputs, and adds the full CoRal license and conversion notice to each WAV's
metadata, as well as attribution and license files alongside the audio.

Run it with `--source`, `--manifests`, `--mapping`, `--outputs`, `--license-file`
and `--destination`; `--help` lists the required paths. The destination must
not exist. It contains `audio/coral_conversation` and `audio/coral_read_aloud`.
Run it with the project's Python environment (`numpy` and `soundfile` are
required).

The full test audio is hosted in
[`RyeAI/coral-v3-test-playback`](https://huggingface.co/datasets/RyeAI/coral-v3-test-playback).
The site fetches its manifests and playback copies from there. The comparison
data currently includes 50 examples per CoRal dataset.

The CoRal license and its distribution and use restrictions apply to the audio.
The browser acknowledgement is a UI affordance: direct audio URLs can bypass it.
Publishing updated audio is separate from the Space deployment.
