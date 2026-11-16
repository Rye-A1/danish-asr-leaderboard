# Four ASR model evaluations

These are the public five-dataset runs from Box 1. The adjacent files preserve the local inference adapters and CLI wrappers used for the evaluations; they are not registered as general-purpose benchmark backends. Model revisions and decoding settings are also recorded in each output directory's `meta.json`.

All scores use the standard Danish normalizer and the same 26,780 reference clips as the other leaderboard entries. Fluid and Parakeet Ultra use automatic language detection; Qwen uses the Danish language prompt; Phonon uses its default language detection. No private benchmark data is included.

## Inference settings

- **Fluid Speech Pro:** pinned revision `46027a3bffc5817ce8f9b29add0a062610134a90`, NeMo, float32, greedy batch 16, clean mode off, A100 80 GB.
- **Qwen3-ASR-0.6B-hf:** pinned revision `7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c`, Transformers, bfloat16, greedy batch 16, Danish, 256 output tokens, A100 80 GB. On 100 matching clips per dataset, 221 of 500 normalized transcripts differed from single-clip inference; per-dataset WER differed by 0.16–0.72 percentage points.
- **Phonon-2.5:** pinned revision `2af4bb8ccceb66a8ce06c4081bb968f1802c1e58`, five-value quantized CPU runtime, greedy with its default 1.05 repetition penalty, batch 1, four pinned AMD EPYC 7V13 vCPUs. Its measured CPU speed is recorded in metadata but omitted from the A100 speed column.
- **Parakeet Ultra:** pinned revision `73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b`, NeMo TDT float32, greedy batch 16, max 10 symbols per frame, A100 80 GB. Its 723 ASR tensors were mapped into the NVIDIA Parakeet v3 architecture and verified for equality; only its six VAD tensors were omitted. The tokenizer and preprocessor came from NVIDIA's revision `541d1f99c6b0c3cd0b11a95167540bb8edefd82b`. The copied adapter uses `PARAKEET_V3_NEMO` instead of the original machine-specific path.

## Commands used on Box 1

These commands are recorded with their original local paths; each used the corresponding `custom_backends.py` and `run_eval.py` preserved here, the benchmark's dataset configuration, and the pinned local model snapshot.

```sh
# Fluid
/home/azureuser/canary-triton-deploy/.venv/bin/python -u /nvme/leaderboard-fluid-qwen-20261115/run_eval.py --model /nvme/leaderboard-fluid-qwen-20261115/models/fluid/fluid-speech-pro.nemo --model-id FluidVoice-Labs/fluid-speech-pro --model-revision 46027a3bffc5817ce8f9b29add0a062610134a90 --backend fluid-speech-pro --params-b 0.63 --batch-size 16 --audio-cache-dir /nvme/run1-public-five-eval-20261020-v3/manifests/leaderboard_audio_cache --out-dir /nvme/leaderboard-fluid-qwen-20261115/results --outputs-dir /nvme/leaderboard-fluid-qwen-20261115/outputs

# Qwen
/home/azureuser/.venv-plapre/bin/python -u /nvme/leaderboard-fluid-qwen-20261115/run_eval.py --model /nvme/leaderboard-fluid-qwen-20261115/models/qwen --model-id Qwen/Qwen3-ASR-0.6B-hf --model-revision 7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c --backend qwen3-asr --params-b 0.6 --batch-size 16 --audio-cache-dir /nvme/run1-public-five-eval-20261020-v3/manifests/leaderboard_audio_cache --out-dir /nvme/leaderboard-fluid-qwen-20261115/results --outputs-dir /nvme/leaderboard-fluid-qwen-20261115/outputs

# Phonon
taskset -c 20-23 /home/azureuser/.venv-plapre/bin/python -u /nvme/leaderboard-fluid-qwen-20261115/run_eval.py --model /nvme/leaderboard-fluid-qwen-20261115/models/phonon-unpacked --model-id FermionResearch/Phonon-2.5 --model-revision 2af4bb8ccceb66a8ce06c4081bb968f1802c1e58 --backend phonon-2.5 --params-b 0.6 --device cpu --batch-size 1 --audio-cache-dir /nvme/run1-public-five-eval-20261020-v3/manifests/leaderboard_audio_cache --out-dir /nvme/leaderboard-fluid-qwen-20261115/results --outputs-dir /nvme/leaderboard-fluid-qwen-20261115/outputs

# Parakeet Ultra
/home/azureuser/canary-triton-deploy/.venv/bin/python -u /nvme/leaderboard-parakeet-ultra-nemo-20261116/run_eval.py --model /nvme/leaderboard-parakeet-ultra-20261116/models/ultra --model-id moondream/parakeet-ultra --model-revision 73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b --backend ultra-nemo --params-b 0.6 --batch-size 16 --audio-cache-dir /nvme/run1-public-five-eval-20261020-v3/manifests/leaderboard_audio_cache --out-dir /nvme/leaderboard-parakeet-ultra-nemo-20261116/results --outputs-dir /nvme/leaderboard-parakeet-ultra-nemo-20261116/outputs
```

The runs also set `EVAL_MODEL_REVISION`, `EVAL_BACKEND`, `EVAL_BATCH_SIZE`, `CV_DATA_DIR`, `PYTHONPATH`, offline Hugging Face settings and thread counts in their launch environment. Phonon additionally set `FERMION_DEVICE=cpu` and `FERMION_CPU_THREADS=4`. The copied Fluid/Qwen/Phonon wrapper originally recorded a generic Danish target for all three runs; the published metadata removes it for the two automatic-language models. CPU speed is kept separately in Phonon's metadata so rescoring cannot publish it as A100 throughput.
