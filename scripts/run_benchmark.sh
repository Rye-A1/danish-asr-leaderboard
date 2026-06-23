#!/usr/bin/env bash
# Run the full Danish ASR benchmark sweep — one model per process.
#
# Each model is a separate process, so the OS reclaims 100% of the GPU memory
# between models (no cross-model OOM). Raw per-sample outputs are saved to
# outputs/<slug>/ automatically; scoring uses the published default normalisation
# (NFC). Failures are logged and skipped so one bad model doesn't abort the sweep.
#
# Per-backend interpreters
# -------------------------
# Backends pin conflicting transformers versions, so each conflicting family lives
# in its own venv. NeMo needs transformers<=4.52; qwen-asr pins ==4.57.6; the rest
# (whisper/wav2vec2/seamless/vibevoice/voxtral) want >=4.54/>=5.3. The sweep routes
# by backend to an interpreter chosen via env vars; defaults to a plain `python3`
# in the active env, so a single-venv user changes nothing:
#   DEFAULT_PY   every backend not listed below           (default: python3)
#   NEMO_PY      nemo / nemo-salm                          (default: $DEFAULT_PY)
#   QWEN_PY      qwen-asr                                  (default: $DEFAULT_PY)
# Example (this repo's three-venv layout):
#   DEFAULT_PY=.venv/bin/python NEMO_PY=.venv-nemo/bin/python QWEN_PY=.venv-qwen/bin/python \
#     bash scripts/run_benchmark.sh
#
# Common Voice: cv17_da needs a local manifest (modern `datasets` can't load the
# script-based HF repo). Fetch it once and export CV_DATA_DIR, else cv17 is skipped:
#   python scripts/fetch_common_voice_da.py --output-dir cv_da   # needs MOZILLA_API_KEY
#   export CV_DATA_DIR=$PWD/cv_da
#
# Usage:
#   bash scripts/run_benchmark.sh                 # local (open-weight) models
#   RUN_API=1 bash scripts/run_benchmark.sh       # also run hosted-API models
#   MAX_SAMPLES=50 bash scripts/run_benchmark.sh  # quick smoke (cap per dataset)
#   ONLY_BACKENDS=nemo bash scripts/run_benchmark.sh        # run only these backends
#
# After the sweep:
#   python scripts/push_results.py     # publish scores
#   python scripts/push_outputs.py     # publish raw outputs (Parquet per model)
set -uo pipefail

# Run from the repo root so `run_eval.py` and results/outputs paths resolve.
cd "$(dirname "$0")/.." || exit 1

MAX_SAMPLES="${MAX_SAMPLES:-0}"
RUN_API="${RUN_API:-0}"
DEFAULT_PY="${DEFAULT_PY:-python3}"
NEMO_PY="${NEMO_PY:-$DEFAULT_PY}"
QWEN_PY="${QWEN_PY:-$DEFAULT_PY}"
ONLY_BACKENDS="${ONLY_BACKENDS:-}"   # comma-separated allow-list; empty = all

py_for() {  # interpreter for a given backend
  case "$1" in
    nemo|nemo-salm) echo "$NEMO_PY" ;;
    qwen-asr)       echo "$QWEN_PY" ;;
    *)              echo "$DEFAULT_PY" ;;
  esac
}

# Mattermost progress notifications: no-op unless MATTERMOST_WEBHOOK_URL is set
# (in the env or a repo-root .env). Never blocks or fails the sweep.
notify() { "$DEFAULT_PY" notify.py "$1" >/dev/null 2>&1 || true; }

want_backend() {  # honour ONLY_BACKENDS allow-list
  [ -z "$ONLY_BACKENDS" ] && return 0
  case ",$ONLY_BACKENDS," in *",$1,"*) return 0 ;; *) return 1 ;; esac
}
# Batch size: 16 matches the established leaderboard methodology (canary repo's
# eval default), so scores stay comparable to existing entries. Properly-masked
# batching is output-invariant, so this only affects throughput/speed_x — override
# with BATCH=… if you want to trade comparability for speed. (qwen-asr and voxtral
# transcribe one utterance at a time regardless — batch size won't help them.)
BATCH="${BATCH:-16}"
COMMON=(--device cuda --unicode-form NFC --batch-size "$BATCH")
[ "$MAX_SAMPLES" -gt 0 ] && COMMON+=(--max-samples "$MAX_SAMPLES")

# model | backend | extra args (space-separated, may be empty)
LOCAL_MODELS=(
  "syvai/hviske-v5|cohere-asr|"     # CohereASR architecture (not Whisper) — needs librosa
  "syvai/hviske-v5.3|cohere-asr|"
  "syvai/hviske-v5.1|cohere-asr|"
  "syvai/hviske-v3-conversation|transformers|"
  "CoRal-project/roest-v3-whisper-1.5b|transformers|"
  "openai/whisper-large-v3|transformers|"
  "openai/whisper-large-v3-turbo|transformers|"
  "openai/whisper-small|transformers|"
  "openai/whisper-base|transformers|"
  "openai/whisper-tiny|transformers|"
  "capacit-ai/saga|qwen-asr|"
  "pluttodk/milo-asr|qwen-asr|"
  "Qwen/Qwen3-ASR-1.7B|qwen-asr|"
  "CoRal-project/roest-v2-wav2vec2-2B|wav2vec2|"
  "CoRal-project/roest-v2-wav2vec2-1B|wav2vec2|"
  "CoRal-project/roest-v3-wav2vec2-315m|wav2vec2|"
  "facebook/mms-1b-all|wav2vec2|"
  "nvidia/canary-1b-v2|nemo|--nemo-model-type canary"
  "nvidia/parakeet-tdt-0.6b-v3|nemo|--nemo-model-type parakeet"
  "nvidia/parakeet-rnnt-110m-da-dk|nemo|--nemo-model-type parakeet"   # NEW — raw .nemo, auto-restored
  # nemotron-3.5-asr-streaming-0.6b: DEFERRED — restore_from resolves to abstract ASRModel
  #   (needs the correct concrete NeMo class / newer nemo_toolkit) and emits <da-DK> language tags.
  # RyeAI in-house (rebranded from canary-1b-v2-da). Three rows: greedy, +KenLM, and the turbo parakeet.
  # KenLM is the 6-gram NeMo LM in the same repo; alpha=0.1 was the leaderboard-winning value.
  "RyeAI/krumme-v1|nemo|--nemo-model-type canary"   # NEW — canary-pnc-v2, greedy (no LM)
  "RyeAI/krumme-v1|nemo|--nemo-model-type canary --model-id RyeAI/krumme-v1+kenlm --kenlm-model RyeAI/krumme-v1:nemo_kenlm_6gram_light_100pct.nemo --kenlm-alpha 0.1 --kenlm-beam-size 5"  # NEW — canary-pnc-v2 + repo KenLM (separate row)
  "RyeAI/krumme-v1-turbo|nemo|--nemo-model-type parakeet"   # NEW — Parakeet fine-tune, raw .nemo, auto-restored
  "mistralai/Voxtral-Small-24B-2507|voxtral|"
  "mistralai/Voxtral-Mini-3B-2507|voxtral|"
  "facebook/seamless-m4t-v2-large|seamless|"
  "microsoft/VibeVoice-ASR-HF|vibevoice|"
)

# Hosted APIs (need credentials in env; only run when RUN_API=1).
API_MODELS=(
  "scribe_v2|elevenlabs|--elevenlabs-api-key ${ELEVENLABS_API_KEY:-}"
  "gpt-4o-transcribe-benchmark|azure-openai|--azure-openai-api-key ${AZURE_OPENAI_API_KEY:-} --azure-openai-endpoint ${AZURE_OPENAI_ENDPOINT:-}"
  "gpt-4o-mini-transcribe-benchmark|azure-openai|--azure-openai-api-key ${AZURE_OPENAI_API_KEY:-} --azure-openai-endpoint ${AZURE_OPENAI_ENDPOINT:-}"
)

run_one() {
  local model="$1" backend="$2" extra="$3"
  want_backend "$backend" || { echo "--- skip $model (backend=$backend not in ONLY_BACKENDS)"; return; }
  local py; py="$(py_for "$backend")"
  echo ""
  echo "=================================================================="
  echo ">>> $model  (backend=$backend, py=$py)"
  echo "=================================================================="
  notify "▶️ \`$model\` ($backend) starting…"
  local log; log="$(mktemp)"
  # shellcheck disable=SC2086
  "$py" run_eval.py --model "$model" --backend "$backend" "${COMMON[@]}" $extra 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then
    local wer; wer="$(grep -oE 'mean_wer \(core 5\)[^0-9]*[0-9.]+' "$log" | grep -oE '[0-9.]+$' | tail -1)"
    echo "<<< OK: $model  (mean_wer=${wer:-?})"
    notify "✅ \`$model\` done — mean_wer=**${wer:-?}**"
  else
    echo "!!! FAILED: $model (backend=$backend) — continuing" >&2
    echo "$model" >> benchmark_failures.txt
    notify "❌ \`$model\` ($backend) FAILED — continuing"
  fi
  rm -f "$log"
}

: > benchmark_failures.txt
notify "🚀 **Danish ASR sweep** started — backends=${ONLY_BACKENDS:-all}, max_samples=${MAX_SAMPLES}, batch=${BATCH}"
for entry in "${LOCAL_MODELS[@]}"; do
  IFS='|' read -r model backend extra <<< "$entry"
  run_one "$model" "$backend" "$extra"
done

if [ "$RUN_API" = "1" ]; then
  for entry in "${API_MODELS[@]}"; do
    IFS='|' read -r model backend extra <<< "$entry"
    run_one "$model" "$backend" "$extra"
  done
fi

echo ""
echo "Sweep complete. Results in results/, raw outputs in outputs/."
if [ -s benchmark_failures.txt ]; then
  echo "Failures:"; cat benchmark_failures.txt
  notify "🏁 **Sweep complete** with failures:
$(sed 's/^/• /' benchmark_failures.txt)"
else
  echo "No failures."
  notify "🏁 **Sweep complete** — all models OK (backends=${ONLY_BACKENDS:-all})"
fi
