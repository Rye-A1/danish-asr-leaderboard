"""Command-line entry point: evaluate one model and write a result JSON."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from danish_asr_leaderboard.backends import LoadOptions, available_backends, load_backend
from danish_asr_leaderboard.datasets import CORE_COLUMNS, DATASETS, DEFAULT_DATASETS
from danish_asr_leaderboard.metrics import compute_cer, compute_wer
from danish_asr_leaderboard.raw_outputs import write_dataset_outputs, write_meta
from danish_asr_leaderboard.results import (
    EvalResult,
    fetch_params_b,
    mean,
    model_link,
    today_iso,
    write_result_json,
)
from danish_asr_leaderboard.scoring import transcribe_dataset

API_BACKENDS = {"elevenlabs", "azure-openai", "google-chirp", "soniox"}


def _notify(msg: str) -> None:
    """Best-effort webhook ping; silent no-op if no webhook is configured."""
    try:
        from danish_asr_leaderboard.notify import notify

        notify(msg)
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate a Danish ASR model for the leaderboard")
    ap.add_argument("--model", required=True,
                    help="HF model id, local .nemo path, or API deployment/model name")
    ap.add_argument("--model-id", default="",
                    help="Override the id used for the result slug / leaderboard link")
    ap.add_argument("--backend", required=True, choices=available_backends())
    ap.add_argument("--nemo-model-type", default="canary", choices=["canary", "parakeet"],
                    help="NeMo model family (backend=nemo)")
    ap.add_argument("--params-b", type=float, default=None,
                    help="Parameter count in billions (auto-detected if omitted)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--compute-type", default="float16",
                    help="faster-whisper compute type (e.g. float16, int8)")
    ap.add_argument("--datasets", default=DEFAULT_DATASETS,
                    help=f"Comma-separated test sets to run. Available: {DEFAULT_DATASETS}")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="Cap samples per dataset (0 = all). Use for smoke tests.")
    ap.add_argument("--audio-cache-dir", default="eval_audio_cache")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--outputs-dir", default="outputs",
                    help="Where to persist raw per-sample model outputs (for offline "
                         "re-scoring). Empty string disables saving.")
    ap.add_argument("--unicode-form", default="NFC", choices=["NFC", "NFKC", "NFD", "NFKD"],
                    help="Unicode normalisation form applied before scoring (published "
                         "default: NFC). Raw outputs are saved regardless, so other forms "
                         "can be compared later via scripts/rescore.py.")
    ap.add_argument("--number-words", action=argparse.BooleanOptionalAction, default=True,
                    help="Expand standalone integer tokens to Danish cardinal words "
                         "(4 -> fire) before scoring, folding the digit<->word formatting "
                         "difference. ON by default (the published methodology); pass "
                         "--no-number-words to recover digit-preserving scoring.")
    ap.add_argument("--filler-words", action="store_true",
                    help="Remove Danish hesitation fillers (øh, hmm, ...) before scoring. "
                         "OFF by default; raw outputs are saved regardless, so this can "
                         "also be applied offline via scripts/rescore.py.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--access", default="open", choices=["open", "proprietary"],
                    help="Whether model weights are openly available")
    # NeMo beam / KenLM
    ap.add_argument("--kenlm-model", default=None,
                    help="KenLM .nemo path or repo_id:filename (Canary beam search)")
    ap.add_argument("--kenlm-alpha", type=float, default=0.075)
    ap.add_argument("--kenlm-beam-size", type=int, default=5)
    ap.add_argument("--nemo-beam-size", type=int, default=1)
    # ElevenLabs
    ap.add_argument("--elevenlabs-api-key", default=None)
    ap.add_argument("--elevenlabs-model-id", default="scribe_v2")
    # Azure OpenAI
    ap.add_argument("--azure-openai-api-key", default=None)
    ap.add_argument("--azure-openai-endpoint", default=None)
    ap.add_argument("--azure-openai-api-version", default="2025-01-01-preview")
    # Google Chirp
    ap.add_argument("--google-cloud-project", default=None)
    ap.add_argument("--google-credentials-file", default=None)
    ap.add_argument("--google-chirp-model-id", default="chirp_3")
    # Soniox
    ap.add_argument("--soniox-api-key", default=None)
    ap.add_argument("--soniox-model", default="soniox-v1")
    return ap.parse_args()


def _options_from_args(args: argparse.Namespace) -> LoadOptions:
    return LoadOptions(
        device=args.device,
        compute_type=args.compute_type,
        # NeMo
        nemo_model_type=args.nemo_model_type,
        nemo_beam_size=args.nemo_beam_size,
        # KenLM
        kenlm_model=args.kenlm_model,
        kenlm_alpha=args.kenlm_alpha,
        kenlm_beam_size=args.kenlm_beam_size,
        # ElevenLabs
        elevenlabs_api_key=args.elevenlabs_api_key,
        elevenlabs_model_id=args.elevenlabs_model_id,
        # Azure OpenAI
        azure_openai_api_key=args.azure_openai_api_key,
        azure_openai_endpoint=args.azure_openai_endpoint,
        azure_openai_api_version=args.azure_openai_api_version,
        # Google Chirp
        google_cloud_project=args.google_cloud_project,
        google_credentials_file=args.google_credentials_file,
        google_chirp_model_id=args.google_chirp_model_id,
        # Soniox
        soniox_api_key=args.soniox_api_key,
        soniox_model=args.soniox_model,
    )


def main() -> None:
    args = parse_args()
    model_id = args.model_id or args.model
    selectors = [s.strip() for s in args.datasets.split(",") if s.strip()]
    unknown = [s for s in selectors if s not in DATASETS]
    if unknown:
        print(f"ERROR: unknown dataset(s): {', '.join(unknown)}. "
              f"Available: {DEFAULT_DATASETS}", file=sys.stderr)
        sys.exit(1)

    audio_dir = Path(args.audio_cache_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    # --- Load test data ---
    loaded: dict[str, list[dict]] = {}
    for sel in selectors:
        spec = DATASETS[sel]
        try:
            rows = spec.loader(audio_dir, args.max_samples)
            if rows:
                loaded[spec.column] = rows
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: failed to load {spec.title}: {exc}", file=sys.stderr)

    # --- Load model ---
    backend = load_backend(
        name=args.backend, 
        model_ref=args.model, 
        options=_options_from_args(args)
    )

    # --- Resolve params_b ---
    params_b = args.params_b
    if params_b is None:
        params_b = fetch_params_b(model_id, backend.model)
    if params_b is None:
        if args.backend in API_BACKENDS:
            print("  params_b not applicable for API models; defaulting to 0.0")
            params_b = 0.0
        else:
            print("ERROR: Could not auto-detect parameter count. Pass --params-b.", file=sys.stderr)
            sys.exit(1)
    print(f"  params_b: {params_b}B")

    # --- Transcribe & score ---
    wer: dict[str, float | None] = {f"{spec.column}_wer": None for spec in DATASETS.values()}
    cer: dict[str, float | None] = {f"{spec.column}_cer": None for spec in DATASETS.values()}
    total_infer = 0.0
    total_audio = 0.0

    n_samples = sum(len(rows) for rows in loaded.values())
    _notify(
        f"🎤 `{model_id}` transcription started — {n_samples} samples across "
        f"{len(loaded)} dataset(s), batch {args.batch_size}, {args.unicode_form}"
    )

    for column, rows in loaded.items():
        print(f"\n--- Transcribing {column} ({len(rows)} samples) ---")
        refs, hyps, infer_s, audio_s, raw = transcribe_dataset(
            backend, rows, batch_size=args.batch_size,
            unicode_form=args.unicode_form, number_words=args.number_words,
            filler_words=args.filler_words,
        )
        total_infer += infer_s
        total_audio += audio_s
        w = round(compute_wer(refs, hyps), 2)
        c = round(compute_cer(refs, hyps), 2)
        wer[f"{column}_wer"] = w
        cer[f"{column}_cer"] = c
        print(f"  {column} WER: {w:.2f}% | CER: {c:.2f}%")
        if args.outputs_dir:
            out = write_dataset_outputs(args.outputs_dir, model_id, column, raw)
            print(f"  raw outputs → {out}")

    backend.release()

    speed_x = round(total_audio / total_infer, 1) if total_infer > 0 else None

    result = EvalResult(
        model=model_link(model_id),
        params_b=params_b,
        mean_wer=mean([wer[f"{c}_wer"] for c in CORE_COLUMNS]),
        mean_cer=mean([cer[f"{c}_cer"] for c in CORE_COLUMNS]),
        speed_x=speed_x,
        submitted=today_iso(),
        access=args.access,
        **wer,
        **cer,
    )

    out_path = write_result_json(result, Path(args.out_dir), model_id)

    if args.outputs_dir:
        write_meta(args.outputs_dir, model_id, {
            "model": model_id,
            "model_link": model_link(model_id),
            "params_b": params_b,
            "access": args.access,
            "submitted": result.submitted,
            "unicode_form": args.unicode_form,
            "number_words": args.number_words,
            "filler_words": args.filler_words,
            "speed_x": speed_x,
            "datasets": list(loaded.keys()),
        })

    print("\n=== Results ===")
    for spec in DATASETS.values():
        col = spec.column
        print(f"  {col + '_wer':<24}: {wer[f'{col}_wer']}%  (CER: {cer[f'{col}_cer']}%)")
    print(f"  {'mean_wer (core 5)':<24}: {result.mean_wer}%")
    print(f"  {'mean_cer (core 5)':<24}: {result.mean_cer}%")
    print(f"  {'speed_x':<24}: {result.speed_x}x real-time")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
