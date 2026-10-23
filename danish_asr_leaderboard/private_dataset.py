"""Local CSV evaluation using the shared scorer, outside leaderboard outputs."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import platform
from itertools import islice
from pathlib import Path

from danish_asr_leaderboard.audio import audio_to_wav, wav_duration
from danish_asr_leaderboard.backends import load_backend
from danish_asr_leaderboard.custom_datasets import load_config
from danish_asr_leaderboard.metrics import compute_cer, compute_wer
from danish_asr_leaderboard.raw_outputs import write_dataset_outputs
from danish_asr_leaderboard.results import slugify, today_iso
from danish_asr_leaderboard.scoring import transcribe_dataset


def load_local_dataset(source_csv: Path, audio_dir: Path, max_samples: int = 0):
    """Validate selected rows and transcode local audio; never silently drop clips."""
    if max_samples < 0:
        raise ValueError("--max-samples must be nonnegative")
    content = source_csv.read_bytes()
    fingerprint = hashlib.sha256(content)
    columns = ("audio_path", "reference_text")
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline=""), strict=True)
    header = reader.fieldnames or []
    if not all(key in header for key in columns) or len(header) != len(set(header)):
        raise ValueError(f"{source_csv.name}: expected unique CSV columns {columns}")
    cache = audio_dir / "private"
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in islice(reader, max_samples or None):
        location = f"{source_csv.name}:{reader.line_num}"
        if None in entry or any(value is None for value in entry.values()) or any(
            not entry[key].strip() for key in columns
        ):
            raise ValueError(f"{location}: expected a complete CSV row with nonempty {columns}")
        source = (source_csv.parent / entry["audio_path"]).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"{location}: audio file not found: {source}")
        audio_hash = hashlib.sha256()
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                audio_hash.update(chunk)
        digest = audio_hash.hexdigest()
        fingerprint.update(digest.encode("ascii"))
        wav = cache / f"{digest}.wav"
        if not wav.exists() and not audio_to_wav(source, wav):
            wav.unlink(missing_ok=True)
            raise ValueError(f"{location}: audio conversion failed")
        if wav_duration(str(wav)) <= 0:
            raise ValueError(f"{location}: audio is empty or unreadable")
        rows.append({"audio_path": str(wav.resolve()), "reference_text": entry["reference_text"]})
    if not rows:
        raise ValueError("Private dataset CSV is empty")
    return rows, fingerprint.hexdigest()


def evaluate_private(args, options) -> None:
    """Use existing backends/normalisation/metrics without public data or notifications."""
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    config, source_csv = load_config(args.dataset)
    rows, dataset_sha = load_local_dataset(
        source_csv, Path(args.audio_cache_dir), args.max_samples,
    )
    backend = load_backend(args.backend, args.model, options)
    try:
        refs, hyps, infer_s, audio_s, raw = transcribe_dataset(
            backend, rows, batch_size=args.batch_size,
            unicode_form=args.unicode_form, number_words=args.number_words,
            filler_words=args.filler_words,
        )
    finally:
        backend.release()
    if len(hyps) != len(rows):
        raise ValueError("Backend must return one transcript per recording")
    if not any(ref.strip() for ref in refs):
        raise ValueError("No references remain after normalisation")
    model_id = args.model_id or args.model
    result = {
        "private_dataset": True,
        "dataset": config.name,
        "model": model_id,
        "model_ref": args.model,
        "model_revision": args.model_revision,
        "backend": args.backend,
        "dataset_sha256": dataset_sha,
        "samples": len(rows),
        "max_samples": args.max_samples,
        "wer": compute_wer(refs, hyps),
        "cer": compute_cer(refs, hyps),
        "speed_x": audio_s / infer_s if infer_s > 0 else None,
        "normalization": {key: getattr(args, key) for key in (
            "unicode_form", "number_words", "filler_words",
        )},
        "decoding": {key: getattr(args, key) for key in (
            "device", "compute_type", "batch_size", "nemo_model_type", "nemo_model_file",
            "nemo_beam_size", "kenlm_model", "kenlm_alpha", "kenlm_beam_size",
            "elevenlabs_model_id", "google_chirp_model_id", "soniox_model",
        )},
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "submitted": today_iso(),
    }
    # Keep private and partial runs out of the public publishing scripts' globs.
    run = f"smoke-{args.max_samples}" if args.max_samples else "full"
    suffix = Path("private") / dataset_sha / run
    if args.outputs_dir:
        write_dataset_outputs(Path(args.outputs_dir) / suffix, model_id, "private", raw)
    out_path = Path(args.out_dir) / suffix / f"{slugify(model_id)}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Private dataset WER: {result['wer']:.2f}% | CER: {result['cer']:.2f}%")
    print(f"Saved: {out_path}")
