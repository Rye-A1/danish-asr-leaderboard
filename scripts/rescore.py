"""Re-score saved raw model outputs under a chosen normaliser configuration.

The eval harness persists every model's *un-normalised* per-sample output under
``<outputs-dir>/<model-slug>/<column>.jsonl`` (plus a ``meta.json``). This script
reads those back, applies a normaliser configuration, recomputes corpus WER/CER per
dataset and the core-5 macro-averages, and writes a fresh result JSON — all without
re-running any model.

That makes normalisation a cheap, reproducible experiment: change the Unicode form
(or, later, plug in a word<->digit converter) and re-score the whole board in
seconds, comparing the deltas against the currently published results before
deciding whether to promote the change.

Examples
--------
    # Re-score every model under NFKC and compare against results/
    python scripts/rescore.py --unicode-form NFKC --compare results

    # Re-score one model, writing JSONs to a separate dir
    python scripts/rescore.py --model openai/whisper-large-v3 \
        --unicode-form NFKC --out-dir results_nfkc
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from danish_asr_leaderboard.datasets import CORE_COLUMNS, DATASETS
from danish_asr_leaderboard.metrics import compute_cer, compute_wer
from danish_asr_leaderboard.normalizer import normalise
from danish_asr_leaderboard.raw_outputs import read_dataset_outputs, read_meta
from danish_asr_leaderboard.results import (
    EvalResult,
    mean,
    slugify,
    write_result_json,
)

ALL_COLUMNS = [spec.column for spec in DATASETS.values()]


def _score_column(jsonl_path: Path, unicode_form: str) -> tuple[float, float]:
    records = read_dataset_outputs(jsonl_path)
    refs = [normalise(r["reference"], unicode_form=unicode_form) for r in records]
    hyps = [normalise(r["hypothesis"], unicode_form=unicode_form) for r in records]
    return round(compute_wer(refs, hyps), 2), round(compute_cer(refs, hyps), 2)


def rescore_model(model_dir: Path, unicode_form: str) -> tuple[dict, dict, dict]:
    """Return (wer, cer, meta) for one model directory."""
    wer: dict[str, float | None] = {f"{c}_wer": None for c in ALL_COLUMNS}
    cer: dict[str, float | None] = {f"{c}_cer": None for c in ALL_COLUMNS}
    for column in ALL_COLUMNS:
        jsonl_path = model_dir / f"{column}.jsonl"
        if jsonl_path.exists():
            w, c = _score_column(jsonl_path, unicode_form)
            wer[f"{column}_wer"] = w
            cer[f"{column}_cer"] = c
    meta = read_meta(model_dir.parent, model_dir.name)
    return wer, cer, meta


def build_result(wer: dict, cer: dict, meta: dict) -> EvalResult:
    return EvalResult(
        model=meta.get("model_link") or meta.get("model", ""),
        params_b=meta.get("params_b", 0.0) or 0.0,
        mean_wer=mean([wer[f"{c}_wer"] for c in CORE_COLUMNS]),
        mean_cer=mean([cer[f"{c}_cer"] for c in CORE_COLUMNS]),
        speed_x=meta.get("speed_x"),
        submitted=meta.get("submitted", ""),
        access=meta.get("access", "open"),
        **wer,
        **cer,
    )


def _published_mean_wer(compare_dir: Path, slug: str) -> float | None:
    path = compare_dir / f"{slug}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("mean_wer")


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-score saved raw outputs under a normaliser config")
    ap.add_argument("--outputs-dir", default="outputs",
                    help="Directory of saved raw outputs (default: outputs)")
    ap.add_argument("--model", default="",
                    help="Re-score only this model id (default: every model in outputs-dir)")
    ap.add_argument("--unicode-form", default="NFKC", choices=["NFC", "NFKC", "NFD", "NFKD"],
                    help="Unicode normalisation form to re-score under (default: NFKC)")
    ap.add_argument("--out-dir", default="results_rescored",
                    help="Where to write the re-scored result JSONs")
    ap.add_argument("--compare", default="",
                    help="Existing results dir to diff mean_wer against (e.g. 'results')")
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.is_dir():
        print(f"ERROR: outputs dir not found: {outputs_dir}", file=sys.stderr)
        sys.exit(1)

    if args.model:
        model_dirs = [outputs_dir / slugify(args.model)]
    else:
        model_dirs = sorted(d for d in outputs_dir.iterdir() if d.is_dir())
    model_dirs = [d for d in model_dirs if d.is_dir()]
    if not model_dirs:
        print(f"ERROR: no model output dirs under {outputs_dir}", file=sys.stderr)
        sys.exit(1)

    compare_dir = Path(args.compare) if args.compare else None
    out_dir = Path(args.out_dir)

    print(f"Re-scoring {len(model_dirs)} model(s) under unicode_form={args.unicode_form}\n")
    header = f"{'model':<48} {'mean_wer':>10}"
    if compare_dir:
        header += f" {'published':>10} {'Δ':>8}"
    print(header)
    print("-" * len(header))

    for model_dir in model_dirs:
        wer, cer, meta = rescore_model(model_dir, args.unicode_form)
        model_id = meta.get("model", model_dir.name)
        result = build_result(wer, cer, meta)
        write_result_json(result, out_dir, model_id)

        line = f"{model_id[:48]:<48} {result.mean_wer:>10.2f}"
        if compare_dir:
            pub = _published_mean_wer(compare_dir, slugify(model_id))
            if pub is None:
                line += f" {'—':>10} {'—':>8}"
            else:
                line += f" {pub:>10.2f} {result.mean_wer - pub:>+8.2f}"
        print(line)

    print(f"\nWrote re-scored results → {out_dir}/")


if __name__ == "__main__":
    main()
