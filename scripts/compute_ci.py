#!/usr/bin/env python3
"""Compute sample-level bootstrap confidence intervals from the raw outputs.

The board used to bootstrap the *five datasets* (resample the 5 per-dataset scores
with replacement). With K=5 that estimates "what if we had picked a different set
of test sets" and yields intervals ~15-20x wider than the sampling noise in the
scores themselves — wide enough that the top models all appear statistically
indistinguishable.

This script instead resamples *utterances within each dataset* (each dataset keeps
its identity and size), recomputes that dataset's corpus WER/CER per replicate,
and averages the five to propagate to the mean. That answers the question a reader
actually asks of a CI: how much would this score move on a different draw of
speakers/utterances?

Corpus WER is a ratio of sums, so a replicate is just
``sum(edits[idx]) / sum(ref_len[idx])``. Per-utterance (edits, ref_len) are
computed once per model, then the bootstrap is pure vectorised resampling — the
edit distances dominate the runtime, not the 1000 replicates.

Writes ``data/ci.json``:

    {"<model name>": {"mean_wer_ci": [lo, hi], "mean_cer_ci": [lo, hi],
                      "<dataset>_wer_ci": [lo, hi], ...}, ...}

``update_space.py`` merges this into ``leaderboard.json`` so the page renders the
intervals instead of recomputing a coarse one in JS. Run after the outputs change:

  uv run --no-project --with pandas --with pyarrow --with numpy --with rapidfuzz \
      --with huggingface_hub --with num2words python scripts/compute_ci.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from danish_asr_leaderboard.datasets import CORE_COLUMNS
from danish_asr_leaderboard.metrics import corpus_rate, per_utterance_counts
from danish_asr_leaderboard.normalizer import normalise

DATASET_REPO_ID = "RyeAI/danish-asr-leaderboard"
OUT_PATH = Path("data/ci.json")
# Resample in blocks so the index matrix stays small for a 27k-utterance dataset.
BLOCK = 100
# The published methodology, stated explicitly rather than inherited from a
# default (scoring.py and normalise disagree on unicode_form).
UNICODE_FORM = "NFKC"
NUMBER_WORDS = True
FILLER_WORDS = False


def per_utterance_stats(refs: list[str], hyps: list[str]):
    """Return (word_edits, ref_words, char_edits, ref_chars) as float arrays.

    Delegates to ``metrics.per_utterance_counts`` — the same primitive the
    published corpus score is built from — so the interval and the point estimate
    cannot diverge in normalisation, in which pairs are dropped, or in how edits
    are measured. Normalisation options are passed explicitly rather than left to
    defaults, which differ between modules.
    """
    import numpy as np

    kw = dict(unicode_form=UNICODE_FORM, number_words=NUMBER_WORDS,
              filler_words=FILLER_WORDS)
    rn = [normalise(r, **kw) for r in refs]
    hn = [normalise(h, **kw) for h in hyps]
    we, rw = per_utterance_counts(rn, hn, unit="word")
    ce, rc = per_utterance_counts(rn, hn, unit="char")
    return (np.asarray(we, float), np.asarray(rw, float),
            np.asarray(ce, float), np.asarray(rc, float))


def bootstrap_means(stats: dict[str, tuple], columns: list[str], B: int, seed: int):
    """Bootstrap per-dataset rates and their mean.

    Returns (per_dataset_draws, mean_draws) where each per-dataset entry is a
    (B,) array of percentages. One shared replicate index per dataset is reused
    for the mean, so the mean's distribution accounts for all five jointly.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    per: dict[str, np.ndarray] = {}
    mean = np.zeros(B)
    used = [c for c in columns if c in stats]
    if not used:
        return per, mean
    for col in used:
        num, den = stats[col]
        n = len(num)
        draws = np.empty(B)
        for start in range(0, B, BLOCK):
            size = min(BLOCK, B - start)
            idx = rng.integers(0, n, size=(size, n))
            draws[start:start + size] = num[idx].sum(1) / den[idx].sum(1) * 100.0
        per[col] = draws
        mean += draws / len(used)
    return per, mean


def _published_mean(slug: str) -> float | None:
    """Published mean_wer for this model, if the result JSON is present."""
    path = Path("results") / f"{slug}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("mean_wer")
    except Exception:
        return None


def pct(arr) -> list[float] | None:
    import numpy as np

    if arr is None or len(arr) == 0:
        return None
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return [round(float(lo), 2), round(float(hi), 2)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample-level bootstrap CIs from raw outputs")
    ap.add_argument("-B", "--replicates", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="", help="Only this output slug (default: all)")
    ap.add_argument("--no-upload", action="store_true", help="Write locally, skip HF upload")
    args = ap.parse_args()

    import pandas as pd
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(DATASET_REPO_ID, repo_type="dataset")
    slugs = sorted(f[len("outputs/"):-len(".parquet")]
                   for f in files
                   if f.startswith("outputs/") and f.endswith(".parquet"))
    if args.model:
        slugs = [s for s in slugs if s == args.model]
        if not slugs:
            print(f"ERROR: no outputs parquet for {args.model!r}", file=sys.stderr)
            sys.exit(1)

    out: dict[str, dict] = {}
    for slug in slugs:
        url = f"https://huggingface.co/datasets/{DATASET_REPO_ID}/resolve/main/outputs/{slug}.parquet"
        try:
            df = pd.read_parquet(url)
        except Exception as exc:
            print(f"  SKIP {slug}: {exc}", file=sys.stderr)
            continue

        wer_stats: dict[str, tuple] = {}
        cer_stats: dict[str, tuple] = {}
        for col in CORE_COLUMNS:
            sub = df[df["dataset"] == col]
            if len(sub) == 0:
                continue
            we, rw, ce, rc = per_utterance_stats(
                [str(x) for x in sub["reference"]], [str(x) for x in sub["hypothesis"]]
            )
            wer_stats[col] = (we, rw)
            cer_stats[col] = (ce, rc)

        if not wer_stats:
            print(f"  SKIP {slug}: no core datasets", file=sys.stderr)
            continue

        wper, wmean = bootstrap_means(wer_stats, CORE_COLUMNS, args.replicates, args.seed)
        cper, cmean = bootstrap_means(cer_stats, CORE_COLUMNS, args.replicates, args.seed + 1)

        entry: dict[str, list[float] | None] = {
            "mean_wer_ci": pct(wmean),
            "mean_cer_ci": pct(cmean),
        }
        for col, draws in wper.items():
            entry[f"{col}_wer_ci"] = pct(draws)
        for col, draws in cper.items():
            entry[f"{col}_cer_ci"] = pct(draws)

        # Point estimate from the *same* counts that produced the interval. It is
        # recorded so the pair is self-consistent by construction, and checked
        # against the published score so a divergence surfaces here rather than
        # as a score sitting outside its own interval on the board.
        # Per-dataset rates are rounded to 2dp before averaging, matching how
        # rescore.py builds the published mean. The bootstrap replicates are
        # deliberately left unrounded — quantising each replicate would distort
        # the distribution the interval is read off.
        used = [c for c in CORE_COLUMNS if c in wer_stats]
        point_wer = round(
            sum(round(corpus_rate(*wer_stats[c]), 2) for c in used) / len(used), 2)
        point_cer = round(
            sum(round(corpus_rate(*cer_stats[c]), 2) for c in used) / len(used), 2)
        entry["mean_wer_point"] = point_wer
        entry["mean_cer_point"] = point_cer

        lo, hi = entry["mean_wer_ci"]
        if not (lo <= point_wer <= hi):
            print(f"  WARNING: {slug} point estimate {point_wer} outside CI "
                  f"[{lo}, {hi}]", file=sys.stderr)
        published = _published_mean(slug)
        drift = "" if published is None or round(abs(published - point_wer), 2) <= 0.01 \
            else f"  ** differs from published {published} **"
        out[slug] = entry
        print(f"  {slug:42} mean_wer {point_wer:6.2f}  95% CI [{lo:.2f}, {hi:.2f}]"
              f"  (width {hi - lo:.2f}){drift}")

    if not out:
        print("Nothing computed.", file=sys.stderr)
        sys.exit(1)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_PATH} ({len(out)} model(s), B={args.replicates})")

    if args.no_upload:
        return
    api.upload_file(
        path_or_fileobj=str(OUT_PATH),
        path_in_repo="data/ci.json",
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        commit_message=f"Update sample-level bootstrap CIs ({len(out)} models)",
    )
    print("Uploaded data/ci.json")


if __name__ == "__main__":
    main()
