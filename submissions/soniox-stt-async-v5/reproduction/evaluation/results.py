"""Rye-compatible scoring and transparent, resumable transcript files."""

import hashlib
import json
from pathlib import Path

from danish_asr_leaderboard.metrics import compute_cer, compute_wer
from danish_asr_leaderboard.normalizer import normalise

NORMALIZATION = {"unicode_form": "NFKC", "number_words": True, "filler_words": True}


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def fingerprint(rows):
    content = [{k: r[k] for k in ("id", "reference_text", "audio_sha256")} for r in rows]
    return hashlib.sha256(json.dumps(content, ensure_ascii=False).encode()).hexdigest()


def resume(path, rows):
    if not path.exists():
        return []
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(records) > len(rows):
        raise ValueError("Saved transcripts exceed dataset size")
    for record, row in zip(records, rows):
        if (record["id"], record["reference"]) != (row["id"], row["reference_text"]):
            raise ValueError("Saved transcript order/references differ from the dataset")
    return records


def score(records, expected):
    if len(records) != expected or not records:
        raise ValueError("Refusing to score an incomplete or empty dataset")
    refs = [normalise(r["reference"], **NORMALIZATION) for r in records]
    hyps = [normalise(r["hypothesis"], **NORMALIZATION) for r in records]
    return {
        "wer": compute_wer(refs, hyps),
        "cer": compute_cer(refs, hyps),
        "samples": len(records),
        "empty_normalized_references": refs.count(""),
    }


def core_mean(scores, names, smoke):
    if smoke or set(scores) != set(names):
        return None
    return {
        metric: sum(s[metric] for s in scores.values()) / len(names) for metric in ("wer", "cer")
    }


def write_overview(root):
    """A readable table of finished full datasets; smoke/partial runs stay out."""
    lines = [
        "# Completed evaluations",
        "",
        "WER/CER are percentages; lower is better.",
        "",
        "| Model | Dataset | Clips | WER | CER |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for path in sorted(root.glob("*/full/summary.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        for name, metrics in result["datasets"].items():
            lines.append(
                f"| {path.parent.parent.name} | {name} | {metrics['samples']} | "
                f"{metrics['wer']:.2f} | {metrics['cer']:.2f} |"
            )
    lines += ["", "Only completed datasets are shown. See each run's config.json for settings."]
    target = root / "summary.md"
    temp = target.with_suffix(".tmp")
    temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temp.replace(target)
