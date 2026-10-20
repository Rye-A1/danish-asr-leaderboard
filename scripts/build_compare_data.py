#!/usr/bin/env python3
"""Build the Compare page's transcript examples and aggregate speaker breakdown.

The source metadata is intentionally local: CoRal test rows are gated, so this
script publishes only group counts and scores, never clip-level speaker labels.
Use the same test split and row order as danish_asr_leaderboard.datasets.

Run after scripts/extract_compare_metadata.py has written its ignored JSONL:
    python scripts/build_compare_data.py \
      --metadata eval_audio_cache/compare-data/metadata.jsonl \
      --cache-dir eval_audio_cache/compare-data/hf
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from rapidfuzz.distance import Levenshtein

from danish_asr_leaderboard.metrics import compute_cer, compute_wer
from danish_asr_leaderboard.normalizer import normalise
from danish_asr_leaderboard.results import slugify

REPO = "RyeAI/danish-asr-leaderboard"
DATASETS = ("coral_conversation", "coral_read_aloud", "cv17_da", "fleurs_da", "ftspeech")
DEMOGRAPHIC_DATASETS = ("coral_conversation", "coral_read_aloud", "ftspeech")
ROOT = Path(__file__).resolve().parent.parent
NORMALISE = dict(unicode_form="NFKC", number_words=True, spoken_numbers=True, filler_words=True)


def output_slugs(outputs_dir: Path):
    """Resolve legacy raw-output names to their model's leaderboard name."""
    aliases = {}
    for model_dir in outputs_dir.iterdir():
        meta = model_dir / "meta.json"
        if model_dir.is_dir() and meta.is_file():
            name = json.loads(meta.read_text(encoding="utf-8")).get("model", "")
            if name:
                aliases[model_dir.name] = slugify(name)
    return aliases
def clip_id(value: str) -> str:
    return Path(str(value)).name


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def load_metadata(path: Path):
    rows = defaultdict(dict)
    revisions = {}
    for row in read_jsonl(path):
        dataset = row["dataset"]
        if dataset not in DEMOGRAPHIC_DATASETS:
            continue
        key = clip_id(row["id"])
        if key in rows[dataset]:
            raise ValueError(f"duplicate metadata ID: {dataset}/{key}")
        rows[dataset][key] = row
        if dataset in revisions and revisions[dataset] != row["source_sha"]:
            raise ValueError(f"mixed metadata revisions for {dataset}")
        revisions[dataset] = row["source_sha"]
    expected = {"coral_conversation": 8438, "coral_read_aloud": 9122, "ftspeech": 5534}
    for dataset, count in expected.items():
        if len(rows[dataset]) != count:
            raise ValueError(f"{dataset}: expected {count} metadata rows; got {len(rows[dataset])}")
    return rows, revisions


def group_label(dataset: str, dimension: str, value: str | None) -> str | None:
    if value is None or not str(value).strip():
        return None
    label = str(value).strip()
    if dimension == "gender":
        return {"M": "Men", "F": "Women", "male": "Men", "female": "Women"}.get(label, label)
    return label[0].upper() + label[1:]


def metadata_groups(metadata):
    groups = {}
    for dataset, rows in metadata.items():
        dims = ("gender", "dialect") if dataset.startswith("coral_") else ("gender",)
        groups[dataset] = {}
        for dim in dims:
            values = defaultdict(lambda: {"ids": set(), "speakers": set()})
            for key, row in rows.items():
                label = group_label(dataset, dim, row.get(dim))
                if label:
                    values[label]["ids"].add(key)
                    if row.get("speaker") is not None:
                        values[label]["speakers"].add(str(row["speaker"]))
            groups[dataset][dim] = {
                label: {"clips": len(value["ids"]), "speakers": len(value["speakers"]), "ids": value["ids"]}
                for label, value in sorted(values.items())
            }
    return groups


def choose_examples(outputs: Path, limit: int):
    """Choose high-disagreement clips from locally saved, recent raw outputs."""
    by_dataset = defaultdict(lambda: defaultdict(lambda: {"reference": "", "hypotheses": []}))
    for model_dir in sorted(outputs.iterdir()):
        if not model_dir.is_dir():
            continue
        for dataset in DATASETS:
            path = model_dir / f"{dataset}.jsonl"
            if not path.is_file():
                continue
            for row in read_jsonl(path):
                key = clip_id(row.get("id", ""))
                expected_prefix = "common_voice_da_" if dataset == "cv17_da" else dataset + "_"
                if not key.startswith(expected_prefix):
                    continue
                entry = by_dataset[dataset][key]
                if entry["reference"] and entry["reference"] != row["reference"]:
                    raise ValueError(f"reference mismatch in local outputs: {dataset}/{key}")
                entry["reference"] = row["reference"]
                entry["hypotheses"].append(row["hypothesis"])
    chosen = {}
    for dataset, samples in by_dataset.items():
        ranked = []
        for key, item in samples.items():
            ref = normalise(item["reference"], **NORMALISE)
            words = ref.split()
            if not 5 <= len(words) <= 80 or len(item["hypotheses"]) < 5:
                continue
            hyps = [normalise(h, **NORMALISE) for h in item["hypotheses"]]
            rates = [min(2.0, Levenshtein.distance(words, h.split()) / len(words)) for h in hyps]
            disagreement = max(rates) - min(rates)
            if len(set(hyps)) < 3 or disagreement < 0.15:
                continue
            ranked.append((disagreement, len(set(hyps)), key, item["reference"]))
        ranked.sort(key=lambda row: (-row[0], -row[1], row[2]))
        chosen[dataset] = {key: {"id": key, "reference": reference, "hypotheses": {}}
                           for _, _, key, reference in ranked[:limit]}
    return chosen


def preserve_examples(path: Path):
    """Keep published clip IDs when re-scoring without replacing their audio assets."""
    current = json.loads(path.read_text(encoding="utf-8"))
    return {
        dataset: {
            item["id"]: {"id": item["id"], "reference": item["reference"], "hypotheses": {}}
            for item in items
        }
        for dataset, items in current["examples"].items()
    }


def score_example(reference: str, hypothesis: str) -> dict[str, float]:
    """Score one displayed clip using the published leaderboard normalization."""
    ref = normalise(reference, **NORMALISE)
    hyp = normalise(hypothesis or "", **NORMALISE)
    return {"wer": round(compute_wer([ref], [hyp]), 1),
            "cer": round(compute_cer([ref], [hyp]), 1)}


def score_groups(frame: pd.DataFrame, dataset: str, metadata, groups):
    records = {}
    for row in frame.itertuples(index=False):
        key = clip_id(row.id)
        if key in records:
            raise ValueError(f"duplicate output ID: {dataset}/{key}")
        records[key] = row
    if set(records) != set(metadata[dataset]):
        return None  # Partial or different test split: never show a misleading score.
    for key, row in records.items():
        if row.reference != metadata[dataset][key]["reference"]:
            return None
    normalized = {key: (normalise(row.reference, **NORMALISE),
                        normalise(row.hypothesis or "", **NORMALISE))
                  for key, row in records.items()}
    result = {}
    for dimension, values in groups[dataset].items():
        result[dimension] = {}
        for label, group in values.items():
            keys = sorted(group["ids"])
            refs = [normalized[k][0] for k in keys]
            hyps = [normalized[k][1] for k in keys]
            result[dimension][label] = {
                "wer": round(compute_wer(refs, hyps), 2),
                "cer": round(compute_cer(refs, hyps), 2),
            }
    return result


def require_result_coverage(data: dict, results_dir: Path) -> None:
    """Do not deploy a score whose Compare examples or speaker scores are missing."""
    missing = []
    for dataset in DATASETS:
        if not data["examples"].get(dataset):
            missing.append(f"{dataset}: no examples")
    for result in sorted(results_dir.glob("*.json")):
        if result.name.startswith("._"):
            continue
        model = json.loads(result.read_text(encoding="utf-8"))["model"]
        linked = re.fullmatch(r"\[([^]]+)\]\(https?://[^)]+\)", model)
        slug = slugify(linked.group(1) if linked else model)
        groups = data["breakdown"].get(slug, {})
        for dataset in DEMOGRAPHIC_DATASETS:
            if dataset not in groups:
                missing.append(f"{slug}: {dataset} speaker scores")
        for dataset in DATASETS:
            examples = data["examples"].get(dataset, [])
            if not examples:
                continue
            count = sum(slug in example["hypotheses"] for example in examples)
            if count != len(examples):
                missing.append(f"{slug}: {dataset} examples ({count}/{len(examples)})")
    if missing:
        raise ValueError("Compare data is incomplete:\n" + "\n".join(missing))


def build(metadata_path: Path, outputs_dir: Path, cache_dir: Path, limit: int,
          examples_from: Path | None = None):
    metadata, revisions = load_metadata(metadata_path)
    groups = metadata_groups(metadata)
    examples = preserve_examples(examples_from) if examples_from else choose_examples(outputs_dir, limit)
    aliases = output_slugs(outputs_dir)
    api = HfApi()
    info = api.repo_info(REPO, repo_type="dataset")
    files = sorted(x.rfilename for x in info.siblings
                   if x.rfilename.startswith("outputs/") and x.rfilename.endswith(".parquet"))
    breakdown = {}
    coverage = {}
    for n, file in enumerate(files, 1):
        source_slug = Path(file).stem
        slug = aliases.get(source_slug, source_slug)
        path = hf_hub_download(REPO, file, repo_type="dataset", revision=info.sha,
                               local_dir=cache_dir)
        frame = pd.read_parquet(path, columns=["dataset", "id", "reference", "hypothesis"])
        model_breakdown = {}
        for dataset, part in frame.groupby("dataset", sort=False):
            if dataset not in DATASETS:
                continue
            if dataset in metadata:
                values = score_groups(part, dataset, metadata, groups)
                if values is not None:
                    model_breakdown[dataset] = values
            if dataset in examples:
                for row in part.itertuples(index=False):
                    key = clip_id(row.id)
                    example = examples[dataset].get(key)
                    if example and example["reference"] == row.reference:
                        example["hypotheses"][slug] = row.hypothesis or ""
        if model_breakdown:
            breakdown[slug] = model_breakdown
        coverage[slug] = sorted(model_breakdown)
        print(f"[{n}/{len(files)}] {slug}: {', '.join(coverage[slug]) or 'no exact demographic match'}", flush=True)
    public_groups = {
        dataset: {dimension: {label: {"clips": group["clips"], "speakers": group["speakers"]}
                              for label, group in labels.items()}
                  for dimension, labels in dimensions.items()}
        for dataset, dimensions in groups.items()
    }
    for items in examples.values():
        for example in items.values():
            example["normalized_reference"] = normalise(example["reference"], **NORMALISE)
            example["normalized_hypotheses"] = {
                slug: normalise(hypothesis or "", **NORMALISE)
                for slug, hypothesis in example["hypotheses"].items()
            }
            example["scores"] = {slug: score_example(example["reference"], hypothesis)
                                 for slug, hypothesis in example["hypotheses"].items()}
    return {
        "generated": date.today().isoformat(),
        "output_revision": info.sha,
        "metadata_revisions": revisions,
        "method": "Corpus WER/CER after leaderboard NFKC, spelling, abbreviation, unit, number and filler normalization; exact clip ID and reference match required.",
        "groups": public_groups,
        "breakdown": breakdown,
        "examples": {dataset: list(items.values()) for dataset, items in examples.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True, help="Local gated/public test metadata JSONL")
    parser.add_argument("--outputs-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "huggingface/space/compare_data.json")
    parser.add_argument("--examples-per-dataset", type=int, default=50)
    parser.add_argument("--preserve-examples-from", type=Path,
                        help="Existing compare_data.json whose clip IDs should stay fixed while re-scoring")
    parser.add_argument("--require-results-dir", type=Path,
                        help="Fail if a result lacks examples or speaker scores")
    args = parser.parse_args()
    data = build(args.metadata, args.outputs_dir, args.cache_dir,
                 args.examples_per_dataset, args.preserve_examples_from)
    if args.require_results_dir:
        require_result_coverage(data, args.require_results_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}: {len(data['breakdown'])} models, "
          f"{sum(map(len, data['examples'].values()))} examples")


if __name__ == "__main__":
    main()
