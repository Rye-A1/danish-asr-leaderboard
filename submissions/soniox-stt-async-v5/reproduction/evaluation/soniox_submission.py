"""Prepare an offline RyeAI submission from an audited Soniox run (no API calls)."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

from evaluation.checkpoints import sha256

UPSTREAM_REVISION = "1de3f8ad2724c363c1c93349f48337792d15bf12"
COLUMNS = {
    "fleurs": "fleurs_da",
    "coral_conversation": "coral_conversation",
    "coral_read_aloud": "coral_read_aloud",
    "ftspeech": "ftspeech",
    "cv17": "cv17_da",
}
MODEL = "soniox-stt-async-v5"
MODEL_LINK = f"[{MODEL}](https://soniox.com/docs/stt/models#june-11-2026)"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare(run, upstream, destination):
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError("Review/pin the upstream revision before using a different checkout")
    if subprocess.check_output(
        ["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip():
        raise ValueError("Upstream checkout has modified tracked files")
    audit = read(run / "audit.json")
    summary = read(run / "summary.json")
    model = audit["models"]["soniox-stt-async-v5"]
    if not audit["complete"] or audit["missing_datasets"] or summary["smoke_test"]:
        raise ValueError("A complete audited five-set evaluation is required")
    if set(model["datasets"]) != set(COLUMNS) or summary["datasets"] != model["datasets"]:
        raise ValueError("Dataset scores differ from audit")
    for name in ("config", "summary"):
        if sha256(run / f"{name}.json") != audit[f"{name}_sha256"]:
            raise ValueError(f"Changed {name}; run the full audit again")
    if destination.exists():
        raise FileExistsError("Use a new destination; existing submission bundles are preserved")
    outputs = destination / "outputs" / MODEL
    id_map = {}
    cv_manifest = Path(".cache/common_voice/test/test_manifest.jsonl")
    cv_rows = [json.loads(line) for line in cv_manifest.read_text(encoding="utf-8").splitlines()]
    for name, column in COLUMNS.items():
        source = run / f"{name}.jsonl"
        if sha256(source) != model["transcript_sha256"][name]:
            raise ValueError(f"Changed {name} transcripts; run the full audit again")
        records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
        if len(records) != model["datasets"][name]["samples"]:
            raise ValueError(f"Wrong count: {name}")
        converted, mappings = [], []
        for index, record in enumerate(records):
            if record["id"] != f"{name}:{index}":
                raise ValueError(f"Unexpected sample order: {name}:{index}")
            clip_id = f"{column}_{index:05d}.wav"
            if name == "cv17":
                row = cv_rows[index]
                if row["text"].strip() != record["reference"]:
                    raise ValueError("Common Voice ID/reference mapping differs")
                clip_id = Path(row["audio_filepath"]).name
            converted.append({"id": clip_id, **{k: record[k] for k in ("reference", "hypothesis")}})
            mappings.append({"source_id": record["id"], "submission_id": clip_id})
        outputs.mkdir(parents=True, exist_ok=True)
        (outputs / f"{column}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in converted),
            encoding="utf-8",
        )
        id_map[column] = mappings
    meta = {
        "model": MODEL,
        "model_link": MODEL_LINK,
        "params_b": 0.0,
        "access": "proprietary",
        "submitted": audit["evaluation_completed_at"][:10],
        "unicode_form": "NFKC",
        "number_words": True,
        "spoken_numbers": True,
        "filler_words": True,
        "speed_x": None,
        "datasets": list(COLUMNS.values()),
        "model_revision": None,
        "api_model": "stt-async-v5",
        "endpoint": audit["settings"]["base_url"],
        "request_settings": audit["settings"]["request_settings"],
        "translation": None,
        "context": None,
        "inference_harness_revision": audit["settings"]["harness_revision"],
        "scoring_harness_revision": revision,
        "timing_note": "Speed omitted: archived timing sums concurrent per-clip API latencies.",
        "hardware": "Hosted Soniox service; server hardware undisclosed. Local Windows CPU client; no local GPU inference.",
    }
    write(outputs / "meta.json", meta)
    subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            str(upstream / "scripts/rescore.py"),
            "--outputs-dir",
            str(destination / "outputs"),
            "--model",
            MODEL,
            "--out-dir",
            str(destination / "results"),
        ],
        check=True,
    )
    result = read(destination / "results" / f"{MODEL}.json")
    write(
        destination / "model_metadata.entry.json",
        {
            MODEL: {
                "entry_kind": "system",
                "configuration": "Hosted async API; Danish hint",
                "released": {
                    "date": "2026-06-11",
                    "source": "published",
                    "note": "https://soniox.com/docs/stt/models#june-11-2026",
                    "plot_by": "submitted",
                },
                "training_code": None,
                "training_data": None,
                "model_card": "https://soniox.com/docs/stt/models#june-11-2026",
                "paper": None,
            }
        },
    )
    write(destination / "provenance" / "sample-id-map.json", id_map)
    write(destination / "provenance" / "original-audit.json", audit)
    write(
        destination / "provenance" / "scoring-comparison.json",
        {
            "original_harness_revision": audit["settings"]["harness_revision"],
            "submission_harness_revision": revision,
            "original_scores": model["datasets"],
            "original_mean": model["core5_mean"],
            "submission_result": result,
            "packages": {name: version(name) for name in ("text2num", "num2words", "rapidfuzz")},
            "upstream_source_sha256": {
                str(path.relative_to(upstream)): sha256(path)
                for path in [
                    upstream / "scripts/rescore.py",
                    upstream / "danish_asr_leaderboard/metrics.py",
                    *sorted((upstream / "danish_asr_leaderboard/normalizer").glob("*.py")),
                ]
            },
        },
    )
    reproduction = destination / "reproduction"
    sources = [
        "pyproject.toml",
        "uv.lock",
        "LICENSE",
        "README.md",
        *[
            f"evaluation/{name}"
            for name in (
                "__init__.py",
                "checkpoints.py",
                "data.py",
                "results.py",
                "pins.json",
                "benchmark.json",
                "soniox.py",
                "soniox_evaluation.py",
                "soniox_run.py",
                "soniox_upgrade.py",
                "soniox_report.py",
                "soniox_submission.py",
                "test_soniox.py",
                "test_soniox_report.py",
                "SONIOX.md",
                "SONIOX_SUBMISSION.md",
                "SONIOX_PR_DESCRIPTION.md",
            )
        ],
    ]
    for source in sources:
        target = reproduction / source
        target.parent.mkdir(parents=True, exist_ok=True)
        archived = run / "source" / Path(source).name
        if source.startswith("evaluation/") and archived.is_file():
            digest = audit["settings"]["code_sha256"].get(Path(source).name)
            if digest and sha256(archived) != digest:
                raise ValueError(f"Audited source changed: {archived.name}")
            shutil.copyfile(archived, target)
        else:
            shutil.copyfile(source, target)
    write(
        destination / "SHA256SUMS.json",
        {
            str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(destination.rglob("*"))
            if p.is_file()
        },
    )
    print(f"Prepared {sum(v['samples'] for v in model['datasets'].values())} clips: {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, default=Path("evaluation/results/soniox-stt-async-v5-us/full")
    )
    parser.add_argument("--upstream", type=Path, default=Path(".cache/soniox-upstream-review"))
    parser.add_argument("--destination", type=Path, default=Path(".cache/soniox-pr-submission"))
    args = parser.parse_args()
    prepare(args.run.resolve(), args.upstream.resolve(), args.destination.resolve())


if __name__ == "__main__":
    main()
