#!/usr/bin/env python3
"""Fetch the Common Voice Danish **test** split for the cv17_da leaderboard column.

Modern ``datasets`` (>=4) no longer runs the script-based
``mozilla-foundation/common_voice_17_0`` loader, and that repo ships no plain
parquet, so the HF path can't materialise the test split. Instead we pull the
official Common Voice Danish tarball from the Mozilla Data Collective API,
extract the requested split's clips (mp3 → 16 kHz mono wav), and write a
NeMo-style JSONL manifest that the eval harness reads via ``CV_DATA_DIR``.

Reproduce (fresh machine):
    export MOZILLA_API_KEY=...        # https://datacollective.mozillafoundation.org/
    python scripts/fetch_common_voice_da.py --output-dir cv_da
    export CV_DATA_DIR=$PWD/cv_da     # the eval harness now uses the local test set

If you already have the tarball, skip the API/key entirely:
    python scripts/fetch_common_voice_da.py --tarball /path/to/danish.tar.gz --output-dir cv_da

The manifest lands at ``<output-dir>/test/test_manifest.jsonl`` with
``audio_filepath`` + ``text`` fields — exactly what ``load_common_voice`` expects.

Requires: ffmpeg on PATH, ``soundfile`` (a core project dependency).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

import soundfile as sf

# Common Voice Danish scripted-speech dataset on the Mozilla Data Collective.
# Overridable via --dataset-id / --tarball for other versions.
CV_DATASET_ID = "cmn2cptsh01hymm07mulngxv0"
CV_API_BASE = "https://mozilladatacollective.com/api/datasets"
CV_TARBALL_NAME = "common-voice-scripted-speech-25-0-danish.tar.gz"
CV_TARBALL_GLOBS = ["common-voice-scripted-speech-*danish*.tar.gz", "cv-corpus-*da*.tar.gz"]


def _get_mozilla_api_key() -> str | None:
    key = os.environ.get("MOZILLA_API_KEY", "").strip()
    if key:
        return key
    for env_path in (Path(".env"), Path(__file__).resolve().parent.parent / ".env"):
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("MOZILLA_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip("\"'")
                    if val:
                        return val
    return None


def _find_tarball(search_dir: Path) -> Path | None:
    for pattern in CV_TARBALL_GLOBS:
        matches = sorted(search_dir.glob(pattern))
        if matches:
            return matches[-1]
    return None


def _download_tarball(output_dir: Path, dataset_id: str) -> Path:
    dest = output_dir / CV_TARBALL_NAME
    if dest.exists():
        print(f"Tarball already present: {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest
    api_key = _get_mozilla_api_key()
    if not api_key:
        raise SystemExit(
            "MOZILLA_API_KEY not found (env or .env). Get a key at "
            "https://datacollective.mozillafoundation.org/ or pass --tarball."
        )
    print("Requesting download URL from Mozilla Data Collective…")
    req = urllib.request.Request(
        f"{CV_API_BASE}/{dataset_id}/download", method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        data=b"{}",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode())
    url = body.get("downloadUrl") or body.get("download_url")
    if not url:
        raise SystemExit(f"No download URL in API response: {body}")
    print(f"Downloading {CV_TARBALL_NAME}…")
    urllib.request.urlretrieve(url, str(dest))
    print(f"Downloaded: {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
    return dest


def _mp3_to_wav(mp3_bytes: bytes, out_path: Path) -> bool:
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", "pipe:0", "-ar", "16000", "-ac", "1", str(out_path)],
            input=mp3_bytes, capture_output=True, timeout=30,
        )
        return r.returncode == 0 and out_path.exists()
    except Exception:
        return False


def _parse_tsv_from_tar(tarball_path: Path, split: str) -> dict[str, str]:
    """clip_filename → sentence from the split's TSV (dev.tsv for validation)."""
    # Common Voice names the validation split's TSV "dev.tsv".
    base = {"validation": "dev", "val": "dev"}.get(split, split)
    tsv_name = f"{base}.tsv"
    out: dict[str, str] = {}
    with tarfile.open(tarball_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not (member.name.endswith(f"/{tsv_name}") or member.name == tsv_name):
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            text = f.read().decode("utf-8")
            path_idx = text_idx = None
            for line_no, line in enumerate(text.splitlines()):
                if line_no == 0:
                    headers = line.split("\t")
                    path_idx = headers.index("path") if "path" in headers else None
                    for col in ("sentence", "text"):
                        if col in headers:
                            text_idx = headers.index(col)
                            break
                    if path_idx is None or text_idx is None:
                        print(f"WARNING: no path/text cols in {tsv_name}: {headers}", file=sys.stderr)
                        return {}
                    continue
                fields = line.split("\t")
                if len(fields) <= max(path_idx, text_idx):
                    continue
                clip, sentence = fields[path_idx].strip(), fields[text_idx].strip()
                if clip and sentence:
                    out[clip] = sentence
            break
    return out


def extract_split(tarball_path: Path, split: str, output_dir: Path, force: bool) -> int:
    split_dir = output_dir / split
    audio_dir = split_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = split_dir / f"{split}_manifest.jsonl"
    if manifest_path.exists() and not force:
        n = sum(1 for _ in manifest_path.open())
        print(f"[{split}] manifest exists ({n} entries): {manifest_path}")
        return n

    print(f"\n--- Extracting {split} split ---")
    tsv = _parse_tsv_from_tar(tarball_path, split)
    if not tsv:
        print(f"[{split}] no entries in {split}.tsv", file=sys.stderr)
        return 0
    print(f"[{split}] {len(tsv)} entries in TSV")

    rows: list[dict] = []
    skipped = 0
    with tarfile.open(tarball_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            clip = Path(member.name).name
            if clip not in tsv:
                continue
            wav_path = audio_dir / (Path(clip).stem + ".wav")
            if not (wav_path.exists() and not force):
                f = tar.extractfile(member)
                if f is None:
                    skipped += 1
                    continue
                raw = f.read()
                if clip.lower().endswith(".wav"):
                    wav_path.write_bytes(raw)
                    ok = True
                else:
                    ok = _mp3_to_wav(raw, wav_path)
                if not ok:
                    skipped += 1
                    continue
            try:
                duration = sf.info(str(wav_path)).duration
            except Exception:
                wav_path.unlink(missing_ok=True)
                skipped += 1
                continue
            rows.append({
                "audio_filepath": str(wav_path.resolve()),
                "text": tsv[clip],
                "duration": round(duration, 3),
                "dataset": "common_voice_da",
            })
            if len(rows) % 1000 == 0:
                print(f"  [{split}] {len(rows)} extracted…", flush=True)

    with open(manifest_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[{split}] done: {len(rows)} samples, {skipped} skipped → {manifest_path}")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch Common Voice Danish split(s) for the leaderboard")
    ap.add_argument("--output-dir", default="cv_da", help="Output dir (point CV_DATA_DIR here)")
    ap.add_argument("--splits", default="test", help="Comma-separated splits (default: test)")
    ap.add_argument("--tarball", default="", help="Existing tarball path (skips API download)")
    ap.add_argument("--dataset-id", default=CV_DATASET_ID, help="Mozilla Data Collective dataset id")
    ap.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.tarball:
        tarball_path = Path(args.tarball)
        if not tarball_path.exists():
            raise SystemExit(f"Tarball not found: {tarball_path}")
    else:
        tarball_path = _find_tarball(output_dir) or _find_tarball(output_dir.parent) \
            or _download_tarball(output_dir, args.dataset_id)
    print(f"Using tarball: {tarball_path}")

    total = sum(extract_split(tarball_path, s.strip(), output_dir, args.force)
                for s in args.splits.split(",") if s.strip())
    print(f"\nDone: {total} samples across splits [{args.splits}]")
    print(f"Set CV_DATA_DIR={output_dir.resolve()} for the eval harness.")


if __name__ == "__main__":
    main()
