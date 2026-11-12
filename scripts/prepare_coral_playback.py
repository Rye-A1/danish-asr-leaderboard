#!/usr/bin/env python3
"""Prepare a local CoRal audio bundle from the audited public benchmark cache.

This does not publish the bundle. See the leaderboard's CoRal playback notes
before making it accessible to others.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

COUNTS = {"coral_conversation": 8438, "coral_read_aloud": 9122}
SOURCE = "https://huggingface.co/datasets/CoRal-project/coral-v3"


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def prepare_audio(source: Path, destination: Path) -> bool:
    audio, sample_rate = sf.read(source, dtype="float32")
    peak = float(np.max(np.abs(audio), initial=0))
    if 0 < peak < 0.5:
        gain = min(10 ** (-3 / 20) / peak, 100)
        sf.write(destination, audio * gain, sample_rate, subtype="PCM_16")
        return True
    shutil.copyfile(source, destination)
    return False


def add_license_notice(path: Path, license_text: str) -> None:
    notice = (
        b"CoRal v3 test audio: converted to mono 16 kHz PCM; quiet playback copies "
        b"may be amplified by up to 40 dB. Source: CoRal project, "
        b"https://huggingface.co/datasets/CoRal-project/coral-v3 . "
        b"This copy is subject to the following complete licence and use restrictions:\n\n"
    ) + license_text.encode("utf-8") + b"\0"
    comment = b"ICMT" + struct.pack("<I", len(notice)) + notice + (b"\0" if len(notice) % 2 else b"")
    info = b"INFO" + comment
    chunk = b"LIST" + struct.pack("<I", len(info)) + info
    with path.open("r+b") as wav:
        header = wav.read(12)
        wav.seek(0, 2)
        size = wav.tell()
        if (len(header) != 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE"
                or struct.unpack_from("<I", header, 4)[0] != size - 8):
            raise ValueError(f"Unexpected WAV: {path.name}")
        wav.write(chunk)
        wav.seek(4)
        wav.write(struct.pack("<I", size + len(chunk) - 8))


def prepare(source: Path, manifests: Path, mapping: Path, outputs: Path,
            license_file: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    license_text = license_file.read_text(encoding="utf-8")
    if "LICENSED MATERIAL" not in license_text or "Distribution and Redistribution" not in license_text:
        raise ValueError("Expected the complete CoRal license")

    stage_map = {}
    for entry in read_jsonl(mapping):
        if not isinstance(entry, list) or len(entry) != 4:
            raise ValueError("Unexpected benchmark mapping format")
        original, staged, size, sha = entry
        name = Path(original).name
        if name in stage_map:
            raise ValueError(f"Duplicate mapped clip: {name}")
        stage_map[name] = (Path(staged), size, sha)

    verified = {}
    for dataset, expected in COUNTS.items():
        rows = read_jsonl(manifests / f"{dataset}.manifest.jsonl")
        output = read_jsonl(outputs / f"{dataset}.jsonl")
        refs = {Path(row["id"]).name: row["reference"] for row in output}
        if len(rows) != expected or len(output) != expected or len(refs) != expected:
            raise ValueError(f"Unexpected row count or duplicate IDs: {dataset}")
        items = {}
        for row in rows:
            name = Path(row["audio_path"]).name
            if not re.fullmatch(rf"{dataset}_\d{{5}}\.wav", name) or name in items:
                raise ValueError(f"Unexpected or duplicate clip ID: {name}")
            reference = row["reference_text"]
            if refs.get(name) != reference:
                raise ValueError(f"Reference mismatch: {name}")
            staged, size, sha = stage_map[name]
            if staged.parent.name != dataset or not re.fullmatch(r"\d{5}\.wav", staged.name):
                raise ValueError(f"Unexpected staged clip path: {name}")
            wav = source / dataset / staged.name
            if wav.stat().st_size != size or digest(wav) != sha:
                raise ValueError(f"Audio checksum mismatch: {name}")
            items[name] = (reference, wav)
        verified[dataset] = items
        print(f"Verified {len(items)} {dataset} clips and references", flush=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".coral-playback-", dir=destination.parent) as tmp:
        root = Path(tmp) / destination.name
        root.mkdir(mode=0o700)
        for dataset, items in verified.items():
            folder = root / "audio" / dataset
            folder.mkdir(parents=True)
            adjusted = 0
            for name, (_, wav) in items.items():
                target = folder / name
                adjusted += prepare_audio(wav, target)
                add_license_notice(target, license_text)
            manifest = {
                "source": SOURCE,
                "license": "CoRal OpenRAIL-D",
                "examples": {name: ref for name, (ref, _) in items.items()},
            }
            (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            (folder / "LICENSE").write_text(license_text, encoding="utf-8")
            (folder / "README.md").write_text(
                f"# CoRal v3: {dataset} test audio\n\n"
                f"Source: {SOURCE}.\n"
                "CoRal project: Alexandra Institute, University of Copenhagen, "
                "Agency for Digital Government, Alvenir and Corti.\n\n"
                "Citation: Dan Saattrup Smart, Sif Bernstorff Lehmann, Simon Rands "
                "Leminen, Anders Jess Pedersen, Anna Katrine van Zee and Torben "
                "Blach (2024), CoRal: A Diverse Danish ASR Dataset Covering "
                "Dialects, Accents, Genders, and Age Groups.\n\n"
                "License and use restrictions: see LICENSE in this directory.\n\n"
                "The benchmark converted the original audio to mono 16 kHz PCM WAV. "
                "Quiet clips were amplified for playback, with a maximum gain of 40 dB. "
                "The speech content was not edited.\n", encoding="utf-8")
            print(f"Prepared {len(items)} {dataset} clips ({adjusted} amplified)", flush=True)
        root.rename(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "manifests", "mapping", "outputs", "license-file", "destination"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.manifests, args.mapping, args.outputs, args.license_file, args.destination)


if __name__ == "__main__":
    main()
