"""Danish ASR test-set loaders and the benchmark dataset registry.

Each loader returns a list of ``{"audio_path": str, "reference_text": str}`` rows,
materialising audio to 16 kHz mono WAV under an on-disk cache and writing a
JSONL manifest so subsequent runs skip the download/transcode step.

The five *core* test sets (whose macro-average forms ``mean_wer`` / ``mean_cer``)
are CoRal-v3 conversation, CoRal-v3 read-aloud, Common Voice 17 (da), FLEURS
(da_dk) and FTSpeech.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from danish_asr_leaderboard.audio import audio_bytes_to_wav

Row = dict[str, str]
Loader = Callable[[Path, int], list[Row]]


# ---------------------------------------------------------------------------
# Shared materialisation helpers
# ---------------------------------------------------------------------------

def _read_manifest(manifest_path: Path) -> list[Row]:
    rows: list[Row] = []
    with open(manifest_path, encoding="utf-8") as mf:
        for line in mf:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if Path(entry["audio_path"]).exists():
                rows.append(entry)
    return rows


def _write_manifest(manifest_path: Path, rows: list[Row]) -> None:
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for entry in rows:
            mf.write(json.dumps(entry) + "\n")
    print(f"  Manifest written: {manifest_path}")


def _materialise(
    *,
    slug: str,
    label: str,
    audio_dir: Path,
    load_ds: Callable[[], "object"],
    text_keys: tuple[str, ...],
    max_samples: int,
    progress_every: int = 0,
) -> list[Row]:
    """Download, transcode and cache a HF dataset split into WAV rows.

    ``load_ds`` returns a ``datasets.Dataset`` already cast to
    ``Audio(decode=False)``. ``text_keys`` are tried in order for the transcript.
    """
    manifest_path = audio_dir / f"{slug}.manifest.jsonl"
    if manifest_path.exists() and max_samples == 0:
        print(f"\n--- Loading {label} (from manifest) ---")
        rows = _read_manifest(manifest_path)
        print(f"  {label}: {len(rows)} usable samples (manifest)")
        return rows

    print(f"\n--- Loading {label} ---")
    ds = load_ds()
    out_dir = audio_dir / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    # Iterate (don't index): works for both map-style and *streaming* datasets.
    # Streaming is essential for large corpora — the non-streaming path generates
    # every split (train+val+test) on first download, so ``split="test"`` on a
    # dataset with a huge train split (e.g. CoRal-v3's 147k train examples) would
    # pull the entire corpus. Streaming reads only the requested split's shards.
    cap = max_samples if max_samples > 0 else None
    known = len(ds) if hasattr(ds, "__len__") else None
    print(f"  processing {'all' if cap is None else cap} test sample(s)"
          + (f" of {known}" if known is not None else " (streaming)"))

    rows: list[Row] = []
    for i, row in enumerate(ds):
        if cap is not None and i >= cap:
            break
        text = ""
        for key in text_keys:
            text = (row.get(key) or "").strip()
            if text:
                break
        if not text:
            continue
        wav_path = out_dir / f"{slug}_{i:05d}.wav"
        if not wav_path.exists():
            audio_bytes_to_wav(row["audio"], wav_path)
        if wav_path.exists():
            rows.append({"audio_path": str(wav_path), "reference_text": text})
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  [{slug}] {i + 1} processed...")

    print(f"  {label}: {len(rows)} usable samples")
    if max_samples == 0:
        _write_manifest(manifest_path, rows)
    return rows


# ---------------------------------------------------------------------------
# Per-dataset loaders
# ---------------------------------------------------------------------------

def _coral_loader(config: str, slug: str, label: str) -> Loader:
    def load(audio_dir: Path, max_samples: int = 0) -> list[Row]:
        from datasets import Audio, load_dataset

        def load_ds():
            # Stream the test split: CoRal-v3 has a 147k-example train split, and
            # the non-streaming path generates *all* splits on first download.
            ds = load_dataset("CoRal-project/coral-v3", config, split="test", streaming=True)
            return ds.cast_column("audio", Audio(decode=False))

        return _materialise(
            slug=slug, label=label, audio_dir=audio_dir, load_ds=load_ds,
            text_keys=("text",), max_samples=max_samples, progress_every=200,
        )

    return load


def load_common_voice(audio_dir: Path, max_samples: int = 0) -> list[Row]:
    """Common Voice (da) test split.

    Reads a locally-prepared manifest at ``$CV_DATA_DIR/test/test_manifest.jsonl``
    (NeMo-style ``audio_filepath``/``text`` rows), produced by
    ``scripts/fetch_common_voice_da.py``. This is the supported path: modern
    ``datasets`` (>=4) no longer runs Mozilla's script-based loader and the repo
    ships no parquet, so the HF fallback below typically fails — set ``CV_DATA_DIR``.
    """
    cv_data_dir = os.environ.get("CV_DATA_DIR", "")
    local_manifest = Path(cv_data_dir) / "test" / "test_manifest.jsonl" if cv_data_dir else None
    if local_manifest and local_manifest.exists():
        print(f"\n--- Loading Common Voice da test split (local manifest) ---")
        rows: list[Row] = []
        with open(local_manifest, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                audio_path = entry.get("audio_filepath", "")
                text = (entry.get("text") or "").strip()
                if audio_path and text and Path(audio_path).exists():
                    rows.append({"audio_path": audio_path, "reference_text": text})
        if max_samples > 0:
            rows = rows[:max_samples]
        print(f"  Common Voice da: {len(rows)} usable samples (local)")
        return rows

    print(
        "  NOTE: CV_DATA_DIR not set (or manifest missing). The supported path is a "
        "local manifest from scripts/fetch_common_voice_da.py:\n"
        "        python scripts/fetch_common_voice_da.py --output-dir cv_da\n"
        "        export CV_DATA_DIR=$PWD/cv_da\n"
        "  Attempting the HF fallback (usually fails on datasets>=4)…",
    )

    from datasets import Audio, load_dataset

    def load_ds():
        # Stream the test split: the non-streaming path generates all splits
        # (cv17 da train) and was failing with "doesn't contain any data files".
        ds = load_dataset(
            "mozilla-foundation/common_voice_17_0", "da", split="test",
            streaming=True, token=True,
        )
        return ds.cast_column("audio", Audio(decode=False))

    return _materialise(
        slug="cv17_da", label="Common Voice 17 da test split", audio_dir=audio_dir,
        load_ds=load_ds, text_keys=("sentence", "text"), max_samples=max_samples,
    )


def load_fleurs(audio_dir: Path, max_samples: int = 0) -> list[Row]:
    """FLEURS da_dk test split.

    ``google/fleurs`` ships a deprecated loading script, so the parquet files are
    loaded directly from the Hub's auto-converted parquet branch.
    """
    from datasets import Audio, load_dataset

    def load_ds():
        ds = load_dataset(
            "parquet",
            data_files={"test": "hf://datasets/google/fleurs@refs/convert/parquet/da_dk/test/*.parquet"},
            split="test",
            num_proc=8,
        )
        return ds.cast_column("audio", Audio(decode=False))

    return _materialise(
        slug="fleurs_da", label="FLEURS da_dk test split", audio_dir=audio_dir,
        load_ds=load_ds, text_keys=("transcription", "raw_transcription"),
        max_samples=max_samples,
    )


def load_ftspeech(audio_dir: Path, max_samples: int = 0) -> list[Row]:
    """FTSpeech test_balanced split (parliamentary speech)."""
    from datasets import Audio, load_dataset

    def load_ds():
        # Stream test_balanced: the non-streaming path generates *all* splits,
        # and FTSpeech's ~1800h train will fill the disk.
        ds = load_dataset("alexandrainst/ftspeech", split="test_balanced", streaming=True)
        return ds.cast_column("audio", Audio(decode=False))

    return _materialise(
        slug="ftspeech", label="FTSpeech test_balanced split", audio_dir=audio_dir,
        load_ds=load_ds, text_keys=("text", "sentence"), max_samples=max_samples,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSpec:
    selector: str        # CLI selector name (--datasets)
    column: str          # leaderboard column prefix, e.g. "cv17_da" -> cv17_da_wer
    title: str           # human-readable label
    hf_id: str
    split: str
    core: bool           # part of the mean_wer / mean_cer macro-average
    loader: Loader = field(repr=False)


DATASETS: dict[str, DatasetSpec] = {
    spec.selector: spec
    for spec in [
        DatasetSpec("coral_conversation", "coral_conversation",
                    "CoRal-v3 conversation", "CoRal-project/coral-v3", "test", True,
                    _coral_loader("conversation", "coral_conversation", "CoRal-v3 conversation test split")),
        DatasetSpec("coral_read_aloud", "coral_read_aloud",
                    "CoRal-v3 read-aloud", "CoRal-project/coral-v3", "test", True,
                    _coral_loader("read_aloud", "coral_read_aloud", "CoRal-v3 read_aloud test split")),
        DatasetSpec("cv17", "cv17_da",
                    "Common Voice 17 (da)", "mozilla-foundation/common_voice_17_0", "test", True,
                    load_common_voice),
        DatasetSpec("fleurs", "fleurs_da",
                    "FLEURS (da_dk)", "google/fleurs", "test", True,
                    load_fleurs),
        DatasetSpec("ftspeech", "ftspeech",
                    "FTSpeech", "alexandrainst/ftspeech", "test_balanced", True,
                    load_ftspeech),
    ]
}

DEFAULT_DATASETS = ",".join(DATASETS)
CORE_COLUMNS = [spec.column for spec in DATASETS.values() if spec.core]
