"""Stable inputs for incremental CI and Compare builds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCORING_FILES = (
    "scripts/build_cache.py",
    "danish_asr_leaderboard/metrics.py",
    "danish_asr_leaderboard/normalizer/__init__.py",
    "danish_asr_leaderboard/normalizer/danish.py",
)


def source_signature(*paths: str, options: dict | None = None) -> str:
    digest = hashlib.sha256()
    for name in (*SCORING_FILES, *paths):
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    digest.update(json.dumps(options or {}, sort_keys=True).encode())
    return digest.hexdigest()


def output_fingerprints(siblings) -> dict[str, str | None]:
    """LFS content hashes survive unrelated dataset commits and repeated uploads.

    A missing hash disables reuse for that file rather than risking stale scores.
    """
    result = {}
    for sibling in siblings:
        name = sibling.rfilename
        if name.startswith("outputs/") and name.endswith(".parquet"):
            lfs = getattr(sibling, "lfs", None)
            result[name] = (lfs.get("sha256") if isinstance(lfs, dict)
                            else getattr(lfs, "sha256", None)) if lfs else None
    return dict(sorted(result.items()))


def reusable(slug: str, file: str, fingerprint: str | None, signature: str,
             manifest: dict, published: dict) -> bool:
    return bool(fingerprint and slug in published
                and manifest.get("signature") == signature
                and manifest.get("outputs", {}).get(file) == fingerprint)
