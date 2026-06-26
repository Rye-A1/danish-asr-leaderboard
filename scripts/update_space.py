#!/usr/bin/env python3
"""Build and deploy the static HTML Space.

Bakes leaderboard.json and models.py from the results parquet, resolving
provider logos and formatting sizes server-side, then uploads static files and
removes obsolete gradio files from the Space repo.

Usage:
  python scripts/update_space.py

Requires HF_TOKEN with write access to the RyeAI org:
  export HF_TOKEN=hf_...
"""
from __future__ import annotations

import functools
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from huggingface_hub import HfApi, get_token
from PIL import Image, ImageColor, ImageDraw, ImageFont

SPACE_REPO_ID   = "RyeAI/danish-asr-leaderboard"
DATASET_PARQUET = "hf://datasets/RyeAI/danish-asr-leaderboard/data/results.parquet"
SPACE_DIR = Path(__file__).resolve().parent.parent / "space"

UPLOAD = ["index.html", "leaderboard.json", "models.py", "README.md", "cover.jpeg"]
OBSOLETE = ["app.py", "requirements.txt"]

# Orgs to drop from the published board (case-insensitive match on the model's
# "<org>/..." prefix). Temporary: RyeAI models aren't public yet — clear this
# set (or remove the org) once the model repos are published.
EXCLUDE_ORGS = {"ryeai"}
# Specific models to drop from the board (case-insensitive, exact model name).
EXCLUDE_MODELS = {"syvai/hviske-v5.2"}

# Hosted/API models have no HF repo, so the "huggingface.co/<name>" link 404s.
# Point them at the provider's docs instead (substring match on the model name,
# case-insensitive). Unmatched API models get no link rather than a broken one.
PROVIDER_DOCS = {
    "scribe_v": "https://elevenlabs.io/docs/capabilities/speech-to-text",
    "gpt-4o": "https://platform.openai.com/docs/guides/speech-to-text",
    "chirp": "https://cloud.google.com/speech-to-text/v2/docs/chirp_3-model",
    "soniox": "https://soniox.com/docs/stt/get-started/transcribe-audio-file",
    "azure": "https://learn.microsoft.com/azure/ai-services/openai/concepts/models",
    "syv-transcribe": "https://syv.ai",
}

# Logos for hosted/API models (no HF avatar). Same substring keys as PROVIDER_DOCS.
PROVIDER_LOGO = {
    "scribe_v": "https://elevenlabs.io/_next/image?url=https%3A%2F%2Feleven-public-cdn.elevenlabs.io%2Fpayloadcms%2Felevenlabs-official-logo-11-icon.webp&w=1920&q=95",
    "syv-transcribe": "https://syv.ai/_next/image?url=%2F7.png&w=256&q=75",
}

# API models that have an HF org — fetch the avatar the same way as HF repos.
PROVIDER_HF_ORG = {
    "gpt-4o": "openai",
}

_MD_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
# A size the model advertises in its own name: 24B, 1.7B, 0.6b, 315m, …
_SIZE_IN_NAME = re.compile(r"(\d+(?:\.\d+)?)\s*([bBmM])(?![a-zA-Z])")

# Manual size overrides (in B), highest precedence. For models whose official
# size isn't in their name and differs from the safetensors count. Keyed by the
# exact model name as it appears on the board.
OFFICIAL_SIZE: dict[str, float] = {
    "syvai/hviske-v5": 2.0,
    "syvai/hviske-v5.3": 2.0,
    "syvai/hviske-v3-conversation": 2.0,
    "capacit-ai/saga": 1.7,
    "pluttodk/milo-asr": 1.7,
    "microsoft/VibeVoice-ASR-HF": 8.0,
    "facebook/seamless-m4t-v2-large": 2.0,
    "openai/whisper-large-v3": 2.0,
}

# 1200x630 = the OG/social-card standard. Content is centred so the card still
# reads when messengers (Slack/Mattermost/iMessage) crop it to a square thumbnail.
THUMBNAIL_SIZE = (1200, 630)
THUMBNAIL_BG = "#08060C"
THUMBNAIL_TEXT = "#FFFFFF"
THUMBNAIL_MUTED = "#C0A8B4"
THUMBNAIL_EYEBROW = "#B0A090"
THUMBNAIL_STAT = "#F5A0B0"
THUMBNAIL_STAT_LABEL = "#A0808C"
THUMBNAIL_DIVIDER = "#4A2030"
THUMBNAIL_RED = "#C8102E"
# Crimson radial glow, anchored lower-left, fading to near-black (offset → colour).
THUMBNAIL_GRADIENT = [(0.0, "#6E1230"), (0.40, "#480C20"), (0.72, "#1C0810"), (1.0, "#08060C")]
THUMBNAIL_OUT = SPACE_DIR / "cover.jpeg"


@functools.lru_cache(maxsize=256)
def _provider_logo(org: str) -> str:
    """Best-effort HF avatar URL for an org/user handle (cached per run)."""
    for kind in ("organizations", "users"):
        try:
            r = requests.get(
                f"https://huggingface.co/api/{kind}/{org}/avatar", timeout=4
            )
            if r.ok:
                url = r.json().get("avatarUrl")
                if url:
                    return url
        except Exception:
            continue
    return ""


def _fmt_size(x) -> str:
    """1 decimal (2 for sub-0.1B), em dash for 0 / NaN (API models)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if pd.isna(v) or v <= 0:
        return "—"
    return f"{v:.2f}" if v < 0.1 else f"{v:.1f}"


def _size_from_name(name: str) -> float | None:
    """Parameter count (in B) a model advertises in its own name, or None.

    e.g. ``Voxtral-Small-24B`` -> 24.0, ``Qwen3-ASR-1.7B`` -> 1.7,
    ``roest-v3-wav2vec2-315m`` -> 0.315. ``m``/``M`` is treated as millions.
    """
    matches = _SIZE_IN_NAME.findall(name)
    if not matches:
        return None
    num, unit = matches[-1]
    val = float(num)
    return val / 1000.0 if unit in "mM" else val


def _official_size(name: str, params_b) -> str:
    """Manual override > size advertised in the model name > computed count."""
    if name in OFFICIAL_SIZE:
        return _fmt_size(OFFICIAL_SIZE[name])
    named = _size_from_name(name)
    return _fmt_size(named if named is not None else params_b)


def _api_docs_url(name: str) -> str:
    """Provider-docs URL for a hosted/API model name, or '' if unknown."""
    low = name.lower()
    for key, url in PROVIDER_DOCS.items():
        if key in low:
            return url
    return ""


def _api_logo(name: str) -> str:
    """Provider logo URL for a hosted/API model name, or '' if unknown."""
    low = name.lower()
    for key, url in PROVIDER_LOGO.items():
        if key in low:
            return url
    for key, org in PROVIDER_HF_ORG.items():
        if key in low:
            return _provider_logo(org)
    return ""


def _num(x) -> float | None:
    """JSON-safe float rounded to 2 decimal places, or None."""
    try:
        v = float(x)
        if pd.isna(v):
            return None
        return round(v, 2)
    except (TypeError, ValueError):
        return None


def _parse_model(cell: str) -> tuple[str, str]:
    """Return (display_name, url) from a markdown link cell."""
    if not isinstance(cell, str):
        return str(cell), ""
    m = _MD_LINK.fullmatch(cell.strip())
    if m:
        return m.group(1), m.group(2)
    return cell, ""


def _is_hf_model_repo(name: str, url: str) -> bool:
    """Whether a leaderboard row points at a real HF model repository."""
    return bool(name and "/" in name and url.startswith("https://huggingface.co/"))


def build_models_py(data: dict) -> str:
    """Return the models.py content used by HF to backlink model cards."""
    model_names = sorted(
        {
            row["name"]
            for row in data["wer"]
            for url in [row.get("url", "")]
            for name in [row.get("name", "")]
            if _is_hf_model_repo(name, url)
        },
        key=str.casefold,
    )
    body = ",\n".join(f'    {name!r}' for name in model_names)
    return (
        '"""Auto-generated list of models registered in the Danish ASR leaderboard."""\n\n'
        f"MODEL_NAMES = [\n{body}\n]\n"
    )


def load_leaderboard_df() -> pd.DataFrame:
    df = pd.read_parquet(DATASET_PARQUET)

    if "rtf" in df.columns and "speed_x" not in df.columns:
        df["speed_x"] = (1.0 / df["rtf"]).round(1)
        df = df.drop(columns=["rtf"])
    if "access" not in df.columns:
        df["access"] = "open"

    # Drop excluded orgs and models before ranking so ranks stay contiguous.
    if EXCLUDE_ORGS or EXCLUDE_MODELS:
        excl_models = {m.lower() for m in EXCLUDE_MODELS}

        def _drop(cell) -> bool:
            name, _ = _parse_model(cell)
            org = name.split("/", 1)[0].lower() if "/" in name else ""
            return org in EXCLUDE_ORGS or name.lower() in excl_models

        keep = ~df["model"].map(_drop)
        dropped = int((~keep).sum())
        df = df[keep].reset_index(drop=True)
        if dropped:
            print(f"  excluded {dropped} row(s) "
                  f"(orgs={sorted(EXCLUDE_ORGS)}, models={sorted(EXCLUDE_MODELS)})")

    return df


def build_leaderboard_json(df: pd.DataFrame) -> dict:

    def build_rows(df_sorted: pd.DataFrame, metric_cols: list[str]) -> list[dict]:
        rows = []
        for rank, (_, row) in enumerate(df_sorted.iterrows(), 1):
            name, url = _parse_model(row.get("model", ""))
            is_repo = "/" in name
            org = name.split("/", 1)[0] if is_repo else ""
            # Hosted/API models aren't HF repos → provider docs + logo, not a 404.
            if is_repo:
                logo = _provider_logo(org)
            else:
                url = _api_docs_url(name)
                logo = _api_logo(name)
            submitted = row.get("submitted")
            entry: dict = {
                "rank": rank,
                "name": name,
                "url": url,
                "logo": logo,
                "access": str(row.get("access", "open")),
                "size": _official_size(name, row.get("params_b")),
                "submitted": str(submitted)[:10] if pd.notna(submitted) else "",
            }
            for col in metric_cols:
                entry[col] = _num(row.get(col))
            rows.append(entry)
        return rows

    wer_metrics = [
        "mean_wer", "mean_cer", "speed_x",
        "coral_conversation_wer", "coral_read_aloud_wer",
        "ftspeech_wer", "cv17_da_wer", "fleurs_da_wer",
    ]
    cer_metrics = [
        "mean_cer",
        "coral_conversation_cer", "coral_read_aloud_cer",
        "ftspeech_cer", "cv17_da_cer", "fleurs_da_cer",
    ]

    wer_df = (df.dropna(subset=["mean_wer"])
                .sort_values("mean_wer", ascending=True)
                .reset_index(drop=True))
    cer_df = (df.dropna(subset=["mean_cer"])
                .sort_values("mean_cer", ascending=True)
                .reset_index(drop=True))

    return {
        "updated": date.today().isoformat(),
        "org_logo": _provider_logo("RyeAI"),
        "wer": build_rows(wer_df, wer_metrics),
        "cer": build_rows(cer_df, cer_metrics),
    }


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a reasonable sans font available on macOS/Linux runners."""
    candidates = []
    if bold:
        candidates.extend([
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ])
    else:
        candidates.extend([
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ])
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_width(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    left, _top, right, _bottom = draw.textbbox((0, 0), text, font=font)
    return right - left


def _fit_font(draw: ImageDraw.ImageDraw, text: str, max_width: int, start_size: int, bold: bool = False):
    size = start_size
    while size > 28:
        font = _load_font(size, bold=bold)
        if _text_width(draw, text, font) <= max_width:
            return font
        size -= 4
    return _load_font(28, bold=bold)


def _radial_gradient(size: tuple[int, int], center: tuple[float, float],
                     radius_px: float, stops: list[tuple[float, str]]) -> Image.Image:
    """A smooth multi-stop radial gradient (offset 0→1 mapped to distance/radius)."""
    width, height = size
    cx, cy = center
    yy, xx = np.mgrid[0:height, 0:width]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / radius_px
    dist = np.clip(dist, 0.0, 1.0)
    offsets = [s[0] for s in stops]
    colors = [ImageColor.getrgb(s[1]) for s in stops]
    channels = [np.interp(dist, offsets, [c[i] for c in colors]) for i in range(3)]
    arr = np.dstack(channels).astype(np.uint8)
    return Image.fromarray(arr, "RGB").convert("RGBA")


def _count_datasets(data: dict) -> int:
    """Distinct test sets = per-dataset WER columns (excludes the macro mean)."""
    rows = data.get("wer", [])
    if not rows:
        return 5
    n = sum(1 for k in rows[0] if k.endswith("_wer") and k != "mean_wer")
    return n or 5


def generate_cover_image(data: dict, out_path: Path = THUMBNAIL_OUT) -> Path:
    """Generate the social thumbnail. Model/dataset counts come from live data;
    the layout is centred so it survives a square crop in messenger link cards."""
    width, height = THUMBNAIL_SIZE
    cx = width / 2

    image = _radial_gradient(
        THUMBNAIL_SIZE, (0.20 * width, 0.55 * height), 0.95 * width, THUMBNAIL_GRADIENT
    )
    draw = ImageDraw.Draw(image)

    eyebrow_font = _load_font(22, bold=True)
    title_font = _fit_font(draw, "Open Danish ASR", 1000, 92, bold=True)
    subtitle_font = _load_font(25)
    stat_value_font = _load_font(60, bold=True)
    stat_label_font = _load_font(19, bold=True)

    n_models = len(data.get("wer", []))
    n_datasets = _count_datasets(data)

    # Eyebrow + centred Danish flag.
    draw.text((cx, 72), "RYE AI", font=eyebrow_font, fill=THUMBNAIL_EYEBROW, anchor="ms")
    draw.rectangle((551, 102, 650, 176), fill=THUMBNAIL_RED)
    draw.rectangle((582, 102, 596, 176), fill=THUMBNAIL_TEXT)
    draw.rectangle((551, 132, 650, 146), fill=THUMBNAIL_TEXT)

    # Two-line title + subtitle, all centred.
    draw.text((cx, 282), "Open Danish ASR", font=title_font, fill=THUMBNAIL_TEXT, anchor="ms")
    draw.text((cx, 381), "Leaderboard", font=title_font, fill=THUMBNAIL_TEXT, anchor="ms")
    draw.text((cx, 438), "Consistent WER and CER evaluation across Danish test sets.",
              font=subtitle_font, fill=THUMBNAIL_MUTED, anchor="ms")

    divider = ImageColor.getrgb(THUMBNAIL_DIVIDER)
    draw.line((409, 480, 791, 480), fill=divider, width=1)

    # Stats: MODELS | DATASETS — both counts dynamic.
    draw.text((503, 551), str(n_models), font=stat_value_font, fill=THUMBNAIL_STAT, anchor="ms")
    draw.text((503, 586), "MODELS", font=stat_label_font, fill=THUMBNAIL_STAT_LABEL, anchor="ms")
    draw.line((600, 508, 600, 593), fill=divider, width=1)
    draw.text((697, 551), str(n_datasets), font=stat_value_font, fill=THUMBNAIL_STAT, anchor="ms")
    draw.text((697, 586), "DATASETS", font=stat_label_font, fill=THUMBNAIL_STAT_LABEL, anchor="ms")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(out_path, format="JPEG", quality=92, subsampling=0)
    return out_path


def main() -> None:
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        print(
            "ERROR: no HF credentials found. "
            "Set HF_TOKEN or run `huggingface-cli login`.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Building leaderboard.json …")
    df = load_leaderboard_df()
    data = build_leaderboard_json(df)
    out = SPACE_DIR / "leaderboard.json"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"  {out}  ({len(data['wer'])} WER rows, {len(data['cer'])} CER rows)")

    cover_out = generate_cover_image(data)
    print(f"  {cover_out}")

    models_out = SPACE_DIR / "models.py"
    models_out.write_text(build_models_py(data), encoding="utf-8")
    print(f"  {models_out}")

    api = HfApi(token=token)

    print("\nUploading static files …")
    for name in UPLOAD:
        path = SPACE_DIR / name
        if not path.exists():
            print(f"  skip (missing): {name}")
            continue
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=name,
            repo_id=SPACE_REPO_ID,
            repo_type="space",
            commit_message=f"Update {name}",
        )
        print(f"  ✓ {name}")

    print("\nRemoving obsolete gradio files …")
    space_files = set(api.list_repo_files(repo_id=SPACE_REPO_ID, repo_type="space"))
    for name in OBSOLETE:
        if name in space_files:
            api.delete_file(
                path_in_repo=name,
                repo_id=SPACE_REPO_ID,
                repo_type="space",
                commit_message=f"Remove {name} (static rewrite)",
            )
            print(f"  ✓ deleted {name}")
        else:
            print(f"  (not present) {name}")

    print(f"\nDone → https://huggingface.co/spaces/{SPACE_REPO_ID}")


if __name__ == "__main__":
    main()
