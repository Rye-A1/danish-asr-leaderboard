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
from html import escape as html_escape
import re
import sys
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from huggingface_hub import HfApi, get_token
from PIL import Image, ImageDraw, ImageFont

SPACE_REPO_ID   = "RyeAI/danish-asr-leaderboard"
DATASET_PARQUET = "hf://datasets/RyeAI/danish-asr-leaderboard/data/results.parquet"
# Sample-level bootstrap CIs, precomputed from the raw outputs by
# scripts/compute_ci.py (too expensive to recompute on every deploy).
DATASET_CI_JSON = "https://huggingface.co/datasets/RyeAI/danish-asr-leaderboard/resolve/main/data/ci.json"
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
    "ordbogen": "https://odincore.ai/docs/models/ordbogen-whisper",
    # Distinct from the "gpt-4o" key above: gpt-transcribe is a separate model
    # with its own docs page, and its name contains no "gpt-4o" substring, so
    # without this entry it would render with no link and no logo at all.
    "gpt-transcribe": "https://developers.openai.com/api/docs/models/gpt-transcribe",
}

# Logos for hosted/API models (no HF avatar). Same substring keys as PROVIDER_DOCS.
PROVIDER_LOGO = {
    "scribe_v": "https://elevenlabs.io/_next/image?url=https%3A%2F%2Feleven-public-cdn.elevenlabs.io%2Fpayloadcms%2Felevenlabs-official-logo-11-icon.webp&w=1920&q=95",
    "syv-transcribe": "https://syv.ai/_next/image?url=%2F7.png&w=256&q=75",
    "ordbogen": "https://odincore.ai/apple-touch-icon.png",
    # The API docs' own favicon (48x48, blue ground, transparent corners) rather
    # than the HF "openai" org avatar — these rows are the hosted API, not the
    # open-weights repos, and the blue mark is how the product presents itself.
    "gpt-4o": "https://developers.openai.com/favicon.png",
    "gpt-transcribe": "https://developers.openai.com/favicon.png",
}

# API models that have an HF org — fall back to that org's avatar when there is
# no explicit PROVIDER_LOGO entry (which takes precedence).
PROVIDER_HF_ORG: dict[str, str] = {}

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
# reads when messengers (Slack/Mattermost/iMessage) crop it to a square thumbnail
# — the centred 630px zone is the safe area, and the stat row is sized to fit it.
THUMBNAIL_SIZE = (1200, 630)
# Everything is drawn at SS times final size and LANCZOS-downsampled, which is
# what makes the rounded corners and tight type read as crisp rather than jagged.
THUMBNAIL_SS = 3
THUMBNAIL_BG = (8, 7, 11)
THUMBNAIL_TEXT = (255, 255, 255)
THUMBNAIL_MUTED = (198, 176, 187)
THUMBNAIL_STAT = (255, 176, 191)
THUMBNAIL_STAT_LABEL = (152, 126, 140)
THUMBNAIL_RED = (198, 12, 46)
# Mesh gradient: additive radial sources over a near-black base, as
# (cx_frac, cy_frac, radius_frac_of_width, rgb, strength, falloff). A crimson
# bloom low-left, a violet counter-tone top-right, and a soft rose behind the
# title — enough variation that the field never reads as a flat wash.
THUMBNAIL_GLOWS = [
    (0.12, 0.86, 0.86, (212, 18, 52), 0.50, 2.3),
    (0.46, 0.52, 0.60, (136, 18, 64), 0.28, 2.2),
    (0.94, 0.08, 0.60, (66, 24, 100), 0.30, 2.7),
    (0.86, 0.90, 0.50, (160, 30, 76), 0.22, 2.4),
    (0.50, -0.16, 0.52, (255, 194, 208), 0.06, 2.0),
]
THUMBNAIL_GRAIN = 7.0
THUMBNAIL_VIGNETTE = 0.40
# Vendored so the card renders identically here and on the deploy runner; see
# assets/fonts/NOTICE. Falls back to system fonts if the directory is missing.
FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
FONT_BOLD, FONT_MEDIUM, FONT_REGULAR = "Roboto-Bold.ttf", "Roboto-Medium.ttf", "Roboto-Regular.ttf"
THUMBNAIL_OUT = SPACE_DIR / "cover.jpeg"


@functools.lru_cache(maxsize=1)
def _bootstrap_cis() -> dict:
    """Precomputed sample-level bootstrap CIs, keyed by output slug.

    Empty dict if the file isn't published yet — the page falls back to showing
    no interval rather than a coarse dataset-level one.
    """
    try:
        r = requests.get(DATASET_CI_JSON, timeout=10)
        if r.ok:
            return r.json()
        print(f"  WARNING: no ci.json (HTTP {r.status_code}) — CIs omitted", file=sys.stderr)
    except Exception as exc:
        print(f"  WARNING: could not fetch ci.json ({exc}) — CIs omitted", file=sys.stderr)
    return {}


def _slugify(model_id: str) -> str:
    """Match danish_asr_leaderboard.results.slugify (outputs/ file naming)."""
    return re.sub(r"[^a-zA-Z0-9_-]", "__", model_id.strip("/"))


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


@functools.lru_cache(maxsize=256)
def _model_license(model_id: str) -> str:
    """Licence tag from the model's HF repo, or '' if none/unavailable.

    Read live from the Hub (same approach as the org avatars) rather than stored
    in the result JSONs, so a re-licensed model corrects itself on the next
    deploy instead of going stale. Hosted API models have no repo — they get ''
    and render as an em dash.
    """
    try:
        r = requests.get(f"https://huggingface.co/api/models/{model_id}", timeout=4)
        if not r.ok:
            return ""
        for tag in r.json().get("tags", []):
            if tag.startswith("license:"):
                return tag.split(":", 1)[1]
    except Exception:
        pass
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
            access = str(row.get("access", "open"))
            # "Has a slash" is not enough to mean "is an HF repo": ordbogen/whisper
            # is a hosted API whose name is org-shaped, and the slash heuristic sent
            # it to a huggingface.co URL that 404s. An explicit provider mapping
            # wins instead — but only for proprietary rows, so an open HF model
            # whose name happens to contain e.g. "azure" keeps its real repo link.
            docs = _api_docs_url(name) if access == "proprietary" else ""
            is_repo = "/" in name and not docs
            org = name.split("/", 1)[0] if is_repo else ""
            # Hosted/API models aren't HF repos → provider docs + logo, not a 404.
            if is_repo:
                logo = _provider_logo(org)
            else:
                url = docs
                logo = _api_logo(name)
            submitted = row.get("submitted")
            entry: dict = {
                "rank": rank,
                "name": name,
                "url": url,
                "logo": logo,
                "access": str(row.get("access", "open")),
                "license": _model_license(name) if is_repo else "",
                "size": _official_size(name, row.get("params_b")),
                "submitted": str(submitted)[:10] if pd.notna(submitted) else "",
            }
            for col in metric_cols:
                entry[col] = _num(row.get(col))
            # speed_x is a throughput measurement of *our* run. For a local model
            # that is a real property (inference on one A100). For a hosted API it
            # is network latency plus whatever concurrency our client happened to
            # use, so it says nothing about the vendor: ordbogen once published
            # 21.1x against gpt-4o's 8.6x purely because its backend had a thread
            # pool and the other did not. Rather than publish a number that invites
            # a comparison it cannot support, API rows render an em dash.
            if entry["access"] == "proprietary":
                entry["speed_x"] = None
            # Sample-level bootstrap CIs (see scripts/compute_ci.py). Only the
            # keys relevant to this table's metrics are attached.
            ci = _bootstrap_cis().get(_slugify(name), {})
            for col in metric_cols:
                bounds = ci.get(f"{col}_ci")
                if bounds:
                    # ci.json stores [lo, hi]; the page's appendCI() reads .lo/.hi,
                    # the same shape its own bootstrap produces. Emit that shape —
                    # an array silently renders no interval at all.
                    entry[f"{col}_ci"] = {"lo": bounds[0], "hi": bounds[1]}
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
        "wer": build_rows(wer_df, wer_metrics),
        "cer": build_rows(cer_df, cer_metrics),
    }


def _load_font(name: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Vendored Roboto first; system sans only if the repo copy is unavailable."""
    candidates = [FONT_DIR / name]
    stem = {FONT_BOLD: "Bold", FONT_MEDIUM: "Bold", FONT_REGULAR: ""}[name]
    candidates += [
        Path(f"/System/Library/Fonts/Supplemental/Arial{' ' + stem if stem else ''}.ttf"),
        Path(f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-' + stem if stem else ''}.ttf"),
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(str(path), size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _tracked_width(draw: ImageDraw.ImageDraw, text: str, font, tracking: float) -> float:
    return sum(draw.textlength(c, font=font) for c in text) + tracking * (len(text) - 1)


def _tracked(draw: ImageDraw.ImageDraw, xy, text: str, font, fill, tracking: float = 0.0):
    """Draw centred text with letter-spacing (Pillow has no tracking of its own)."""
    x = xy[0] - _tracked_width(draw, text, font, tracking) / 2
    for c in text:
        draw.text((x, xy[1]), c, font=font, fill=fill, anchor="ls")
        x += draw.textlength(c, font=font) + tracking


def _mesh_gradient(size: tuple[int, int]) -> Image.Image:
    """Additive multi-source radial mesh over a near-black base."""
    width, height = size
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    out = np.empty((height, width, 3), np.float32)
    out[:] = THUMBNAIL_BG
    for fx, fy, fr, rgb, strength, falloff in THUMBNAIL_GLOWS:
        cx, cy, radius = fx * width, fy * height, fr * width
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / radius
        out += (np.clip(1.0 - dist, 0.0, 1.0) ** falloff * strength)[..., None] * np.asarray(rgb, np.float32)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def _apply_vignette(image: Image.Image) -> Image.Image:
    width, height = image.size
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    d = np.sqrt(((xx - width / 2) / (width / 2)) ** 2
                + ((yy - height / 2) / (height / 2)) ** 2) / np.sqrt(2)
    falloff = np.clip(1.0 - THUMBNAIL_VIGNETTE * d ** 2, 0.0, 1.0)[..., None]
    return Image.fromarray((np.asarray(image, np.float32) * falloff).astype(np.uint8), "RGB")


def _apply_grain(image: Image.Image, seed: int = 7) -> Image.Image:
    """Monochrome film grain, strongest in midtones.

    Applied at final resolution so it survives as 1px texture rather than being
    averaged away by the downsample. It doubles as a dither: a smooth gradient
    this dark bands visibly in 8-bit JPEG, and the noise breaks the contours up.
    """
    arr = np.asarray(image, np.float32)
    luma = (arr @ np.array([0.2126, 0.7152, 0.0722], np.float32)) / 255.0
    weight = 0.35 + 0.65 * (4.0 * luma * (1.0 - luma))
    noise = np.random.default_rng(seed).normal(0.0, THUMBNAIL_GRAIN, luma.shape).astype(np.float32)
    return Image.fromarray(np.clip(arr + (noise * weight)[..., None], 0, 255).astype(np.uint8), "RGB")


def _dannebrog(height: int, radius: int) -> Image.Image:
    """Danish flag at official 37:28 proportions, rounded with a hairline edge.

    Splits are 12:4:21 horizontally and 12:4:12 vertically, in units of 1/28 of
    the height. The cross bars are drawn full-bleed and then clipped by the
    rounded mask, so the corners stay clean at any radius.
    """
    u = height / 28.0
    width = int(round(37 * u))
    flag = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(flag)
    draw.rectangle((0, 0, width, height), fill=THUMBNAIL_RED)
    draw.rectangle((int(12 * u), 0, int(16 * u), height), fill=THUMBNAIL_TEXT)
    draw.rectangle((0, int(12 * u), width, int(16 * u)), fill=THUMBNAIL_TEXT)
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, width - 1, height - 1), radius=radius, fill=255)
    flag.putalpha(mask)
    ImageDraw.Draw(flag).rounded_rectangle(
        (0, 0, width - 1, height - 1), radius=radius,
        outline=(255, 255, 255, 60), width=max(1, height // 40))
    return flag


def _metric_card(width: int, height: int, radius: int) -> Image.Image:
    card = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    ImageDraw.Draw(card).rounded_rectangle(
        (0, 0, width - 1, height - 1), radius=radius,
        fill=(255, 255, 255, 16), outline=(255, 255, 255, 46),
        width=max(1, round(height / 90)))
    return card


TABLE_WER_EMPTY = '<table id="table-wer"></table>'


def bake_static_table(html: str, data: dict) -> str:
    """Write the current rankings into the HTML that gets served.

    The page fetches leaderboard.json and builds its table in JavaScript, so the
    served document contains no model names and no scores -- a crawler, or anyone
    whose JS has not run yet, sees an empty table. That is most of the page's
    substance missing from every index of it.

    renderTable() clears the element before rendering, so this copy costs nothing
    at runtime: it is what the page shows until the fetch resolves, and what
    search engines get. Deliberately narrower than the live table (rank, model,
    the two error rates, speed) -- enough to carry the rankings without
    reimplementing buildRow() in Python and having two renderers drift apart.
    """
    rows = data.get("wer", [])
    if not rows:
        return html

    def cell(value: object, places: int = 2) -> str:
        """Match the JS formatting: error rates to 2 places, speed to 1."""
        if value is None:
            return "&mdash;"
        if isinstance(value, float):
            return html_escape(f"{value:.{places}f}")
        return html_escape(str(value))

    out = [
        "<thead><tr>",
        '<th data-key="#">#</th>',
        '<th data-key="model">Model</th>',
        '<th data-key="mean_wer">Mean WER &darr;</th>',
        '<th data-key="mean_cer">Mean CER &darr;</th>',
        '<th data-key="speed_x">Speed (x) &uarr;</th>',
        "</tr></thead><tbody>",
    ]
    for r in rows:
        name = html_escape(str(r.get("name", "")))
        url = html_escape(str(r.get("url") or ""))
        label = (f'<a href="{url}" target="_blank" rel="noopener">{name}</a>'
                 if url else name)
        speed = r.get("speed_x")
        out.append(
            f'<tr data-n="{name.lower()}">'
            f'<td data-key="#">{cell(r.get("rank"))}</td>'
            f'<td data-key="model" class="c-model">{label}</td>'
            f'<td data-key="mean_wer">{cell(r.get("mean_wer"))}</td>'
            f'<td data-key="mean_cer">{cell(r.get("mean_cer"))}</td>'
            f'<td data-key="speed_x">{cell(speed, 1)}{"x" if speed is not None else ""}</td>'
            "</tr>"
        )
    out.append("</tbody>")

    baked = TABLE_WER_EMPTY.replace("></table>", ">" + "".join(out) + "</table>")
    if TABLE_WER_EMPTY not in html:
        raise ValueError(
            "index.html no longer contains the empty WER table this injects into; "
            f"expected {TABLE_WER_EMPTY!r}"
        )
    return html.replace(TABLE_WER_EMPTY, baked)


def _count_datasets(data: dict) -> int:
    """Distinct test sets = per-dataset WER columns (excludes the macro mean)."""
    rows = data.get("wer", [])
    if not rows:
        return 5
    n = sum(1 for k in rows[0] if k.endswith("_wer") and k != "mean_wer")
    return n or 5


def generate_cover_image(data: dict, out_path: Path = THUMBNAIL_OUT) -> Path:
    """Generate the social thumbnail. Model count, dataset count and the leading
    score come from live data; the layout is centred so it survives a square crop
    in messenger link cards."""
    ss = THUMBNAIL_SS
    width, height = THUMBNAIL_SIZE[0] * ss, THUMBNAIL_SIZE[1] * ss

    # Background is finished first, at final size: the grain has to land as 1px
    # texture, and it must not settle on the type. Artwork goes on a separate
    # transparent layer that is composited over it afterwards, so the lettering
    # stays clean while the field behind it stays gritty.
    background = _apply_grain(_apply_vignette(_mesh_gradient(THUMBNAIL_SIZE))).convert("RGBA")
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    cx = width / 2

    f_title = _load_font(FONT_BOLD, 82 * ss)
    f_sub = _load_font(FONT_REGULAR, 25 * ss)
    f_label = _load_font(FONT_MEDIUM, 16 * ss)

    flag = _dannebrog(78 * ss, radius=9 * ss)
    image.alpha_composite(flag, (int(cx - flag.width / 2), 88 * ss))

    # Tight tracking and tight leading are what make the title read as display
    # type rather than as a heading scaled up.
    for i, line in enumerate(("Open Danish ASR", "Leaderboard")):
        _tracked(draw, (cx, (268 + i * 84) * ss), line, f_title, THUMBNAIL_TEXT, tracking=-1.6 * ss)
    _tracked(draw, (cx, 400 * ss), "Consistent WER and CER evaluation across Danish test sets.",
             f_sub, THUMBNAIL_MUTED, tracking=0.2 * ss)

    cards = [(str(len(data.get("wer", []))), "MODELS"),
             (str(_count_datasets(data)), "DATASETS")]

    # Sized to sit inside the centred square-crop safe zone, so no card is cut
    # off when a messenger renders the link as a square thumbnail.
    cw, ch, gap = 200 * ss, 120 * ss, 20 * ss
    total = len(cards) * cw + (len(cards) - 1) * gap
    x0, y0 = int(cx - total / 2), 450 * ss
    for i, (value, label) in enumerate(cards):
        x = x0 + i * (cw + gap)
        image.alpha_composite(_metric_card(cw, ch, radius=18 * ss), (x, y0))
        mid = x + cw / 2
        size = 54
        f_value = _load_font(FONT_BOLD, size * ss)
        while size > 30 and _tracked_width(draw, value, f_value, -0.5 * ss) > cw - 34 * ss:
            size -= 2
            f_value = _load_font(FONT_BOLD, size * ss)
        _tracked(draw, (mid, y0 + 70 * ss), value, f_value, THUMBNAIL_STAT, tracking=-0.5 * ss)
        _tracked(draw, (mid, y0 + 100 * ss), label, f_label, THUMBNAIL_STAT_LABEL, tracking=2.2 * ss)

    # Downsample through premultiplied alpha ("RGBa"), or LANCZOS averages the
    # black of the transparent pixels into every glyph edge and haloes the type.
    background.alpha_composite(
        image.convert("RGBa").resize(THUMBNAIL_SIZE, Image.LANCZOS).convert("RGBA"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    background.convert("RGB").save(
        out_path, format="JPEG", quality=94, subsampling=0, optimize=True)
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

    # index.html is uploaded with the rankings baked in; the repo copy stays the
    # JS-only source so there is one renderer to maintain, not two.
    tmp = tempfile.TemporaryDirectory()
    baked_index = Path(tmp.name) / "index.html"
    source_html = (SPACE_DIR / "index.html").read_text(encoding="utf-8")
    baked_index.write_text(bake_static_table(source_html, data), encoding="utf-8")
    print(f"  baked {len(data['wer'])} rows into index.html "
          f"(+{len(baked_index.read_text(encoding='utf-8')) - len(source_html)} bytes)")

    print("\nUploading static files …")
    for name in UPLOAD:
        path = baked_index if name == "index.html" else SPACE_DIR / name
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
