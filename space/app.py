import os
import re

import gradio as gr
import pandas as pd
from gradio_leaderboard import Leaderboard
from huggingface_hub import login

DATASET = "RyeAI/danish-asr-leaderboard"

_token = os.environ.get("HF_TOKEN")
if _token:
    login(token=_token, add_to_git_credential=False)

COL_MAP = {
    "model":                  "Model",
    "params_b":               "Size (B)",
    "mean_wer":               "Mean WER ↓",
    "mean_cer":               "Mean CER ↓",
    "speed_x":                "Speed (x) ↑",
    "coral_conversation_wer": "CoRal Conv.",
    "coral_read_aloud_wer":   "CoRal Read",
    "ftspeech_wer":           "FTSpeech",
    "cv17_da_wer":            "CV17 (da)",
    "fleurs_da_wer":          "FLEURS (da)",
    "submitted":              "Submitted",
}

CER_COL_MAP = {
    "model":                  "Model",
    "params_b":               "Size (B)",
    "access":                 "Access",
    "mean_cer":               "Mean CER ↓",
    "coral_conversation_cer": "CoRal Conv. CER",
    "coral_read_aloud_cer":   "CoRal Read CER",
    "ftspeech_cer":           "FTSpeech CER",
    "cv17_da_cer":            "CV17 (da) CER",
    "fleurs_da_cer":          "FLEURS (da) CER",
    "submitted":              "Submitted",
}

# "#" is "str" so it can hold medals (🥇🥈🥉); "Model" is markdown (HTML links).
ALL_COLS  = ["#"] + list(COL_MAP.values())
HIDE_COLS = ["Submitted"]
DATATYPES = ["str", "markdown", "number", "number", "number", "number", "number",
             "number", "number", "number", "number", "str"]
WIDTHS    = ["46px", "240px", "62px", "90px", "90px", "88px", "92px",
             "88px", "82px", "80px", "92px", "90px"]

CER_ALL_COLS  = ["#"] + list(CER_COL_MAP.values())
CER_HIDE_COLS = ["Submitted"]
CER_DATATYPES = ["str", "markdown", "number", "str", "number", "number", "number",
                 "number", "number", "number", "str"]
CER_WIDTHS    = ["46px", "240px", "62px", "88px", "90px", "120px", "118px",
                 "112px", "108px", "120px", "90px"]


_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
_MEDALS = {1: "🥇", 2: "🥈", 3: "🥉"}


def _linkify(value):
    """Turn a markdown link into an HTML anchor that opens in a new tab."""
    if not isinstance(value, str):
        return value
    m = _MD_LINK_RE.fullmatch(value.strip())
    if not m:
        return value
    text, url = m.group(1), m.group(2)
    return f'<a href="{url}" target="_blank" rel="noopener noreferrer">{text}</a>'


def _rank_column(n: int) -> list[str]:
    """Rank labels: medals for the top three, plain numbers afterwards."""
    return [_MEDALS.get(i, str(i)) for i in range(1, n + 1)]


def _base_df() -> pd.DataFrame:
    df = pd.read_parquet(f"hf://datasets/{DATASET}/data/results.parquet")
    if "rtf" in df.columns and "speed_x" not in df.columns:
        df["speed_x"] = (1.0 / df["rtf"]).round(1)
        df = df.drop(columns=["rtf"])
    if "access" not in df.columns:
        df["access"] = "open"  # backwards compat
    return df


def load_results() -> pd.DataFrame:
    try:
        df = _base_df()
        df = df.rename(columns=COL_MAP)
        for col in list(COL_MAP.values()):
            if col not in df.columns:
                df[col] = None
        df = (df[list(COL_MAP.values())]
              .sort_values("Mean WER ↓", ascending=True, na_position="last")
              .reset_index(drop=True))
        df["Model"] = df["Model"].map(_linkify)
        df.insert(0, "#", _rank_column(len(df)))
        return df[ALL_COLS]
    except Exception:
        return pd.DataFrame(columns=ALL_COLS)


def load_cer_results() -> pd.DataFrame:
    try:
        df = _base_df()
        df = df.rename(columns=CER_COL_MAP)
        for col in list(CER_COL_MAP.values()):
            if col not in df.columns:
                df[col] = None
        df = (df[list(CER_COL_MAP.values())]
              .sort_values("Mean CER ↓", ascending=True, na_position="last")
              .reset_index(drop=True))
        df["Model"] = df["Model"].map(_linkify)
        df.insert(0, "#", _rank_column(len(df)))
        return df[CER_ALL_COLS]
    except Exception:
        return pd.DataFrame(columns=CER_ALL_COLS)


_TOP_N = 5
_step = (0.10 - 0.02) / (_TOP_N - 1)
_gradient_rows = "\n".join(
    f"#lb-col table tbody tr:nth-child({i}) td {{ background-color: rgba(34, 197, 94, {0.10 - (i-1)*_step:.3f}) !important; }}"
    for i in range(1, _TOP_N + 1)
)
_gradient_cer = "\n".join(
    f"#cer-col table tbody tr:nth-child({i}) td {{ background-color: rgba(34, 197, 94, {0.10 - (i-1)*_step:.3f}) !important; }}"
    for i in range(1, _TOP_N + 1)
)

CSS = f"""
footer {{ display: none !important; }}
{_gradient_rows}
{_gradient_cer}
/* About tab: no gradient */
#about-col table td {{ background-color: transparent !important; }}
/* Disable sort on # and Model columns */
#lb-col table thead tr th:nth-child(1),
#lb-col table thead tr th:nth-child(2),
#cer-col table thead tr th:nth-child(1),
#cer-col table thead tr th:nth-child(2) {{ pointer-events: none !important; cursor: default !important; }}
#lb-col table thead tr th:nth-child(1) span, #lb-col table thead tr th:nth-child(1) button, #lb-col table thead tr th:nth-child(1) svg,
#lb-col table thead tr th:nth-child(2) span, #lb-col table thead tr th:nth-child(2) button, #lb-col table thead tr th:nth-child(2) svg,
#cer-col table thead tr th:nth-child(1) span, #cer-col table thead tr th:nth-child(1) button, #cer-col table thead tr th:nth-child(1) svg,
#cer-col table thead tr th:nth-child(2) span, #cer-col table thead tr th:nth-child(2) button, #cer-col table thead tr th:nth-child(2) svg
{{ pointer-events: none !important; }}
/* Hide search box rendered by gradio_leaderboard */
#lb-col .block:has(textarea[placeholder*="Separate"]),
#lb-col .block:has(input[placeholder*="Separate"]),
#cer-col .block:has(textarea[placeholder*="Separate"]),
#cer-col .block:has(input[placeholder*="Separate"]) {{ display: none !important; }}
/* Scrollable tables */
#lb-col, #cer-col {{ overflow-x: auto; }}
/* Fit columns to content instead of stretching to full width */
#lb-col table, #cer-col table {{ width: max-content !important; min-width: 0 !important; }}
/* Cover banner */
#cover {{ max-width: 1000px; margin: 0 auto 0.75rem auto; }}
#cover img {{ width: 100%; height: auto; border-radius: 14px; display: block; }}
#cover button {{ display: none !important; }}
"""

ABOUT_MD = """
## Test sets

| Column | Dataset | Split | Domain |
|--------|---------|-------|--------|
| **CoRal Conv.** | [CoRal-project/coral-v3](https://huggingface.co/datasets/CoRal-project/coral-v3) — conversation | test | Spontaneous conversation |
| **CoRal Read** | [CoRal-project/coral-v3](https://huggingface.co/datasets/CoRal-project/coral-v3) — read_aloud | test | Read-aloud speech |
| **CV17 (da)** | [mozilla-foundation/common_voice_17_0](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) — da | test | Crowd-sourced read speech |
| **FLEURS (da)** | [google/fleurs](https://huggingface.co/datasets/google/fleurs) — da_dk | test | Read speech |
| **FTSpeech** | [alexandrainst/ftspeech](https://huggingface.co/datasets/alexandrainst/ftspeech) | test_balanced | Parliamentary / broadcast |

## Methodology

### Text normalisation
Applied identically to hypothesis and reference before scoring:
1. Unicode NFC
2. Number formatting — digit separators removed so a numeral scores identically regardless of formatting: thousand separators (e.g. `1.234` → `1234`) and decimal separators (e.g. `3,14` and `3.14` → `314`)
3. Lowercase
4. Strip punctuation (all punctuation including apostrophes)
5. Collapse whitespace

This matches the normalization applied to training targets and is broadly consistent with the [HF Open ASR Leaderboard](https://github.com/huggingface/open_asr_leaderboard) (`BasicTextNormalizer`), with the addition of Danish digit formatting.

> **Note:** digit–word equivalence (e.g. `"4"` vs `"fire"`) is **not** normalised. Both forms are accepted as correct by human transcribers, but a model that consistently outputs one form when the reference uses the other will incur errors. This is a known limitation shared by most public ASR leaderboards; a robust fix requires a domain-specific Danish number-to-word converter that handles edge cases (years, phone numbers, ordinals) correctly.

### Metric
**WER** (Word Error Rate, %) and **CER** (Character Error Rate, %) — lower is better.

`WER = (substitutions + deletions + insertions) / reference_words × 100`

`CER = (char substitutions + deletions + insertions) / reference_chars × 100`

**Mean WER** and **Mean CER** are macro-averages across all five test sets (equally weighted).

### Speed
**Speed (x)** = total audio duration / total inference time.
A value of 30x means 30 seconds of audio processed per wall-clock second.
Measured on an NVIDIA RTX Pro 5000 Blackwell (48 GB) — not directly comparable across hardware.

### Code & data
Results: [RyeAI/danish-asr-leaderboard](https://huggingface.co/datasets/RyeAI/danish-asr-leaderboard)
Eval code: [github.com/Rye-A1/danish-asr-leaderboard](https://github.com/Rye-A1/danish-asr-leaderboard)

> Want your model added? Open a [discussion](https://huggingface.co/spaces/RyeAI/danish-asr-leaderboard/discussions).
"""

with gr.Blocks(css=CSS, title="Danish ASR Leaderboard") as demo:
    gr.Image(
        "cover.jpeg",
        show_label=False,
        container=False,
        interactive=False,
        show_download_button=False,
        show_fullscreen_button=False,
        elem_id="cover",
    )
    gr.Markdown(
        "Benchmarking Danish ASR models — open-source and proprietary — on five independent test sets. "
        "**Lower WER is better.** Mean WER is macro-averaged across all five sets. "
        "**Speed (x)** = times faster than real-time (network-bound for API models)."
    )

    with gr.Tabs():
        with gr.Tab("\U0001f3c6 Leaderboard"):
            with gr.Column(elem_id="lb-col"):
                leaderboard = Leaderboard(
                    value=load_results(),
                    search_columns=["Model"],
                    hide_columns=HIDE_COLS,
                    datatype=DATATYPES,
                    column_widths=WIDTHS,
                )
            refresh_btn = gr.Button("\U0001f504 Refresh results", size="sm", variant="secondary")
            refresh_btn.click(fn=load_results, outputs=leaderboard)

        with gr.Tab("\U0001f524 CER Breakdown"):
            with gr.Column(elem_id="cer-col"):
                cer_leaderboard = Leaderboard(
                    value=load_cer_results(),
                    search_columns=["Model"],
                    hide_columns=CER_HIDE_COLS,
                    datatype=CER_DATATYPES,
                    column_widths=CER_WIDTHS,
                )
            cer_refresh_btn = gr.Button("\U0001f504 Refresh results", size="sm", variant="secondary")
            cer_refresh_btn.click(fn=load_cer_results, outputs=cer_leaderboard)

        with gr.Tab("ℹ️ About"):
            with gr.Column(elem_id="about-col"):
                gr.Markdown(ABOUT_MD)

demo.launch()
