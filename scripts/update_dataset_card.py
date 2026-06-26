#!/usr/bin/env python3
"""Regenerate the dataset card's `configs` block from the parquets on HF.

The HF dataset viewer is driven by the `configs:` key in the dataset card
(README.md) YAML frontmatter. We expose:

  - `results`  (default) -> data/results.parquet   — the leaderboard table
  - one config per model  -> outputs/<slug>.parquet — that model's raw
    transcriptions, browsable from the viewer's config dropdown

Rather than hand-maintain ~30 config entries, this script enumerates the
parquets that actually exist in the dataset repo and rebuilds the block, so a
newly-submitted model self-registers in the viewer on the next deploy. Only the
`configs` key is touched; the rest of the card (body + other frontmatter) is
preserved verbatim.

Run after push_outputs.py / push_results.py have uploaded the parquets:
  uv run --no-project --with huggingface_hub --with pyyaml python scripts/update_dataset_card.py

Requires HF_TOKEN with write access to the RyeAI org.
"""
from __future__ import annotations

import sys

import yaml
from huggingface_hub import HfApi, hf_hub_download

DATASET_REPO_ID = "RyeAI/danish-asr-leaderboard"
RESULTS_PARQUET = "data/results.parquet"


def build_configs(api: HfApi) -> list[dict]:
    """results (default) + one config per outputs/<slug>.parquet on the repo."""
    files = api.list_repo_files(DATASET_REPO_ID, repo_type="dataset")
    slugs = sorted(
        f[len("outputs/"):-len(".parquet")]
        for f in files
        if f.startswith("outputs/") and f.endswith(".parquet")
    )

    configs: list[dict] = []
    if RESULTS_PARQUET in files:
        configs.append({
            "config_name": "results",
            "data_files": [{"split": "leaderboard", "path": RESULTS_PARQUET}],
            "default": True,
        })
    for slug in slugs:
        configs.append({
            "config_name": slug,
            "data_files": [{"split": "test", "path": f"outputs/{slug}.parquet"}],
        })
    return configs


def split_frontmatter(text: str) -> tuple[str, str]:
    """Return (frontmatter_yaml, body). Empty frontmatter if none present."""
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            return text[4:end], text[end + len("\n---\n"):]
    return "", text


def main() -> None:
    api = HfApi()

    readme_path = hf_hub_download(
        repo_id=DATASET_REPO_ID, filename="README.md", repo_type="dataset"
    )
    with open(readme_path, encoding="utf-8") as fh:
        original = fh.read()

    fm_text, body = split_frontmatter(original)
    front = yaml.safe_load(fm_text) if fm_text.strip() else {}
    if not isinstance(front, dict):
        print("ERROR: could not parse README frontmatter as a mapping", file=sys.stderr)
        sys.exit(1)

    configs = build_configs(api)
    if not configs:
        print("ERROR: no parquets found in dataset repo — nothing to configure", file=sys.stderr)
        sys.exit(1)
    front["configs"] = configs

    new_fm = yaml.dump(front, sort_keys=False, allow_unicode=True, default_flow_style=False)
    new_readme = f"---\n{new_fm}---\n{body}"

    if new_readme == original:
        print("Dataset card configs already up to date — no change.")
        return

    n_models = len(configs) - 1  # minus the `results` config
    api.upload_file(
        path_or_fileobj=new_readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        commit_message=f"Refresh dataset viewer configs (results + {n_models} models)",
    )
    print(f"Updated dataset card: results + {n_models} per-model configs.")


if __name__ == "__main__":
    main()
