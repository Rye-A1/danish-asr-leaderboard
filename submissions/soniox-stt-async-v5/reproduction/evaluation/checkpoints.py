"""Download pinned ESPnet exports, or load an exported local checkpoint."""

import hashlib
import json
from pathlib import Path

import yaml
from huggingface_hub import HfApi, hf_hub_download

PINS = json.loads(Path(__file__).with_name("pins.json").read_text(encoding="utf-8"))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve(model, cache, revision=None, kind=None):
    """Local directories contain model.pth, config.yaml and referenced assets."""
    local = Path(model).is_dir()
    if local:
        root = Path(model).resolve()
        info = {"local": str(root), "kind": kind or "seq2seq"}
        weight, config = root / "model.pth", root / "config.yaml"

        def fetch(name):
            return str((root / name).resolve())
    else:
        known = PINS["models"].get(model) or next(
            (pin for pin in PINS["models"].values() if pin["repo"] == model), None
        )
        info = dict(known or {"repo": model, "kind": kind or "seq2seq"})
        info["revision"] = revision or info.get("revision")
        if not info["revision"] or len(info["revision"]) != 40:
            info["revision"] = HfApi().model_info(info["repo"], revision=info["revision"]).sha

        def fetch(name):
            return hf_hub_download(
                info["repo"], name, revision=info["revision"], cache_dir=str(cache / "hub")
            )

        meta = yaml.safe_load(Path(fetch("meta.yaml")).read_text(encoding="utf-8"))
        config = Path(fetch(meta["yaml_files"]["s2t_train_config"]))
        weight = Path(fetch(meta["files"]["s2t_model_file"]))
    config_data = yaml.safe_load(config.read_text(encoding="utf-8"))
    bpe = fetch(config_data["bpemodel"])
    assets = {"model": sha256(weight), "config": sha256(config), "bpe": sha256(bpe)}
    if config_data.get("normalize") == "global_mvn":
        stats = fetch(config_data["normalize_conf"]["stats_file"])
        assets["stats"] = sha256(stats)
        config_data["normalize_conf"]["stats_file"] = stats
    config_data["bpemodel"] = bpe
    target = cache / "configs" / (assets["model"][:16] + ".yaml")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config_data, allow_unicode=True), encoding="utf-8")
    info["sha256"] = assets
    return {"s2t_train_config": str(target), "s2t_model_file": str(weight), "bpemodel": bpe}, info
