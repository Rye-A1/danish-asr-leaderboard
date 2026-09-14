import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from snapshot_hf_downloads import model_repositories, upsert_snapshot


def test_upsert_snapshot_replaces_today_and_preserves_prior_values():
    history = {
        "version": 1,
        "snapshots": [
            {"date": "2026-09-13", "downloads": {"example/asr": 100}},
            {"date": "2026-09-14", "downloads": {"example/asr": 110}},
        ],
    }

    updated = upsert_snapshot(history, "2026-09-14", {"example/asr": 120, "other/asr": 5})

    assert updated["snapshots"] == [
        {"date": "2026-09-13", "downloads": {"example/asr": 100}},
        {"date": "2026-09-14", "downloads": {"example/asr": 120, "other/asr": 5}},
    ]


def test_model_repositories_skips_proprietary_rows(tmp_path):
    (tmp_path / "open.json").write_text(
        '{"model":"[example/asr](https://huggingface.co/example/asr)","access":"open"}',
        encoding="utf-8",
    )
    (tmp_path / "proprietary.json").write_text(
        '{"model":"[vendor/api](https://huggingface.co/vendor/api)","access":"proprietary"}',
        encoding="utf-8",
    )

    assert model_repositories(tmp_path) == {"example/asr": "example/asr"}