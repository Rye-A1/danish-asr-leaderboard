"""Re-scoring published Parquets retains metadata and the core-5 score rule."""

import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from rescore_published import rescore_result
from danish_asr_leaderboard.datasets import CORE_COLUMNS


def test_rescore_published_preserves_metadata_and_uses_new_normalizer():
    rows = [
        {"dataset": dataset, "reference": "hundrede procents sikkerhed",
         "hypothesis": "100% sikkerhed"}
        for dataset in CORE_COLUMNS
    ]
    rows[0].update(reference="2.4ghz- og 5.0ghz-frekvenserne",
                   hypothesis="2,4 GHz og 5,00 GHz frekvenserne")
    original = {"model": "Example/model", "submitted": "2026-01-01",
                "speed_x": 12.5, "mean_wer": 99.0}
    updated = rescore_result(original, pd.DataFrame(rows))
    assert updated["mean_wer"] == updated["mean_cer"] == 0.0
    assert updated["speed_x"] == 12.5
    assert updated["submitted"] == "2026-01-01"
    assert original["mean_wer"] == 99.0


def test_rescore_published_rejects_partial_snapshot():
    with pytest.raises(ValueError, match="missing core dataset"):
        rescore_result({}, pd.DataFrame([
            {"dataset": CORE_COLUMNS[0], "reference": "hej", "hypothesis": "hej"}
        ]))
