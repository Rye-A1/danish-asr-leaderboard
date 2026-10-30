"""Reproducible benchmark and leaderboard for Danish automatic speech recognition.

Public API:
    from danish_asr_leaderboard import DatasetConfig
    from danish_asr_leaderboard.normalizer import normalise
    from danish_asr_leaderboard.metrics import compute_wer, compute_cer
"""

from .custom_datasets import DatasetConfig as DatasetConfig

__version__ = "0.1.0"
