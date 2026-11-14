"""
Audio features (MFCC) — ONE implementation for both TalkNet and SyncNet.

Both models consume identical features (13 coefficients, 100 rows/second),
so the clip's MFCC matrix is computed ONCE per wav and every consumer slices
its window out of the cached result. No ffmpeg subprocess, no temp files —
the clip's PCM wav is read directly and sliced by row math.

The cache holds ONE clip at a time (clips are processed sequentially) and is
replaced when the next clip's wav is requested — RAM frees itself naturally.
"""

from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger


class MfccExtractor:
    """Per-clip cached MFCC computation over the analysis wav.

    One shared instance serves ASD (step 4) and SyncNet (step 6): the first
    caller computes, the second slices from cache.
    """

    ROWS_PER_SECOND = 100   # winstep=0.010 — the rate TalkNet/SyncNet expect

    def __init__(self, num_coefficients: int = 13):
        self.num_coefficients = num_coefficients
        self._cached_wav_path: Optional[Path] = None
        self._cached_features: Optional[np.ndarray] = None

    def features_for_window(
        self,
        wav_path: Path,
        start_time: float,
        end_time: float,
    ) -> np.ndarray:
        """MFCC rows for [start_time, end_time] seconds of the wav.

        Returns (T, num_coefficients) at 100 rows/second, clamped to the
        available range — a slightly-too-long window never raises.
        """
        features = self._full_features(wav_path)
        start_row = max(0, int(start_time * self.ROWS_PER_SECOND))
        end_row = min(len(features), int(end_time * self.ROWS_PER_SECOND))
        return features[start_row:end_row]

    def _full_features(self, wav_path: Path) -> np.ndarray:
        """Whole-file MFCC, computed once and cached until the wav changes."""
        wav_path = Path(wav_path)
        if self._cached_wav_path == wav_path and self._cached_features is not None:
            return self._cached_features

        self._cached_features = self._compute_mfcc(wav_path)
        self._cached_wav_path = wav_path
        return self._cached_features

    def _compute_mfcc(self, wav_path: Path) -> np.ndarray:
        """Read the PCM wav and compute MFCC (overridable in unit tests)."""
        from scipy.io import wavfile
        import python_speech_features

        sample_rate, audio = wavfile.read(str(wav_path))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        features = python_speech_features.mfcc(
            audio, sample_rate,
            numcep=self.num_coefficients,
            winlen=0.025, winstep=0.010,
        )
        logger.debug(f"MFCC computed for {wav_path.name}: {features.shape[0]} rows")
        return features

    def clear_cache(self) -> None:
        self._cached_wav_path = None
        self._cached_features = None
