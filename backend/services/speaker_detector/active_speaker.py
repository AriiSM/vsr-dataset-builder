"""
Active Speaker Detection (TalkNet-ASD) — the scoring model.

Answers, for ONE track at a time: "are THESE lips producing THIS audio?"
Consumes face crops (from TrackCropReader) + MFCC slices (from
MfccExtractor) and returns per-frame scores on a 0-20 scale.

Fallback policy (Phase 0, unchanged): a missing TalkNet install or missing
weights is a HARD ERROR unless asd.allow_fallback explicitly accepts the
motion-correlation fallback — whose scores are always labeled as such.

Reference: "Is Someone Speaking? Exploring Long-term Temporal Features
for Audio-visual Active Speaker Detection" (Tao et al., 2021)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from loguru import logger


@dataclass
class ASDResult:
    """Per-track active-speaker scores for one clip window."""
    track_id: int
    start_frame: int
    end_frame: int
    scores: List[float]  # per-frame, 0-20 (higher = more likely speaking)
    # Which scoring path produced these scores (written to segments_index):
    #   "talknet"          — real TalkNet model with pretrained weights
    #   "fallback_motion"  — lip-motion/audio-energy correlation (no model)
    method: str = "unknown"

    @property
    def mean_score(self) -> float:
        return float(np.mean(self.scores)) if self.scores else 0.0


class TalkNetASD:
    """TalkNet wrapper: lazy load + score one track's crops against audio."""

    def __init__(
        self,
        model_path: Optional[Path],
        device: str,
        num_frames_context: int,
        allow_fallback: bool = False,
    ):
        self.device = device
        # Correlation window for the FALLBACK scorer only (frames)
        self.num_frames_context = num_frames_context
        self.model_path = model_path
        # When False (default), missing TalkNet install/weights is a HARD
        # ERROR instead of silently degrading to motion-correlation scores.
        self.allow_fallback = allow_fallback

        self._model = None
        self._loaded = False

    @property
    def method(self) -> str:
        return "talknet" if self._model is not None else "fallback_motion"

    def ensure_loaded(self):
        """Lazy load TalkNet.

        Raises:
            RuntimeError: if TalkNet (or its weights) is unavailable and
                allow_fallback is False (config: asd.allow_fallback).
        """
        if self._loaded:
            return

        try:
            # Requires the external TalkNet-ASD repo to be installed
            from talkNet import talkNet

            if not (self.model_path and self.model_path.exists()):
                if not self.allow_fallback:
                    raise RuntimeError(
                        f"TalkNet weights not found at {self.model_path}. "
                        "Scores from an untrained model would be meaningless. "
                        "Download the weights, or set asd.allow_fallback: true "
                        "to use the motion-correlation fallback instead."
                    )
                logger.warning(
                    "TalkNet weights missing — using motion-correlation "
                    "fallback (asd.allow_fallback is enabled)"
                )
                self._loaded = True  # fallback mode, no model
                return

            self._model = talkNet(device=self.device)
            self._model.loadParameters(str(self.model_path))
            logger.info(f"Loaded TalkNet model from {self.model_path}")
            self._model = self._model.to(self.device)
            self._model.eval()
            self._loaded = True

        except ImportError:
            if not self.allow_fallback:
                raise RuntimeError(
                    "TalkNet-ASD is not installed "
                    "(git clone https://github.com/TaoRuijie/TalkNet-ASD.git && "
                    "pip install -e TalkNet-ASD/). "
                    "Set asd.allow_fallback: true only if you accept "
                    "motion-correlation scores instead of real ASD."
                ) from None
            logger.warning(
                "TalkNet-ASD not installed — using motion-correlation "
                "fallback (asd.allow_fallback is enabled)"
            )
            self._loaded = True

    def score_crops(
        self,
        face_crops: np.ndarray,       # (T, 112, 112) grayscale
        audio_features: np.ndarray,   # (T_a, 13) MFCC at 100 rows/s
    ) -> List[float]:
        """Per-frame speaking scores (0-20) for one track's crops."""
        self.ensure_loaded()
        if self._model is not None:
            return self._score_with_model(face_crops, audio_features)
        return self._score_with_fallback(face_crops, audio_features)

    # ------------------------------------------------------- scoring paths

    def _score_with_model(
        self,
        face_crops: np.ndarray,
        audio_features: np.ndarray,
    ) -> List[float]:
        """TalkNet forward pass.

        TalkNet expects audio at 100 fps and video at 25 fps; its audio
        encoder downsamples 4×, so both sequences are trimmed to the same
        effective length before the forward pass.
        """
        effective_length = min(len(face_crops), len(audio_features) // 4)
        if effective_length < 1:
            return self._score_with_fallback(face_crops, audio_features)

        face_crops = face_crops[:effective_length]
        audio_features = audio_features[:effective_length * 4]

        visual = torch.from_numpy(face_crops).float().unsqueeze(0).to(self.device)
        audio = torch.from_numpy(audio_features).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            inner = self._model.model
            audio_embedding = inner.forward_audio_frontend(audio)
            visual_embedding = inner.forward_visual_frontend(visual)
            audio_embedding, visual_embedding = inner.forward_cross_attention(
                audio_embedding, visual_embedding
            )
            audio_visual_output = inner.forward_audio_visual_backend(
                audio_embedding, visual_embedding
            )
            # lossAV with labels=None returns the raw "speaking" logits
            raw_scores = self._model.lossAV.forward(audio_visual_output, None)

        if not isinstance(raw_scores, np.ndarray):
            raw_scores = np.array([raw_scores])
        raw_scores = raw_scores.flatten()

        # logits → sigmoid probabilities → 0-20 scale
        probabilities = 1.0 / (1.0 + np.exp(-raw_scores))
        return (probabilities * 20.0).tolist()

    def _score_with_fallback(
        self,
        face_crops: np.ndarray,
        audio_features: np.ndarray,
    ) -> List[float]:
        """Motion-correlation fallback: lip-region motion vs audio energy."""
        if len(face_crops) < 2:
            return [5.0] * len(face_crops)

        # Frame-to-frame motion in the lower half (mouth region)
        motion_scores = []
        for i in range(1, len(face_crops)):
            half = face_crops[i].shape[0] // 2
            difference = np.abs(
                face_crops[i][half:].astype(float)
                - face_crops[i - 1][half:].astype(float)
            )
            motion_scores.append(np.mean(difference))
        motion_scores = [motion_scores[0]] + motion_scores

        if audio_features is not None and len(audio_features) > 0:
            audio_energy = np.mean(np.abs(audio_features), axis=1)
            if len(audio_energy) != len(motion_scores):
                from scipy.ndimage import zoom
                audio_energy = zoom(
                    audio_energy, len(motion_scores) / len(audio_energy)
                )
        else:
            audio_energy = np.ones(len(motion_scores))

        motion_norm = (
            (motion_scores - np.mean(motion_scores))
            / (np.std(motion_scores) + 1e-6)
        )
        audio_norm = (
            (audio_energy - np.mean(audio_energy))
            / (np.std(audio_energy) + 1e-6)
        )

        window = min(self.num_frames_context, len(motion_norm))
        scores = []
        for i in range(len(motion_norm)):
            start = max(0, i - window // 2)
            end = min(len(motion_norm), i + window // 2 + 1)
            motion_slice = motion_norm[start:end]
            audio_slice = audio_norm[start:end]
            if len(motion_slice) > 1:
                correlation = np.corrcoef(motion_slice, audio_slice)[0, 1]
                score = (correlation + 1) * 10   # [-1, 1] → [0, 20]
            else:
                score = 10.0
            scores.append(max(0, min(20, score)))
        return scores
