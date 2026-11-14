"""
Audio-visual sync verification (SyncNet).

Runs ONLY on the winning track (after SpeakerSelector decided who speaks)
and measures whether that person's lips and the audio are aligned in time.
The result is QUALITY METADATA (feeds quality_tier), never a hard filter.

Memory bound: analysis is capped to a window of max_analysis_seconds from
the MIDDLE of the segment (config syncnet.max_analysis_seconds). A/V desync
does not vary within one clip, so the measurement is unchanged while RAM
falls from ~260 MB to ~65 MB on 60-second clips.

Fallback policy (Phase 0, unchanged): missing/broken weights are a HARD
ERROR unless syncnet.allow_fallback explicitly accepts correlation scores.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from loguru import logger

from services.speaker_detector.audio_features import MfccExtractor
from services.speaker_detector.syncnet_model import SyncNetModel
from services.speaker_detector.track_crops import CropRequest, TrackCropReader

# SyncNet's trained input specification
SYNC_CROP_SIZE = (240, 240)
VIDEO_FRAMES_PER_WINDOW = 5     # 0.2 s of video at 25 fps
AUDIO_ROWS_PER_WINDOW = 20      # 0.2 s of MFCC at 100 rows/s
BATCH_SIZE = 32                 # windows per GPU forward


@dataclass
class SyncResult:
    """Audio-visual synchronization verdict for one segment."""
    track_id: int
    confidence: float    # 0-1, higher = better sync
    offset_frames: int   # estimated A/V offset (positive = audio ahead)
    # Which scoring path produced this confidence (written to segments_index):
    #   "syncnet"               — real model with pretrained weights
    #   "fallback_correlation"  — simple correlation, no model
    #   "insufficient_data"     — clip too short to measure anything
    method: str = "unknown"


class SyncNetVerifier:
    """Lazy-loaded SyncNet + windowed A/V correlation measurement."""

    def __init__(
        self,
        model_path: Optional[Path],
        device: str,
        max_offset_frames: int,
        allow_fallback: bool = False,
        max_analysis_seconds: float = 10.0,
        mfcc_extractor: Optional[MfccExtractor] = None,
        crop_reader: Optional[TrackCropReader] = None,
    ):
        self.model_path = model_path
        self.device = device
        self.max_offset_frames = max_offset_frames
        # When False (default), missing/broken weights are a HARD ERROR
        # instead of silently degrading to correlation scores.
        self.allow_fallback = allow_fallback
        self.max_analysis_seconds = max_analysis_seconds
        self.mfcc_extractor = mfcc_extractor or MfccExtractor()
        self.crop_reader = crop_reader or TrackCropReader()

        self._model = None
        self._loaded = False

    # ------------------------------------------------------------ lifecycle

    def _ensure_model_loaded(self):
        """Lazy load SyncNet weights (config: syncnet.allow_fallback)."""
        if self._loaded:
            return

        if not (self.model_path and self.model_path.exists()):
            if not self.allow_fallback:
                raise RuntimeError(
                    f"SyncNet weights not found at {self.model_path}. "
                    "Confidence from an untrained model would be meaningless. "
                    "Download the weights, or set syncnet.allow_fallback: true "
                    "to use the simple correlation fallback instead."
                )
            logger.warning(
                "SyncNet weights missing — using correlation fallback "
                "(syncnet.allow_fallback is enabled)"
            )
            self._loaded = True
            return

        try:
            self._model = SyncNetModel()
            state_dict = torch.load(self.model_path, map_location=self.device)
            self._model.load_state_dict(state_dict)
            self._model = self._model.to(self.device)
            self._model.eval()
            logger.info(f"Loaded SyncNet model from {self.model_path}")
        except Exception as e:
            self._model = None
            if not self.allow_fallback:
                raise RuntimeError(
                    f"SyncNet weights at {self.model_path} failed to load: {e}. "
                    "Fix the weights file, or set syncnet.allow_fallback: true."
                ) from e
            logger.warning(
                f"Could not load SyncNet model ({e}) — using correlation "
                "fallback (syncnet.allow_fallback is enabled)"
            )
        self._loaded = True

    # ------------------------------------------------------------ public API

    def verify_sync(
        self,
        video_path: Path,
        audio_path: Path,
        track,
        speech_start: float,
        speech_end: float,
        fps: float = 25.0,
    ) -> SyncResult:
        """Measure A/V alignment for the winning track's segment.

        Times are clip-file seconds. Analysis is bounded to
        max_analysis_seconds from the middle of the segment.
        """
        self._ensure_model_loaded()

        window_start, window_end = self._analysis_window(speech_start, speech_end)

        crops_by_track = self.crop_reader.read(
            video_path,
            [CropRequest(
                track=track,
                start_frame=int(window_start * fps),
                end_frame=int(window_end * fps),
            )],
            target_size=SYNC_CROP_SIZE, grayscale=False, bbox_margin=0.0,
        )
        face_crops = crops_by_track.get(track.track_id, np.empty(0))

        if len(face_crops) < VIDEO_FRAMES_PER_WINDOW:
            return SyncResult(track.track_id, 0.0, 0, method="insufficient_data")

        audio_features = self.mfcc_extractor.features_for_window(
            audio_path, window_start, window_end,
        )
        if len(audio_features) < VIDEO_FRAMES_PER_WINDOW:
            return SyncResult(track.track_id, 0.0, 0, method="insufficient_data")

        offsets = list(range(-self.max_offset_frames * 3,
                             self.max_offset_frames * 3 + 1))
        if self._model is not None:
            scores = self._correlate_with_model(face_crops, audio_features, offsets)
            method = "syncnet"
        else:
            scores = self._correlate_fallback(face_crops, audio_features, offsets)
            method = "fallback_correlation"

        return self._result_from_scores(track.track_id, scores, method)

    def _analysis_window(self, start: float, end: float) -> tuple:
        """Middle max_analysis_seconds of the segment (whole segment if shorter)."""
        duration = end - start
        if duration <= self.max_analysis_seconds:
            return start, end
        middle = (start + end) / 2.0
        half = self.max_analysis_seconds / 2.0
        return middle - half, middle + half

    def _result_from_scores(
        self, track_id: int, scores: np.ndarray, method: str
    ) -> SyncResult:
        """Best-offset confidence, mapped into [0, 1]."""
        center = len(scores) // 2
        best_index = int(np.argmax(scores))
        offset = best_index - center
        raw = float(scores[best_index])
        confidence = (raw + 1.0) / 2.0 if -1.0 <= raw <= 1.0 else max(0.0, min(1.0, raw))
        return SyncResult(track_id, confidence, offset, method=method)

    # ------------------------------------------------------- scoring paths

    def _correlate_with_model(
        self,
        face_crops: np.ndarray,       # (T, 240, 240, 3) BGR
        audio_features: np.ndarray,   # (T_a, 13)
        offsets: List[int],
    ) -> np.ndarray:
        """SyncNet cosine similarity at each A/V offset (batched windows)."""
        scores = []
        for offset in offsets:
            video_offset = max(offset, 0)
            audio_offset = max(-offset * 4, 0)

            usable_video = len(face_crops) - video_offset
            usable_audio = len(audio_features) - audio_offset
            if (usable_video < VIDEO_FRAMES_PER_WINDOW
                    or usable_audio < AUDIO_ROWS_PER_WINDOW):
                scores.append(0.0)
                continue

            steps = min(usable_video // VIDEO_FRAMES_PER_WINDOW,
                        usable_audio // AUDIO_ROWS_PER_WINDOW)

            video_windows, audio_windows = [], []
            for i in range(steps):
                video_slice = face_crops[
                    video_offset + i * VIDEO_FRAMES_PER_WINDOW:
                    video_offset + (i + 1) * VIDEO_FRAMES_PER_WINDOW
                ]
                audio_slice = audio_features[
                    audio_offset + i * AUDIO_ROWS_PER_WINDOW:
                    audio_offset + (i + 1) * AUDIO_ROWS_PER_WINDOW
                ]
                if (len(video_slice) < VIDEO_FRAMES_PER_WINDOW
                        or len(audio_slice) < AUDIO_ROWS_PER_WINDOW):
                    break
                video_windows.append(
                    torch.from_numpy(video_slice).float().permute(3, 0, 1, 2) / 255.0
                )
                audio_windows.append(
                    torch.from_numpy(audio_slice.T).float().unsqueeze(0)
                )

            if not video_windows:
                scores.append(0.0)
                continue

            similarities = []
            for batch_start in range(0, len(video_windows), BATCH_SIZE):
                video_batch = torch.stack(
                    video_windows[batch_start:batch_start + BATCH_SIZE]
                ).to(self.device)
                audio_batch = torch.stack(
                    audio_windows[batch_start:batch_start + BATCH_SIZE]
                ).to(self.device)
                with torch.no_grad():
                    batch_similarities = self._model(video_batch, audio_batch)
                similarities.extend(
                    batch_similarities.detach().cpu().numpy().tolist()
                )

            scores.append(float(np.mean(similarities)) if similarities else 0.0)

        return np.array(scores)

    def _correlate_fallback(
        self,
        face_crops: np.ndarray,
        audio_features: np.ndarray,
        offsets: List[int],
    ) -> np.ndarray:
        """Simple correlation fallback: mouth motion vs audio energy."""
        if len(face_crops) < 2:
            return np.zeros(len(offsets))

        motion = []
        for i in range(1, len(face_crops)):
            difference = np.abs(
                face_crops[i].astype(float) - face_crops[i - 1].astype(float)
            )
            motion.append(np.mean(difference))
        motion = np.array([motion[0]] + motion)

        audio_energy = np.sqrt(np.sum(audio_features ** 2, axis=1))
        if len(audio_energy) != len(motion):
            from scipy.ndimage import zoom
            audio_energy = zoom(audio_energy, len(motion) / len(audio_energy))

        motion = (motion - np.mean(motion)) / (np.std(motion) + 1e-6)
        audio_energy = (
            (audio_energy - np.mean(audio_energy)) / (np.std(audio_energy) + 1e-6)
        )

        scores = []
        for offset in offsets:
            if offset >= 0:
                motion_slice = motion[offset:]
                audio_slice = audio_energy[:len(motion_slice)]
            else:
                audio_slice = audio_energy[-offset:]
                motion_slice = motion[:len(audio_slice)]

            if len(motion_slice) > 1:
                correlation = np.corrcoef(motion_slice, audio_slice)[0, 1]
                scores.append((correlation + 1) / 2)
            else:
                scores.append(0.5)

        return np.array(scores)
