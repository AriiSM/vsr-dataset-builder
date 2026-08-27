"""
Mouth Landmarks (MediaPipe FaceMesh)

Dense lip localization for the 96×96 mouth crop. Replaces the 2-point
RetinaFace mouth estimate (corners only, every Nth frame) with 40+ lip
points on EVERY frame, plus a head-pose proxy (roll/yaw/pitch) used for
crop alignment and quality metadata.

Runs on CPU (~3-6 ms/frame on a face-sized ROI) — does not compete with
the GPU stages.

Also contains OneEuroFilter: a low-lag smoothing filter for the mouth-center
trajectory (Gaussian smoothing lags behind quick head turns; One-Euro adapts
its cutoff to speed).
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from loguru import logger


# FaceMesh topology (468 landmarks) — canonical lip + eye indices.
# Outer lip contour (enough coverage to center and size the crop):
_OUTER_LIP_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    409, 270, 269, 267, 0, 37, 39, 40, 185,
]
# Inner lip ring adds robustness when the mouth is wide open:
_INNER_LIP_INDICES = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    415, 310, 311, 312, 13, 82, 81, 80, 191,
]
_LIP_INDICES = _OUTER_LIP_INDICES + _INNER_LIP_INDICES
_MOUTH_CORNER_LEFT = 61    # subject's right, image left
_MOUTH_CORNER_RIGHT = 291
_EYE_OUTER_LEFT = 33
_EYE_OUTER_RIGHT = 263
_NOSE_TIP = 1


@dataclass
class LipLandmarks:
    """Per-frame lip geometry in FULL-FRAME pixel coordinates."""
    mouth_center: Tuple[float, float]
    mouth_width: float                 # corner-to-corner distance (pixels)
    roll_degrees: float                # head roll from the eye line
    yaw_proxy: float                   # nose offset / inter-ocular (≈0 frontal)
    pitch_proxy: float                 # vertical nose offset / inter-ocular


class OneEuroFilter:
    """One-Euro filter (Casiez et al., CHI 2012) for one scalar signal.

    Smooths jitter at low speeds without lagging on fast movement — exactly
    the trade-off a mouth-center trajectory needs during head turns.
    """

    def __init__(self, frequency: float = 25.0, min_cutoff: float = 1.0,
                 beta: float = 0.3, derivative_cutoff: float = 1.0):
        self.frequency = frequency
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff
        self._previous_value: Optional[float] = None
        self._previous_derivative = 0.0

    @staticmethod
    def _alpha(cutoff: float, frequency: float) -> float:
        time_constant = 1.0 / (2.0 * math.pi * cutoff)
        period = 1.0 / frequency
        return 1.0 / (1.0 + time_constant / period)

    def filter(self, value: float) -> float:
        if self._previous_value is None:
            self._previous_value = value
            return value

        derivative = (value - self._previous_value) * self.frequency
        alpha_d = self._alpha(self.derivative_cutoff, self.frequency)
        smoothed_derivative = (
            alpha_d * derivative + (1.0 - alpha_d) * self._previous_derivative
        )

        cutoff = self.min_cutoff + self.beta * abs(smoothed_derivative)
        alpha = self._alpha(cutoff, self.frequency)
        smoothed = alpha * value + (1.0 - alpha) * self._previous_value

        self._previous_value = smoothed
        self._previous_derivative = smoothed_derivative
        return smoothed


class MouthLandmarker:
    """MediaPipe FaceMesh wrapper producing LipLandmarks per frame.

    Detection runs on the (already tracked and smoothed) face ROI rather than
    the full frame — faster, and immune to other faces in the shot. Results
    are mapped back to full-frame pixel coordinates.
    """

    def __init__(self, min_detection_confidence: float = 0.5):
        self.min_detection_confidence = min_detection_confidence
        self._face_mesh = None

    def _load(self):
        if self._face_mesh is not None:
            return
        try:
            import mediapipe as mp
        except ImportError:
            raise RuntimeError(
                "mediapipe is not installed but mouth_roi.method is 'mediapipe'. "
                "Install it (pip install mediapipe) or set mouth_roi.method: "
                "'retinaface' to use the legacy 2-point mouth estimate."
            ) from None

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=0.5,
        )
        logger.info("MediaPipe FaceMesh loaded for mouth ROI (CPU)")

    def detect(
        self,
        frame_bgr: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[LipLandmarks]:
        """Detect lip landmarks in a frame.

        Args:
            frame_bgr: full frame (BGR).
            roi: optional (x, y, width, height) face region to search in;
                 results are mapped back to full-frame coordinates.

        Returns:
            LipLandmarks, or None when no face is found in the ROI.
        """
        self._load()

        frame_height, frame_width = frame_bgr.shape[:2]
        if roi is not None:
            x, y, w, h = roi
            x = max(0, x); y = max(0, y)
            w = min(w, frame_width - x); h = min(h, frame_height - y)
            if w <= 0 or h <= 0:
                return None
            search_image = frame_bgr[y:y + h, x:x + w]
            offset = (x, y)
        else:
            search_image = frame_bgr
            offset = (0, 0)

        # FaceMesh expects RGB
        rgb = search_image[:, :, ::-1]
        result = self._face_mesh.process(np.ascontiguousarray(rgb))
        if not result.multi_face_landmarks:
            return None

        landmarks = result.multi_face_landmarks[0].landmark
        roi_height, roi_width = search_image.shape[:2]

        def to_pixels(index: int) -> Tuple[float, float]:
            lm = landmarks[index]
            return (lm.x * roi_width + offset[0], lm.y * roi_height + offset[1])

        lip_points = np.array([to_pixels(i) for i in _LIP_INDICES])
        mouth_center = tuple(lip_points.mean(axis=0))

        corner_left = np.array(to_pixels(_MOUTH_CORNER_LEFT))
        corner_right = np.array(to_pixels(_MOUTH_CORNER_RIGHT))
        mouth_width = float(np.linalg.norm(corner_right - corner_left))

        eye_left = np.array(to_pixels(_EYE_OUTER_LEFT))
        eye_right = np.array(to_pixels(_EYE_OUTER_RIGHT))
        eye_delta = eye_right - eye_left
        roll_degrees = math.degrees(math.atan2(eye_delta[1], eye_delta[0]))

        inter_ocular = float(np.linalg.norm(eye_delta)) or 1.0
        eye_midpoint = (eye_left + eye_right) / 2.0
        nose = np.array(to_pixels(_NOSE_TIP))
        yaw_proxy = float((nose[0] - eye_midpoint[0]) / inter_ocular)
        pitch_proxy = float((nose[1] - eye_midpoint[1]) / inter_ocular)

        return LipLandmarks(
            mouth_center=(float(mouth_center[0]), float(mouth_center[1])),
            mouth_width=mouth_width,
            roll_degrees=roll_degrees,
            yaw_proxy=yaw_proxy,
            pitch_proxy=pitch_proxy,
        )
