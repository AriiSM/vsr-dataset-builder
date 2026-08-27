"""
Face detection (RetinaFace ResNet50).

Finds every face in a frame with a confidence score and 5 landmarks.
The model loads LAZILY with a clear environment error, and video scanning
uses SEQUENTIAL decoding: on H.264, per-frame seeking decodes from the
previous keyframe anyway (slower) and can return the wrong frame on some
backends — reading linearly with cheap grab() on skipped frames is both
faster and guaranteed exact.
"""

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from loguru import logger

from services.face_tracker.geometry import BoundingBox, FaceDetection


class RetinaFaceDetector:
    """RetinaFace wrapper: single-image detection + whole-clip scanning."""

    MODEL_NAME = "resnet50_2020-07-20"
    MODEL_MAX_SIZE = 640

    def __init__(
        self,
        confidence_threshold: float,
        nms_threshold: float,
        gpu_id: int = 0,
    ):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.gpu_id = gpu_id
        self._model = None

    def _ensure_model_loaded(self):
        """Load RetinaFace on first use, with a clear error when missing."""
        if self._model is not None:
            return
        try:
            import torch
            from retinaface.pre_trained_models import get_model
        except ImportError:
            raise RuntimeError(
                "retinaface-pytorch is not installed (NOTE: the package is "
                "'retinaface-pytorch', not 'retina-face'). "
                "Install with: pip install retinaface-pytorch"
            ) from None

        device = (
            f"cuda:{self.gpu_id}"
            if self.gpu_id >= 0 and torch.cuda.is_available()
            else "cpu"
        )
        self._model = get_model(self.MODEL_NAME, max_size=self.MODEL_MAX_SIZE,
                                device=device)
        self._model.eval()
        logger.info(f"RetinaFace detector initialized on {device}")

    def detect_faces(self, frame_bgr: np.ndarray, frame_idx: int = 0) -> List[FaceDetection]:
        """Detect all faces in one BGR frame."""
        self._ensure_model_loaded()

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        raw_faces = self._model.predict_jsons(
            rgb_frame,
            confidence_threshold=self.confidence_threshold,
            nms_threshold=self.nms_threshold,
        )

        detections = []
        for face in raw_faces:
            if not face.get('bbox') or len(face['bbox']) < 4:
                continue
            score = float(face['score'])
            x1, y1, x2, y2 = face['bbox']

            landmarks = None
            if face.get('landmarks'):
                landmarks = np.array(face['landmarks'], dtype=np.float32)

            detections.append(FaceDetection(
                frame_idx=frame_idx,
                bbox=BoundingBox(
                    x=int(x1), y=int(y1),
                    width=int(x2 - x1), height=int(y2 - y1),
                    confidence=score,
                ),
                landmarks=landmarks,
            ))
        return detections

    def detect_faces_in_video(
        self,
        video_path: Path,
        detection_interval: int = 10,
    ) -> Dict[int, List[FaceDetection]]:
        """Detect faces every Nth frame of a clip (sequential decode).

        Returns {frame_idx: detections} for frames where faces were found.
        """
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Scanning {total_frames} frames "
                    f"(detection every {detection_interval})")

        detections_by_frame: Dict[int, List[FaceDetection]] = {}
        frame_idx = 0
        try:
            while True:
                if frame_idx % detection_interval == 0:
                    # Decode + convert this frame and run detection on it
                    frame_read, frame = capture.read()
                    if not frame_read:
                        break
                    detections = self.detect_faces(frame, frame_idx)
                    if detections:
                        detections_by_frame[frame_idx] = detections
                else:
                    # Skipped frame: grab() advances the decoder without the
                    # (expensive) color conversion of retrieve()
                    if not capture.grab():
                        break
                frame_idx += 1
        finally:
            capture.release()

        logger.info(f"Detected faces in {len(detections_by_frame)} frames")
        return detections_by_frame
