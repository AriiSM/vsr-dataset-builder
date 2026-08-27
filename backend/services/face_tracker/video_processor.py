"""
Video Processor — the face_tracker service's orchestrator.

Composes:
    RetinaFaceDetector (face_detector.py) — WHO/WHERE per sampled frame
    FaceTracker        (face_tracker.py)  — identity tracks across frames

Contract (IN → OUT):
    IN  : clip mp4 (video-only, CFR 25 fps)
    OUT : List[FaceTrack] — one per person, with a smoothed per-frame
          bbox trajectory ready for ASD / SyncNet / export.

Memory note: frames stream one at a time (sequential decode); only the
tiny detections + trajectories (hundreds of KB per clip) stay in RAM.
"""

from pathlib import Path
from typing import List

from services.face_tracker.face_detector import RetinaFaceDetector
from services.face_tracker.face_track import FaceTrack
from services.face_tracker.face_tracker import FaceTracker

__all__ = ["VideoProcessor", "FaceTrack"]


class VideoProcessor:
    """Detection → tracking for one clip at a time."""

    def __init__(
        self,
        detection_confidence: float,
        detection_nms_threshold: float,
        tracking_iou: float,
        max_track_age: int,
        min_track_hits: int,
        kalman_process_noise: float,
        kalman_measurement_noise: float,
    ):
        self.detector = RetinaFaceDetector(
            confidence_threshold=detection_confidence,
            nms_threshold=detection_nms_threshold,
        )
        self.tracker = FaceTracker(
            iou_threshold=tracking_iou,
            max_age=max_track_age,
            min_hits=min_track_hits,
            kalman_process_noise=kalman_process_noise,
            kalman_measurement_noise=kalman_measurement_noise,
        )

    def process_video(
        self,
        video_path: Path,
        detection_interval: int = 10,
    ) -> List[FaceTrack]:
        """Detect (every Nth frame) and track all faces in a clip."""
        detections_by_frame = self.detector.detect_faces_in_video(
            video_path,
            detection_interval=detection_interval,
        )
        return self.tracker.track(detections_by_frame)
