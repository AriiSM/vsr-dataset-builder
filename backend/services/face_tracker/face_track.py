"""
FaceTrack — the history of ONE person across a clip.

Holds the raw RetinaFace detections (every Nth frame) and, once tracking is
done, a smoothed PER-FRAME trajectory (Kalman) that downstream consumers
(ASD crops, SyncNet crops, the final export) read via interpolate_bbox().
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from services.face_tracker.geometry import BoundingBox, FaceDetection


@dataclass
class FaceTrack:
    """A tracked face across multiple frames.

    After tracking, FaceTracker calls build_smoothed_trajectory() to populate
    the per-frame bbox dict that interpolate_bbox() reads. Without it (unit
    tests, legacy paths), interpolate_bbox falls back to linear interpolation
    between the surrounding raw detections.
    """
    track_id: int
    detections: List[FaceDetection] = field(default_factory=list)
    _smoothed_trajectory: Optional[Dict[int, BoundingBox]] = field(default=None, repr=False)

    @property
    def start_frame(self) -> int:
        return min(d.frame_idx for d in self.detections)

    @property
    def end_frame(self) -> int:
        return max(d.frame_idx for d in self.detections)

    def get_bbox_at_frame(self, frame_idx: int) -> Optional[BoundingBox]:
        """RAW (un-smoothed) bbox at this exact frame, if RetinaFace saw one."""
        for detection in self.detections:
            if detection.frame_idx == frame_idx:
                return detection.bbox
        return None

    def build_smoothed_trajectory(
        self,
        process_noise: float,
        measurement_noise: float,
    ) -> None:
        """Forward Kalman pass over [start_frame ... end_frame].

        At each frame: PREDICT (extrapolate with inertia); on frames with a
        RetinaFace detection: UPDATE (correct using the measurement).
        """
        # Imported here so the pure logic in this module (and its tests)
        # never requires cv2.
        from services.face_tracker.kalman_filter import KalmanBBoxFilter

        if not self.detections:
            self._smoothed_trajectory = {}
            return

        ordered = sorted(self.detections, key=lambda d: d.frame_idx)
        detection_at_frame = {d.frame_idx: d for d in ordered}
        first_frame = ordered[0].frame_idx
        last_frame = ordered[-1].frame_idx

        kalman = KalmanBBoxFilter(process_noise, measurement_noise, ordered[0].bbox)
        trajectory: Dict[int, BoundingBox] = {first_frame: ordered[0].bbox}

        for frame_idx in range(first_frame + 1, last_frame + 1):
            predicted = kalman.predict()
            if frame_idx in detection_at_frame:
                trajectory[frame_idx] = kalman.update(detection_at_frame[frame_idx].bbox)
            else:
                trajectory[frame_idx] = predicted

        self._smoothed_trajectory = trajectory

    def interpolate_bbox(self, frame_idx: int) -> Optional[BoundingBox]:
        """Bounding box at ANY frame inside [start_frame, end_frame].

        Smoothed trajectory when built; linear interpolation otherwise.
        """
        if frame_idx < self.start_frame or frame_idx > self.end_frame:
            return None
        if self._smoothed_trajectory is not None:
            return self._smoothed_trajectory.get(frame_idx)
        return self._linear_interpolate_bbox(frame_idx)

    def _linear_interpolate_bbox(self, frame_idx: int) -> Optional[BoundingBox]:
        """Fallback: linear interpolation between surrounding detections."""
        before = None
        after = None
        for detection in sorted(self.detections, key=lambda d: d.frame_idx):
            if detection.frame_idx <= frame_idx:
                before = detection
            if detection.frame_idx >= frame_idx and after is None:
                after = detection

        if before is None or after is None:
            return before.bbox if before else (after.bbox if after else None)
        if before.frame_idx == after.frame_idx:
            return before.bbox

        fraction = (frame_idx - before.frame_idx) / (after.frame_idx - before.frame_idx)
        return BoundingBox(
            x=int(before.bbox.x + fraction * (after.bbox.x - before.bbox.x)),
            y=int(before.bbox.y + fraction * (after.bbox.y - before.bbox.y)),
            width=int(before.bbox.width + fraction * (after.bbox.width - before.bbox.width)),
            height=int(before.bbox.height + fraction * (after.bbox.height - before.bbox.height)),
            confidence=(before.bbox.confidence + after.bbox.confidence) / 2,
        )
