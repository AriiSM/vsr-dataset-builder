"""
Face tracking — greedy IoU association of detections into identity tracks.

Turns per-frame detections into FaceTracks (one per PERSON): a detection is
matched to the track whose latest box overlaps it the most; unmatched
detections start new tracks; tracks unseen for max_age frames retire.

Pure logic over geometry types — unit-testable without any ML dependency.
"""

from typing import Dict, List, Set, Tuple

from loguru import logger

from services.face_tracker.face_track import FaceTrack
from services.face_tracker.geometry import FaceDetection


class FaceTracker:
    """Associates detections across frames into per-person tracks.

    Greedy IoU matching — equivalent to Hungarian assignment for
    single/two-speaker footage (the overwhelming majority of this corpus),
    far simpler, and one fewer dependency.
    """

    def __init__(
        self,
        iou_threshold: float,
        max_age: int,
        min_hits: int,
        kalman_process_noise: float,
        kalman_measurement_noise: float,
    ):
        self.iou_threshold = iou_threshold
        self.max_age = max_age        # frames a track survives without a detection
        self.min_hits = min_hits      # detections needed to confirm a track
        self.kalman_process_noise = kalman_process_noise
        self.kalman_measurement_noise = kalman_measurement_noise

    def track(
        self,
        detections_by_frame: Dict[int, List[FaceDetection]],
        build_trajectories: bool = True,
    ) -> List[FaceTrack]:
        """Build confirmed FaceTracks from per-frame detections.

        build_trajectories=False skips the Kalman pass (unit tests without
        cv2; production always builds them).
        """
        active: List[Tuple[FaceTrack, int]] = []   # (track, last_seen_frame)
        retired: List[FaceTrack] = []
        next_track_id = 0

        for frame_idx in sorted(detections_by_frame.keys()):
            detections = detections_by_frame[frame_idx]

            # Retire tracks unseen for longer than max_age
            still_active = []
            for track, last_seen in active:
                if frame_idx - last_seen <= self.max_age:
                    still_active.append((track, last_seen))
                else:
                    retired.append(track)
            active = still_active

            matches, unmatched_detections, unmatched_tracks = \
                self._associate_detections(detections, [t for t, _ in active])

            refreshed: List[Tuple[FaceTrack, int]] = []
            for detection_idx, track_idx in matches:
                track, _ = active[track_idx]
                track.detections.append(detections[detection_idx])
                refreshed.append((track, frame_idx))

            for track_idx in unmatched_tracks:
                refreshed.append(active[track_idx])   # may still match later

            for detection_idx in unmatched_detections:
                new_track = FaceTrack(
                    track_id=next_track_id,
                    detections=[detections[detection_idx]],
                )
                next_track_id += 1
                refreshed.append((new_track, frame_idx))

            active = refreshed

        retired.extend(track for track, _ in active)

        confirmed = [t for t in retired if len(t.detections) >= self.min_hits]

        if build_trajectories:
            for track in confirmed:
                track.build_smoothed_trajectory(
                    self.kalman_process_noise,
                    self.kalman_measurement_noise,
                )

        logger.info(
            f"Created {len(confirmed)} confirmed tracks "
            f"(from {len(retired)} total)"
        )
        return confirmed

    def _associate_detections(
        self,
        detections: List[FaceDetection],
        tracks: List[FaceTrack],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Greedy IoU matching between new detections and existing tracks.

        1. Compute IoU between every detection and every track's LATEST box.
        2. Keep pairs above iou_threshold, best first.
        3. Greedily take pairs whose detection AND track are both unclaimed.

        Returns (matches, unmatched_detection_indices, unmatched_track_indices).
        """
        if not tracks:
            return [], list(range(len(detections))), []
        if not detections:
            return [], [], list(range(len(tracks)))

        candidates: List[Tuple[float, int, int]] = []
        for detection_idx, detection in enumerate(detections):
            for track_idx, track in enumerate(tracks):
                if not track.detections:
                    continue
                iou = detection.bbox.iou(track.detections[-1].bbox)
                if iou >= self.iou_threshold:
                    candidates.append((iou, detection_idx, track_idx))
        candidates.sort(reverse=True)

        claimed_detections: Set[int] = set()
        claimed_tracks: Set[int] = set()
        matches: List[Tuple[int, int]] = []
        for _iou, detection_idx, track_idx in candidates:
            if detection_idx in claimed_detections or track_idx in claimed_tracks:
                continue
            matches.append((detection_idx, track_idx))
            claimed_detections.add(detection_idx)
            claimed_tracks.add(track_idx)

        unmatched_detections = [
            i for i in range(len(detections)) if i not in claimed_detections
        ]
        unmatched_tracks = [
            i for i in range(len(tracks)) if i not in claimed_tracks
        ]
        return matches, unmatched_detections, unmatched_tracks
