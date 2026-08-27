"""
Track crops — ONE decode loop for every crop consumer in this service.

Reads a clip once and produces face crops for MULTIPLE tracks in the same
pass (ASD scores all candidates from one decode), parametrized by size,
color and margin so both consumers share the implementation:
    TalkNet:  112×112 grayscale, margin 0.2
    SyncNet:  240×240 BGR,       margin 0.0

Frames where a track has no usable bbox get a zeros placeholder — same
contract the models were fed before.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class CropRequest:
    """One track's crop window (frame range in clip-file coordinates)."""
    track: object            # FaceTrack
    start_frame: int
    end_frame: int           # exclusive


class TrackCropReader:
    """Sequential single-decode crop extraction for one or many tracks."""

    def read(
        self,
        video_path: Path,
        requests: List[CropRequest],
        target_size: Tuple[int, int],
        grayscale: bool,
        bbox_margin: float = 0.0,
    ) -> Dict[int, np.ndarray]:
        """Decode the clip ONCE and crop every requested track window.

        Returns {track_id: crops array} — (T, H, W) grayscale or (T, H, W, 3)
        BGR, T = the request's frame count, zeros where no bbox exists.
        """
        if not requests:
            return {}

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        crop_shape = (
            (*target_size[::-1],) if grayscale else (*target_size[::-1], 3)
        )
        crops_by_track: Dict[int, list] = {r.track.track_id: [] for r in requests}
        last_frame_needed = max(r.end_frame for r in requests)

        frame_idx = 0
        try:
            while frame_idx < last_frame_needed:
                frame_read, frame = capture.read()
                if not frame_read:
                    break
                for request in requests:
                    if request.start_frame <= frame_idx < request.end_frame:
                        crops_by_track[request.track.track_id].append(
                            self._crop_frame(frame, request.track, frame_idx,
                                             target_size, grayscale,
                                             bbox_margin, crop_shape)
                        )
                frame_idx += 1
        finally:
            capture.release()

        # Pad with zeros if the file ended before a request's window did
        result = {}
        for request in requests:
            crops = crops_by_track[request.track.track_id]
            expected = request.end_frame - request.start_frame
            while len(crops) < expected:
                crops.append(np.zeros(crop_shape, dtype=np.uint8))
            result[request.track.track_id] = np.array(crops)
        return result

    @staticmethod
    def _crop_frame(frame, track, frame_idx, target_size, grayscale,
                    bbox_margin, crop_shape) -> np.ndarray:
        """Crop one track's face out of one frame (zeros when unavailable)."""
        if not (track.start_frame <= frame_idx <= track.end_frame):
            return np.zeros(crop_shape, dtype=np.uint8)

        bbox = track.get_bbox_at_frame(frame_idx) or track.interpolate_bbox(frame_idx)
        if bbox is None:
            return np.zeros(crop_shape, dtype=np.uint8)

        frame_height, frame_width = frame.shape[:2]
        margin_x = int(bbox.width * bbox_margin)
        margin_y = int(bbox.height * bbox_margin)
        left = max(0, bbox.x - margin_x)
        top = max(0, bbox.y - margin_y)
        right = min(frame_width, bbox.x + bbox.width + margin_x)
        bottom = min(frame_height, bbox.y + bbox.height + margin_y)

        face = frame[top:bottom, left:right]
        if face.size == 0:
            return np.zeros(crop_shape, dtype=np.uint8)

        face = cv2.resize(face, target_size)
        if grayscale:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        return face
