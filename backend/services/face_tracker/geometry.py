"""
Geometry types for face detection and tracking.

Pure data + math (IoU) — no ML imports, fully unit-testable.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BoundingBox:
    """Axis-aligned face bounding box in pixel coordinates."""
    x: int      # top-left x
    y: int      # top-left y
    width: int
    height: int
    confidence: float = 1.0

    @property
    def area(self) -> int:
        return self.width * self.height

    def iou(self, other: 'BoundingBox') -> float:
        """Intersection over Union with another box (0.0 when disjoint)."""
        left = max(self.x, other.x)
        top = max(self.y, other.y)
        right = min(self.x + self.width, other.x + other.width)
        bottom = min(self.y + self.height, other.y + other.height)

        if right <= left or bottom <= top:
            return 0.0

        intersection = (right - left) * (bottom - top)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0


@dataclass
class FaceDetection:
    """One face found by RetinaFace in one frame."""
    frame_idx: int
    bbox: BoundingBox
    # RetinaFace always returns exactly 5 landmarks:
    # [left_eye, right_eye, nose, left_mouth_corner, right_mouth_corner]
    landmarks: Optional[np.ndarray] = None
    # NOTE: the detection confidence lives on bbox.confidence — one source.
