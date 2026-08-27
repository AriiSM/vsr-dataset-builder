"""
Kalman smoothing for face bounding boxes.

Constant-velocity model over [center_x, center_y, width, height]:
PREDICT extrapolates with the current velocity (frames without a detection);
UPDATE corrects toward the RetinaFace measurement (frames with one).
"""

import cv2
import numpy as np

from services.face_tracker.geometry import BoundingBox


class KalmanBBoxFilter:
    """Constant-velocity Kalman filter for one face track.

    State (8-dim):       [cx, cy, w, h, vx, vy, vw, vh]
    Measurement (4-dim): [cx, cy, w, h]

    process_noise:     smaller = smoother (trust the motion model more)
    measurement_noise: smaller = snappier (trust RetinaFace more)
    """

    def __init__(
        self,
        process_noise: float,
        measurement_noise: float,
        initial_bbox: BoundingBox,
    ):
        self.kalman = cv2.KalmanFilter(dynamParams=8, measureParams=4, type=cv2.CV_32F)

        # Constant-velocity transition (dt = 1 frame)
        transition = np.eye(8, dtype=np.float32)
        transition[0, 4] = transition[1, 5] = transition[2, 6] = transition[3, 7] = 1.0
        self.kalman.transitionMatrix = transition

        # Measurement selects [cx, cy, w, h] from the state
        measurement_matrix = np.zeros((4, 8), dtype=np.float32)
        measurement_matrix[0, 0] = measurement_matrix[1, 1] = 1.0
        measurement_matrix[2, 2] = measurement_matrix[3, 3] = 1.0
        self.kalman.measurementMatrix = measurement_matrix

        self.kalman.processNoiseCov = (process_noise * np.eye(8)).astype(np.float32)
        self.kalman.measurementNoiseCov = (measurement_noise * np.eye(4)).astype(np.float32)
        self.kalman.errorCovPost = np.eye(8, dtype=np.float32)

        center_x = initial_bbox.x + initial_bbox.width / 2.0
        center_y = initial_bbox.y + initial_bbox.height / 2.0
        self.kalman.statePost = np.array(
            [center_x, center_y, initial_bbox.width, initial_bbox.height, 0, 0, 0, 0],
            dtype=np.float32,
        ).reshape(8, 1)

    def predict(self) -> BoundingBox:
        return self._state_to_bbox(self.kalman.predict())

    def update(self, measured_bbox: BoundingBox) -> BoundingBox:
        center_x = measured_bbox.x + measured_bbox.width / 2.0
        center_y = measured_bbox.y + measured_bbox.height / 2.0
        measurement = np.array(
            [center_x, center_y, measured_bbox.width, measured_bbox.height],
            dtype=np.float32,
        ).reshape(4, 1)
        return self._state_to_bbox(
            self.kalman.correct(measurement), confidence=measured_bbox.confidence
        )

    @staticmethod
    def _state_to_bbox(state: np.ndarray, confidence: float = 1.0) -> BoundingBox:
        center_x, center_y = float(state[0]), float(state[1])
        width, height = float(state[2]), float(state[3])
        return BoundingBox(
            x=int(round(center_x - width / 2.0)),
            y=int(round(center_y - height / 2.0)),
            width=int(round(max(1.0, width))),
            height=int(round(max(1.0, height))),
            confidence=confidence,
        )
