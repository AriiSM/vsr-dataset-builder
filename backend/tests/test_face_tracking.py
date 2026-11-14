"""Unit tests for face_tracker's pure logic: IoU geometry, greedy
association, trajectory interpolation. No cv2/torch needed.

Run from the repo root:
    python backend/tests/test_face_tracking.py
"""

import sys
import types
import unittest
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_BACKEND_DIR), str(_BACKEND_DIR / "shared")]

if "loguru" not in sys.modules:
    _loguru = types.ModuleType("loguru")

    class _SilentLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    _loguru.logger = _SilentLogger()
    sys.modules["loguru"] = _loguru

from services.face_tracker.face_track import FaceTrack  # noqa: E402
from services.face_tracker.face_tracker import FaceTracker  # noqa: E402
from services.face_tracker.geometry import BoundingBox, FaceDetection  # noqa: E402


def box(x, y, size=100):
    return BoundingBox(x=x, y=y, width=size, height=size)


def detection(frame_idx, x, y, size=100):
    return FaceDetection(frame_idx=frame_idx, bbox=box(x, y, size))


class IoUTests(unittest.TestCase):
    def test_identical_boxes(self):
        self.assertAlmostEqual(box(0, 0).iou(box(0, 0)), 1.0)

    def test_disjoint_boxes(self):
        self.assertEqual(box(0, 0).iou(box(500, 500)), 0.0)

    def test_touching_edges_is_zero(self):
        self.assertEqual(box(0, 0, 100).iou(box(100, 0, 100)), 0.0)

    def test_half_overlap(self):
        # 100×100 boxes shifted by 50px horizontally → I=5000, U=15000
        self.assertAlmostEqual(box(0, 0).iou(box(50, 0)), 1 / 3)

    def test_contained_box(self):
        outer, inner = box(0, 0, 100), box(25, 25, 50)
        self.assertAlmostEqual(outer.iou(inner), 2500 / 10000)


class AssociationTests(unittest.TestCase):
    def _tracker(self, **overrides):
        params = dict(iou_threshold=0.3, max_age=30, min_hits=2,
                      kalman_process_noise=1e-3, kalman_measurement_noise=1.0)
        params.update(overrides)
        return FaceTracker(**params)

    def test_two_people_stay_two_tracks(self):
        # Person A on the left, person B on the right, both drifting slightly
        detections = {
            0:  [detection(0, 100, 100),  detection(0, 800, 100)],
            10: [detection(10, 110, 100), detection(10, 790, 100)],
            20: [detection(20, 120, 105), detection(20, 780, 95)],
        }
        tracks = self._tracker().track(detections, build_trajectories=False)
        self.assertEqual(len(tracks), 2)
        self.assertTrue(all(len(t.detections) == 3 for t in tracks))

    def test_min_hits_filters_flickers(self):
        # One stable face + a single-frame false positive
        detections = {
            0:  [detection(0, 100, 100), detection(0, 900, 500)],
            10: [detection(10, 105, 100)],
            20: [detection(20, 110, 100)],
        }
        tracks = self._tracker(min_hits=2).track(detections, build_trajectories=False)
        self.assertEqual(len(tracks), 1)

    def test_track_survives_short_disappearance(self):
        # Face missing at frame 10 (blink of detection), back at 20 — same track
        detections = {
            0:  [detection(0, 100, 100)],
            20: [detection(20, 108, 102)],
        }
        tracks = self._tracker(max_age=30).track(detections, build_trajectories=False)
        self.assertEqual(len(tracks), 1)

    def test_long_absence_creates_new_track(self):
        detections = {
            0:   [detection(0, 100, 100)],
            10:  [detection(10, 100, 100)],
            100: [detection(100, 100, 100)],
            110: [detection(110, 100, 100)],
        }
        tracks = self._tracker(max_age=30).track(detections, build_trajectories=False)
        self.assertEqual(len(tracks), 2)

    def test_greedy_prefers_highest_iou(self):
        # New detection overlaps both tracks; must go to the closer one
        tracker = self._tracker(min_hits=1)
        detections = {
            0:  [detection(0, 100, 100), detection(0, 160, 100)],
            10: [detection(10, 102, 100)],   # clearly the left face
        }
        tracks = tracker.track(detections, build_trajectories=False)
        left = next(t for t in tracks if t.detections[0].bbox.x == 100)
        self.assertEqual(len(left.detections), 2)


class TrajectoryTests(unittest.TestCase):
    def test_linear_interpolation_between_detections(self):
        track = FaceTrack(track_id=0, detections=[
            detection(0, 0, 0), detection(10, 100, 0),
        ])
        midpoint = track.interpolate_bbox(5)
        self.assertEqual(midpoint.x, 50)

    def test_outside_range_returns_none(self):
        track = FaceTrack(track_id=0, detections=[detection(5, 0, 0)])
        self.assertIsNone(track.interpolate_bbox(4))
        self.assertIsNone(track.interpolate_bbox(6))

    def test_exact_frame_returns_raw_detection(self):
        track = FaceTrack(track_id=0, detections=[detection(3, 42, 7)])
        self.assertEqual(track.get_bbox_at_frame(3).x, 42)
        self.assertIsNone(track.get_bbox_at_frame(4))


if __name__ == "__main__":
    unittest.main(verbosity=2)
