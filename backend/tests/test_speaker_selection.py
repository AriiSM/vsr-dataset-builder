"""Unit tests for speaker_detector's pure logic: MFCC window slicing,
candidate gating and the winner-selection formula. No GPU/cv2/scipy needed
(the heavy paths are stubbed).

Run from the repo root:
    python backend/tests/test_speaker_selection.py
"""

import sys
import types
import unittest
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_BACKEND_DIR), str(_BACKEND_DIR / "shared")]

for _missing in ("torch", "cv2", "loguru"):
    try:
        __import__(_missing)
    except ImportError:
        _mod = types.ModuleType(_missing)
        if _missing == "loguru":
            class _SilentLogger:
                def __getattr__(self, _name):
                    return lambda *args, **kwargs: None
            _mod.logger = _SilentLogger()
        sys.modules[_missing] = _mod

import numpy as np  # noqa: E402

from services.face_tracker.face_track import FaceTrack  # noqa: E402
from services.face_tracker.geometry import BoundingBox, FaceDetection  # noqa: E402
from services.speaker_detector.audio_features import MfccExtractor  # noqa: E402
from services.speaker_detector.speaker_selector import SpeakerSelector  # noqa: E402


class StubMfcc(MfccExtractor):
    """Replaces the real wav read with a deterministic matrix."""

    def __init__(self, total_seconds=10.0):
        super().__init__()
        self._matrix = np.arange(
            int(total_seconds * self.ROWS_PER_SECOND) * 13, dtype=float
        ).reshape(-1, 13)
        self.compute_calls = 0

    def _compute_mfcc(self, wav_path):
        self.compute_calls += 1
        return self._matrix


class MfccWindowTests(unittest.TestCase):
    def test_window_rows_match_seconds(self):
        extractor = StubMfcc()
        window = extractor.features_for_window(Path("clip.wav"), 1.0, 3.5)
        self.assertEqual(len(window), 250)          # 2.5 s × 100 rows/s
        self.assertEqual(window[0, 0], 100 * 13)    # starts at row 100

    def test_overlong_window_is_clamped_not_raised(self):
        extractor = StubMfcc(total_seconds=2.0)
        window = extractor.features_for_window(Path("clip.wav"), 1.0, 60.0)
        self.assertEqual(len(window), 100)          # only 1 s remained

    def test_cache_computes_once_per_wav(self):
        extractor = StubMfcc()
        extractor.features_for_window(Path("clip.wav"), 0.0, 1.0)   # TalkNet
        extractor.features_for_window(Path("clip.wav"), 2.0, 4.0)   # SyncNet
        self.assertEqual(extractor.compute_calls, 1)                # shared!
        extractor.features_for_window(Path("next_clip.wav"), 0.0, 1.0)
        self.assertEqual(extractor.compute_calls, 2)                # new clip


def make_track(track_id, start_frame, end_frame, face_size, fps=25):
    """A track visible in [start_frame, end_frame] with constant face size."""
    detections = [
        FaceDetection(frame_idx=f, bbox=BoundingBox(0, 0, face_size, face_size))
        for f in range(start_frame, end_frame + 1, 10)
    ]
    return FaceTrack(track_id=track_id, detections=detections)


class StubTalkNet:
    """Injects fixed mean scores per track_id (0-20 scale)."""

    def __init__(self, score_by_track):
        self.score_by_track = score_by_track
        self.method = "talknet"
        self.scored_track_ids = []

    def score_crops(self, crops, audio):
        # crops carry no track info — the selector calls per request in order,
        # so the reader stub records the mapping instead (see StubReader).
        raise NotImplementedError


class SelectorHarness(SpeakerSelector):
    """Overrides the heavy steps (decode + TalkNet) with injected scores."""

    def __init__(self, score_by_track, **kwargs):
        super().__init__(talknet=None, crop_reader=None,
                         mfcc_extractor=None, **kwargs)
        self._score_by_track = score_by_track
        self.gated_track_ids = None

    def select_active_speaker(self, video_path, audio_path, face_tracks,
                              speech_start, speech_end, fps):
        candidates = self._gate_candidates(face_tracks, speech_start, speech_end, fps)
        self.gated_track_ids = [t.track_id for t in candidates]
        if not candidates:
            return None
        from services.speaker_detector.active_speaker import ASDResult
        results = [
            (track, ASDResult(
                track_id=track.track_id, start_frame=0, end_frame=0,
                scores=[self._score_by_track.get(track.track_id, 0.0)],
                method="talknet",
            ))
            for track in candidates
        ]
        return self._pick_winner(results, speech_start, speech_end, fps)


class SelectionTests(unittest.TestCase):
    def test_all_overlapping_tracks_compete_without_cap(self):
        # FIVE speakers visible (panel wide shot) — nobody may be pre-eliminated
        tracks = [make_track(i, 0, 250, face_size=200 - i * 30) for i in range(5)]
        selector = SelectorHarness({}, min_track_speech_overlap=0.3,
                                   max_candidate_tracks=None)
        selector.select_active_speaker("v", "a", tracks, 0.0, 10.0, 25.0)
        self.assertEqual(len(selector.gated_track_ids), 5)

    def test_small_face_with_best_asd_wins(self):
        # The REAL speaker has the smallest face but a dominant ASD score —
        # exactly the 5-speaker wide-shot case the cap removal protects.
        tracks = [
            make_track(0, 0, 250, face_size=300),   # big listener
            make_track(1, 0, 250, face_size=280),   # big listener
            make_track(2, 0, 250, face_size=80),    # small REAL speaker
        ]
        selector = SelectorHarness({0: 1.0, 1: 1.5, 2: 15.0},
                                   min_track_speech_overlap=0.3,
                                   max_candidate_tracks=None)
        selection = selector.select_active_speaker("v", "a", tracks, 0.0, 10.0, 25.0)
        self.assertEqual(selection.track.track_id, 2)

    def test_numeric_cap_still_available_as_speed_knob(self):
        tracks = [make_track(i, 0, 250, face_size=100 + i) for i in range(5)]
        selector = SelectorHarness({}, min_track_speech_overlap=0.3,
                                   max_candidate_tracks=3)
        selector.select_active_speaker("v", "a", tracks, 0.0, 10.0, 25.0)
        self.assertEqual(len(selector.gated_track_ids), 3)

    def test_track_without_speech_overlap_is_gated_out(self):
        tracks = [
            make_track(0, 0, 250, face_size=100),     # covers the speech
            make_track(1, 500, 700, face_size=100),   # appears after speech ends
        ]
        selector = SelectorHarness({0: 10.0}, min_track_speech_overlap=0.3,
                                   max_candidate_tracks=None)
        selection = selector.select_active_speaker("v", "a", tracks, 0.0, 10.0, 25.0)
        self.assertEqual(selector.gated_track_ids, [0])
        self.assertEqual(selection.track.track_id, 0)

    def test_no_overlapping_tracks_returns_none(self):
        tracks = [make_track(0, 500, 700, face_size=100)]
        selector = SelectorHarness({}, min_track_speech_overlap=0.3,
                                   max_candidate_tracks=None)
        self.assertIsNone(
            selector.select_active_speaker("v", "a", tracks, 0.0, 10.0, 25.0)
        )

    def test_zero_asd_scores_yield_no_winner(self):
        tracks = [make_track(0, 0, 250, face_size=100)]
        selector = SelectorHarness({0: 0.0}, min_track_speech_overlap=0.3,
                                   max_candidate_tracks=None)
        self.assertIsNone(
            selector.select_active_speaker("v", "a", tracks, 0.0, 10.0, 25.0)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
