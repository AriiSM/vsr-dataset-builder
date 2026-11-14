"""Unit tests for quality tier derivation (pure logic).

Run from the repo root:
    python backend/tests/test_quality_tiers.py
"""

import sys
import unittest
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_BACKEND_DIR), str(_BACKEND_DIR / "shared")]

from services.quality_indexer.quality_tiers import compute_quality_tier  # noqa: E402


TIER_CONFIG = {
    "preferred_duration": [2.0, 12.0],
    "tier_a": {"whisper_conf": 0.90, "whisper_conf_min": 0.70,
               "face_visibility": 0.95, "mouth_fail_rate": 0.05},
    "tier_b": {"whisper_conf": 0.70, "whisper_conf_min": 0.50,
               "face_visibility": 0.85, "mouth_fail_rate": 0.15},
}


def clean_segment(**overrides):
    metrics = {
        "whisper_conf": 0.95, "whisper_conf_min": 0.80,
        "face_visibility_ratio": 0.99, "mouth_landmark_fail_rate": 0.01,
        "asd_method": "talknet", "syncnet_method": "syncnet",
        "mouth_roi_method": "mediapipe",
        "boundary_start_type": "punctuation", "boundary_end_type": "silence",
        "duration": 5.0,
    }
    metrics.update(overrides)
    return metrics


class QualityTierTests(unittest.TestCase):
    def test_clean_segment_is_tier_a(self):
        self.assertEqual(compute_quality_tier(clean_segment(), TIER_CONFIG), "A")

    def test_weak_word_drops_to_b(self):
        metrics = clean_segment(whisper_conf_min=0.55)
        self.assertEqual(compute_quality_tier(metrics, TIER_CONFIG), "B")

    def test_terrible_confidence_is_c(self):
        metrics = clean_segment(whisper_conf=0.40, whisper_conf_min=0.20)
        self.assertEqual(compute_quality_tier(metrics, TIER_CONFIG), "C")

    def test_fallback_asd_caps_at_b(self):
        metrics = clean_segment(asd_method="fallback_motion")
        self.assertEqual(compute_quality_tier(metrics, TIER_CONFIG), "B")

    def test_mouth_fallback_caps_at_b(self):
        metrics = clean_segment(mouth_roi_method="retinaface_fallback")
        self.assertEqual(compute_quality_tier(metrics, TIER_CONFIG), "B")

    def test_out_of_range_duration_caps_at_b(self):
        metrics = clean_segment(duration=14.5)
        self.assertEqual(compute_quality_tier(metrics, TIER_CONFIG), "B")

    def test_nan_confidence_fails_closed_to_c(self):
        metrics = clean_segment(whisper_conf=float("nan"),
                                whisper_conf_min=float("nan"))
        self.assertEqual(compute_quality_tier(metrics, TIER_CONFIG), "C")

    def test_missing_metric_fails_closed(self):
        metrics = clean_segment()
        del metrics["face_visibility_ratio"]
        self.assertEqual(compute_quality_tier(metrics, TIER_CONFIG), "C")

    def test_disabled_syncnet_does_not_cap(self):
        metrics = clean_segment(syncnet_method="disabled")
        self.assertEqual(compute_quality_tier(metrics, TIER_CONFIG), "A")


if __name__ == "__main__":
    unittest.main(verbosity=2)
