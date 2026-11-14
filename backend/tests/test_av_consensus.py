"""Unit tests for the voice↔face consensus (pure logic, stdlib only).

Run from the repo root:
    python backend/tests/test_av_consensus.py
"""

import sys
import unittest
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_BACKEND_DIR), str(_BACKEND_DIR / "shared")]

from services.quality_indexer.av_consensus import compute_av_consensus  # noqa: E402


class AvConsensusTests(unittest.TestCase):
    def test_consistent_video_has_no_mismatches(self):
        result = compute_av_consensus([
            ("seg_0", "SPEAKER_00", "md_001_spk0"),
            ("seg_1", "SPEAKER_01", "md_001_spk1"),
            ("seg_2", "SPEAKER_00", "md_001_spk0"),
        ])
        self.assertEqual(result.num_judged, 3)
        self.assertEqual(result.num_mismatched, 0)
        self.assertEqual(result.voice_to_face["SPEAKER_00"], "md_001_spk0")
        self.assertEqual(result.voice_to_face["SPEAKER_01"], "md_001_spk1")

    def test_minority_segment_is_flagged(self):
        # Voice SPEAKER_00 belongs to spk0 in 3 of 4 segments; the fourth
        # (a B-roll shot of another face while the anchor keeps talking)
        # contradicts the majority and must be flagged.
        result = compute_av_consensus([
            ("seg_0", "SPEAKER_00", "md_001_spk0"),
            ("seg_1", "SPEAKER_00", "md_001_spk0"),
            ("seg_2", "SPEAKER_00", "md_001_spk1"),
            ("seg_3", "SPEAKER_00", "md_001_spk0"),
        ])
        self.assertEqual(result.num_mismatched, 1)
        self.assertTrue(result.mismatch_by_segment["seg_2"])
        self.assertFalse(result.mismatch_by_segment["seg_0"])

    def test_unlabeled_segments_are_not_judged(self):
        # No diarization label / no identity evidence → skipped, never flagged.
        result = compute_av_consensus([
            ("seg_0", "", "md_001_spk0"),
            ("seg_1", "SPEAKER_00", ""),
            ("seg_2", "SPEAKER_00", "md_001_spk0"),
        ])
        self.assertEqual(result.num_judged, 1)
        self.assertNotIn("seg_0", result.mismatch_by_segment)
        self.assertNotIn("seg_1", result.mismatch_by_segment)

    def test_single_segment_voice_trivially_matches(self):
        result = compute_av_consensus([
            ("seg_0", "SPEAKER_00", "md_001_spk0"),
        ])
        self.assertEqual(result.num_judged, 1)
        self.assertEqual(result.num_mismatched, 0)

    def test_two_voices_one_face_both_consistent(self):
        # Dubbing / same face for two voices: each voice maps to that face,
        # nothing is contradictory per-voice.
        result = compute_av_consensus([
            ("seg_0", "SPEAKER_00", "md_001_spk0"),
            ("seg_1", "SPEAKER_01", "md_001_spk0"),
        ])
        self.assertEqual(result.num_mismatched, 0)
        self.assertEqual(result.voice_to_face["SPEAKER_01"], "md_001_spk0")

    def test_tie_break_is_deterministic(self):
        # 1-1 vote: winner is the lexicographically first face, every run.
        result_a = compute_av_consensus([
            ("seg_0", "SPEAKER_00", "md_001_spk1"),
            ("seg_1", "SPEAKER_00", "md_001_spk0"),
        ])
        result_b = compute_av_consensus([
            ("seg_1", "SPEAKER_00", "md_001_spk0"),
            ("seg_0", "SPEAKER_00", "md_001_spk1"),
        ])
        self.assertEqual(result_a.voice_to_face, result_b.voice_to_face)
        self.assertEqual(result_a.voice_to_face["SPEAKER_00"], "md_001_spk0")

    def test_empty_input(self):
        result = compute_av_consensus([])
        self.assertEqual(result.num_judged, 0)
        self.assertEqual(result.voice_to_face, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
