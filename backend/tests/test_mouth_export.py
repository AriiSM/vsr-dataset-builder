"""Unit tests for mouth_exporter's pure logic: audio slicing (stdlib wave),
annotation write/parse roundtrip, crop trajectory math. No GPU/cv2 needed.

Run from the repo root:
    python backend/tests/test_mouth_export.py
"""

import struct
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_BACKEND_DIR), str(_BACKEND_DIR / "shared")]

for _missing in ("torch", "torchaudio", "cv2", "loguru"):
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

from services.mouth_exporter.audio_slicer import write_segment_audio  # noqa: E402
from services.mouth_exporter.annotation_io import (  # noqa: E402
    parse_lrs2_annotation,
    write_annotation,
)
from services.mouth_exporter.crop_trajectories import (  # noqa: E402
    build_mouth_trajectory_v2,
    fill_nan_gaps,
)
from services.mouth_exporter.mouth_landmarks import LipLandmarks  # noqa: E402


def make_wav(path: Path, seconds: float, sample_rate: int = 16000):
    """PCM16 mono wav whose sample VALUES equal their frame index (mod 2^15)
    — so slices are verifiable by content, not just length."""
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = b"".join(
            struct.pack("<h", i % 32768)
            for i in range(int(seconds * sample_rate))
        )
        handle.writeframes(frames)


class AudioSlicerTests(unittest.TestCase):
    def test_slice_has_exact_length_and_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip_wav = Path(tmp) / "clip.wav"
            out_wav = Path(tmp) / "segment.wav"
            make_wav(clip_wav, seconds=3.0)

            self.assertTrue(write_segment_audio(clip_wav, out_wav, 1.0, 2.0))

            with wave.open(str(out_wav), "rb") as result:
                self.assertEqual(result.getnframes(), 16000)      # exactly 1 s
                self.assertEqual(result.getframerate(), 16000)
                first = struct.unpack("<h", result.readframes(1))[0]
                self.assertEqual(first, 16000)                    # starts at 1.0 s

    def test_overlong_window_clamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip_wav = Path(tmp) / "clip.wav"
            out_wav = Path(tmp) / "segment.wav"
            make_wav(clip_wav, seconds=2.0)
            self.assertTrue(write_segment_audio(clip_wav, out_wav, 1.5, 60.0))
            with wave.open(str(out_wav), "rb") as result:
                self.assertEqual(result.getnframes(), 8000)       # only 0.5 s left

    def test_empty_window_returns_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip_wav = Path(tmp) / "clip.wav"
            make_wav(clip_wav, seconds=1.0)
            self.assertFalse(
                write_segment_audio(clip_wav, Path(tmp) / "x.wav", 5.0, 6.0))

    def test_missing_source_returns_false_not_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(write_segment_audio(
                Path(tmp) / "nope.wav", Path(tmp) / "x.wav", 0.0, 1.0))


class _FakeWord:
    def __init__(self, text, start, end):
        self.text, self.start, self.end = text, start, end


class _FakeTranscription:
    def __init__(self):
        self.text = "amu îi vremea bună"
        self.start = 10.0
        self.words = [
            _FakeWord("amu", 10.0, 10.4), _FakeWord("îi", 10.5, 10.7),
            _FakeWord("vremea", 10.8, 11.3), _FakeWord("bună", 11.4, 11.9),
        ]
        self.confidence = 0.92
        self.confidence_min = 0.74


class AnnotationRoundtripTests(unittest.TestCase):
    def test_write_then_parse(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seg.txt"
            write_annotation(path, _FakeTranscription(), [9.1, 8.7, 9.5, 9.9])
            parsed = parse_lrs2_annotation(path)

            self.assertEqual(parsed["text"], "AMU ÎI VREMEA BUNĂ")
            self.assertEqual(parsed["original"], "AMU ÎI VREMEA BUNĂ")
            self.assertEqual(parsed["confidence"], 3)   # min .74 / mean .92 → high
            self.assertEqual(len(parsed["words"]), 4)
            # word times are RELATIVE to the segment start
            self.assertAlmostEqual(parsed["words"][0]["start"], 0.0)
            self.assertAlmostEqual(parsed["words"][3]["end"], 1.9, places=2)
            self.assertAlmostEqual(parsed["words"][1]["asd_score"], 8.7)


class TrajectoryTests(unittest.TestCase):
    def test_fill_nan(self):
        arr = np.array([1.0, np.nan, np.nan, 4.0])
        self.assertEqual(list(fill_nan_gaps(arr)), [1.0, 2.0, 3.0, 4.0])

    def test_v2_median_sizing_resists_open_mouth_outlier(self):
        # 49 frames with width 40, ONE with width 120 (yawn) —
        # median sizing must ignore the outlier (max would double the crop)
        landmarks = {
            f: LipLandmarks((100.0, 200.0), 120.0 if f == 25 else 40.0,
                            0.0, 0.0, 0.0)
            for f in range(50)
        }
        traj, fail_rate, _ = build_mouth_trajectory_v2(
            landmarks, 0, 50, fps=25.0, smoothing_type="one_euro",
            one_euro_min_cutoff=1.0, one_euro_beta=0.3, gaussian_sigma=3.0,
            width_multiplier=1.8, minimum_half_size_pixels=24)
        self.assertEqual(fail_rate, 0.0)
        _, _, half, _ = traj[0]
        self.assertEqual(half, int(40.0 * 1.8 / 2))   # median, not max

    def test_v2_gaps_carry_forward_and_count_in_fail_rate(self):
        landmarks = {
            f: LipLandmarks((100.0, 200.0), 40.0, 0.0, 0.0, 0.0)
            for f in range(0, 50) if f % 5 != 0
        }
        traj, fail_rate, _ = build_mouth_trajectory_v2(
            landmarks, 0, 50, fps=25.0, smoothing_type="one_euro",
            one_euro_min_cutoff=1.0, one_euro_beta=0.3, gaussian_sigma=3.0,
            width_multiplier=1.8, minimum_half_size_pixels=24)
        self.assertEqual(len(traj), 50)               # no holes in the output
        self.assertAlmostEqual(fail_rate, 0.2, places=2)

    def test_v2_empty_landmarks_signal_total_failure(self):
        traj, fail_rate, head_pose = build_mouth_trajectory_v2(
            {}, 0, 50, fps=25.0, smoothing_type="one_euro",
            one_euro_min_cutoff=1.0, one_euro_beta=0.3, gaussian_sigma=3.0,
            width_multiplier=1.8, minimum_half_size_pixels=24)
        self.assertEqual(traj, {})
        self.assertEqual(fail_rate, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
