"""Live test for ClipCutter: one ffmpeg invocation, one modality per file.

Requires ffmpeg + ffprobe on PATH. Run from the repo root:
    python backend/tests/test_clip_cutter.py
"""

import json
import shutil
import subprocess
import sys
import tempfile
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

from services.segmenter.clip_cutter import ClipCutter  # noqa: E402


def _streams(path: Path) -> list:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", str(path)],
        capture_output=True, check=True,
    )
    return json.loads(result.stdout)["streams"]


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                     "ffmpeg/ffprobe not on PATH")
class ClipCutterTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.source = self.tmp / "source.mp4"
        subprocess.run(
            ["ffmpeg", "-y",
             "-f", "lavfi", "-i", "testsrc=duration=10:size=320x240:rate=30",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
             "-c:v", "libx264", "-c:a", "aac", "-shortest", str(self.source)],
            capture_output=True, check=True,
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_one_modality_per_file(self):
        cutter = ClipCutter(clip_crf=16, clip_preset="veryfast",
                            seek_start_padding=0.05)
        out_video = self.tmp / "clip.mp4"
        out_audio = self.tmp / "clip.wav"

        content_start = cutter.cut(self.source, 2.0, 5.0, out_video, out_audio)
        self.assertIsNotNone(content_start)
        self.assertAlmostEqual(content_start, 1.95, places=6)

        video_streams = _streams(out_video)
        self.assertEqual(len(video_streams), 1)
        self.assertEqual(video_streams[0]["codec_type"], "video")
        self.assertEqual(video_streams[0]["r_frame_rate"], "25/1")

        audio_streams = _streams(out_audio)
        self.assertEqual(len(audio_streams), 1)
        self.assertEqual(audio_streams[0]["codec_type"], "audio")
        self.assertEqual(int(audio_streams[0]["sample_rate"]), 16000)

    def test_failed_cut_returns_none(self):
        cutter = ClipCutter(clip_crf=16, clip_preset="veryfast",
                            seek_start_padding=0.05)
        missing_source = self.tmp / "no_such_file.mp4"
        result = cutter.cut(missing_source, 0.0, 2.0,
                            self.tmp / "v.mp4", self.tmp / "a.wav")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
