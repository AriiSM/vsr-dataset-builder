"""Live test for the single-encode ffmpeg pipe writer.

Requires an ffmpeg binary on PATH (or imageio-ffmpeg installed) and numpy.
Heavy pipeline deps (cv2, loguru) are stubbed — the writer doesn't use them.

Run from the repo root:
    python backend/tests/test_ffmpeg_pipe_writer.py
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

# Stub modules the formatter imports at module level but the writer never uses.
if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.ModuleType("cv2")
if "loguru" not in sys.modules:
    _loguru = types.ModuleType("loguru")

    class _SilentLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    _loguru.logger = _SilentLogger()
    sys.modules["loguru"] = _loguru

import numpy as np  # noqa: E402

from services.mouth_exporter.video_encoder import FfmpegPipeWriter  # noqa: E402


def _ffprobe(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-count_frames", str(path)],
        capture_output=True, check=True,
    )
    return json.loads(result.stdout)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                     "ffmpeg/ffprobe not on PATH")
class FfmpegPipeWriterTests(unittest.TestCase):
    def test_writes_valid_cfr_video(self):
        n_frames, size, fps = 50, (96, 96), 25.0
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "mouth.mp4"
            writer = FfmpegPipeWriter(
                out, size, fps, codec="libx264", crf=16, preset="veryfast",
            )
            rng = np.random.default_rng(0)
            for _ in range(n_frames):
                frame = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
                writer.write(frame)
            self.assertTrue(writer.release())

            info = _ffprobe(out)
            video = next(s for s in info["streams"] if s["codec_type"] == "video")
            self.assertEqual(video["codec_name"], "h264")
            self.assertEqual(int(video["nb_read_frames"]), n_frames)
            self.assertEqual(video["r_frame_rate"], "25/1")
            self.assertEqual(video["width"], 96)
            self.assertEqual(video["height"], 96)

    def test_unwritable_output_reports_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            # ffmpeg cannot create files in a directory that doesn't exist —
            # the writer must surface that as release() == False, not silence.
            out = Path(tmp) / "missing_subdir" / "bad.mp4"
            writer = FfmpegPipeWriter(
                out, (96, 96), 25.0, codec="libx264", crf=16, preset="veryfast",
            )
            frame = np.zeros((96, 96, 3), dtype=np.uint8)
            for _ in range(10):
                writer.write(frame)
            self.assertFalse(writer.release())


if __name__ == "__main__":
    unittest.main(verbosity=2)
