"""Unit tests for the downloader service's pure helpers (no network).

Run from the repo root:
    python backend/tests/test_youtube_downloader.py
"""

import sys
import types
import unittest
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_BACKEND_DIR), str(_BACKEND_DIR / "shared")]

# Stub heavy/optional deps when absent — the pure helpers never touch them.
for _missing in ("yt_dlp", "loguru"):
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

from services.downloader.youtube_downloader import (  # noqa: E402
    VideoInfo,
    YouTubeDownloader,
)


class YoutubeIdExtractionTests(unittest.TestCase):
    def test_watch_short_embed_and_bare_forms(self):
        for url in (
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://www.youtube.com/embed/dQw4w9WgXcQ",
            "dQw4w9WgXcQ",
        ):
            self.assertEqual(YouTubeDownloader._extract_youtube_id(url), "dQw4w9WgXcQ")

    def test_invalid_input_raises(self):
        with self.assertRaises(ValueError):
            YouTubeDownloader._extract_youtube_id("asta nu e un url valid")


class ParsingTests(unittest.TestCase):
    def test_rate_limit(self):
        self.assertEqual(YouTubeDownloader._parse_rate_limit("5M"), 5 * 1024 ** 2)
        self.assertEqual(YouTubeDownloader._parse_rate_limit("500K"), 500 * 1024)
        self.assertEqual(YouTubeDownloader._parse_rate_limit("garbage"), 5 * 1024 ** 2)

    def test_frame_rate(self):
        self.assertAlmostEqual(
            YouTubeDownloader._parse_frame_rate("30000/1001"), 29.97, places=2)
        self.assertEqual(YouTubeDownloader._parse_frame_rate("25/1"), 25.0)
        self.assertEqual(YouTubeDownloader._parse_frame_rate("0/0"), 25.0)
        self.assertEqual(YouTubeDownloader._parse_frame_rate("junk"), 25.0)


class BestResolutionTests(unittest.TestCase):
    def test_picks_max_height_not_first(self):
        raw = {"formats": [
            {"vcodec": "avc1", "height": 360, "width": 640, "fps": 25},
            {"vcodec": "avc1", "height": 1080, "width": 1920, "fps": 30},
            {"vcodec": "none", "height": None},
        ]}
        resolution, fps = YouTubeDownloader._best_available_resolution(raw)
        self.assertEqual(resolution, "1920x1080")
        self.assertEqual(fps, 30)

    def test_no_formats(self):
        self.assertEqual(
            YouTubeDownloader._best_available_resolution({}), ("unknown", 25.0))


class CreativeCommonsTests(unittest.TestCase):
    @staticmethod
    def _info(license_text="", description=""):
        return VideoInfo(
            video_id="x", title="t", channel="c", duration=10,
            license=license_text, description=description,
            resolution="1920x1080", fps=25.0,
        )

    def test_license_field_detected(self):
        self.assertTrue(self._info("Creative Commons Attribution").is_creative_commons)

    def test_description_marker_detected(self):
        self.assertTrue(self._info("", "sursa: creativecommons.org/...").is_creative_commons)

    def test_plain_video_rejected(self):
        self.assertFalse(self._info("", "descriere obișnuită").is_creative_commons)


if __name__ == "__main__":
    unittest.main(verbosity=2)
