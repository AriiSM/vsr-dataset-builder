"""Unit tests for ClipManifest (clips.json persistence) — pure stdlib.

Run from the repo root:
    python backend/tests/test_clip_manifest.py
"""

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

from services.segmenter.clip_manifest import ClipManifest, VideoClip  # noqa: E402
from services.segmenter.sentence_segmenter import TimedWord  # noqa: E402


def make_clip(clip_dir: Path, video_id: str, index: int,
              with_words: bool = True) -> VideoClip:
    clip_id = f"{video_id}_clip_{index:03d}"
    video_path = clip_dir / f"{clip_id}.mp4"
    audio_path = clip_dir / f"{clip_id}.wav"
    video_path.write_bytes(b"fake video")
    audio_path.write_bytes(b"fake audio")

    words = None
    if with_words:
        words = [
            TimedWord("amu", "AMU", 10.0, 10.4, 0.95, "SPEAKER_00"),
            TimedWord("îi.", "ÎI", 10.5, 10.9, 0.90, "SPEAKER_00"),
        ]
    return VideoClip(
        clip_id=clip_id, video_id=video_id, clip_index=index,
        start=9.9, end=11.0, video_path=video_path, audio_path=audio_path,
        content_start=9.85, words=words,
        boundary_start_type="silence", boundary_end_type="punctuation",
        audio_speaker_label="SPEAKER_00",
    )


class ManifestRoundtripTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.clip_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_full_roundtrip_preserves_everything(self):
        original = make_clip(self.clip_dir, "md_001", 0)
        ClipManifest.save(self.clip_dir, "md_001", [original])
        restored = ClipManifest.load(self.clip_dir, "md_001")

        self.assertEqual(len(restored), 1)
        clip = restored[0]
        self.assertEqual(clip.clip_id, original.clip_id)
        self.assertEqual(clip.start, original.start)
        self.assertEqual(clip.content_start, original.content_start)
        self.assertEqual(clip.boundary_end_type, "punctuation")
        self.assertEqual(clip.audio_speaker_label, "SPEAKER_00")
        self.assertEqual(len(clip.words), 2)
        self.assertEqual(clip.words[0].clean_text, "AMU")
        self.assertEqual(clip.words[0].speaker, "SPEAKER_00")
        self.assertEqual(clip.words[1].raw_text, "îi.")

    def test_clip_with_missing_media_is_skipped(self):
        kept = make_clip(self.clip_dir, "md_001", 0)
        lost = make_clip(self.clip_dir, "md_001", 1)
        ClipManifest.save(self.clip_dir, "md_001", [kept, lost])
        lost.video_path.unlink()

        restored = ClipManifest.load(self.clip_dir, "md_001")
        self.assertEqual([c.clip_id for c in restored], [kept.clip_id])

    def test_wrong_video_id_returns_none(self):
        ClipManifest.save(self.clip_dir, "md_001", [make_clip(self.clip_dir, "md_001", 0)])
        self.assertIsNone(ClipManifest.load(self.clip_dir, "md_999"))

    def test_corrupt_json_returns_none(self):
        (self.clip_dir / "clips.json").write_text("{ nu e json valid")
        self.assertIsNone(ClipManifest.load(self.clip_dir, "md_001"))

    def test_missing_manifest_returns_none(self):
        self.assertIsNone(ClipManifest.load(self.clip_dir, "md_001"))

    def test_legacy_clip_without_words_loads_with_none(self):
        legacy = make_clip(self.clip_dir, "md_001", 0, with_words=False)
        ClipManifest.save(self.clip_dir, "md_001", [legacy])
        restored = ClipManifest.load(self.clip_dir, "md_001")
        self.assertIsNone(restored[0].words)
        self.assertEqual(restored[0].audio_speaker_label, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
