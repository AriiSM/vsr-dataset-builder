"""Unit tests for the diarizer's pure overlap-labeling logic (no pyannote).

Run from the repo root:
    python backend/tests/test_diarization.py
"""

import sys
import types
import unittest
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_BACKEND_DIR), str(_BACKEND_DIR / "shared")]

for _missing in ("torch", "loguru"):
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

from services.segmenter.diarization import SpeakerDiarizer  # noqa: E402
from services.segmenter.sentence_segmenter import TimedWord  # noqa: E402


def word(start, end):
    return TimedWord("w", "W", start, end, 0.9)


class OverlapLabelingTests(unittest.TestCase):
    def test_words_get_the_most_overlapping_turn(self):
        words = [word(0.0, 1.0), word(1.2, 2.0), word(5.0, 6.0)]
        turns = [(0.0, 2.1, "SPEAKER_00"), (4.8, 7.0, "SPEAKER_01")]
        labeled = SpeakerDiarizer.label_words_by_overlap(words, turns)
        self.assertEqual(labeled, 3)
        self.assertEqual([w.speaker for w in words],
                         ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01"])

    def test_word_straddling_turns_takes_the_bigger_overlap(self):
        straddler = word(1.8, 2.6)  # 0.2s in turn A, 0.6s in turn B
        turns = [(0.0, 2.0, "SPEAKER_00"), (2.0, 5.0, "SPEAKER_01")]
        SpeakerDiarizer.label_words_by_overlap([straddler], turns)
        self.assertEqual(straddler.speaker, "SPEAKER_01")

    def test_word_outside_all_turns_stays_unlabeled(self):
        outside = word(10.0, 10.5)
        turns = [(0.0, 2.0, "SPEAKER_00")]
        labeled = SpeakerDiarizer.label_words_by_overlap([outside], turns)
        self.assertEqual(labeled, 0)
        self.assertIsNone(outside.speaker)

    def test_no_turns_labels_nothing(self):
        words = [word(0.0, 1.0)]
        self.assertEqual(SpeakerDiarizer.label_words_by_overlap(words, []), 0)
        self.assertIsNone(words[0].speaker)


if __name__ == "__main__":
    unittest.main(verbosity=2)
