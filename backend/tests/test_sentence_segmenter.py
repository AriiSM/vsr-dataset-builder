"""Unit tests for the sentence segmenter (pure logic, no ML dependencies).

Run from the repo root:
    python backend/tests/test_sentence_segmenter.py
"""

import sys
import unittest
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_BACKEND_DIR), str(_BACKEND_DIR / "shared")]

from services.segmenter.sentence_segmenter import (  # noqa: E402
    BOUNDARY_PUNCTUATION,
    BOUNDARY_SILENCE,
    BOUNDARY_SPEAKER_CHANGE,
    BOUNDARY_WORD_GAP,
    SegmentationSettings,
    SentenceSegmenter,
    TimedWord,
    build_sentence_windows,
)


def make_words(specs):
    """specs: list of (raw_text, start, end)."""
    return [
        TimedWord(raw_text=raw, clean_text=raw.strip(".?!…").upper(),
                  start=start, end=end, confidence=0.9)
        for raw, start, end in specs
    ]


def one_region(words, margin=5.0):
    """A single VAD region covering all words."""
    return [(words[0].start - margin, words[-1].end + margin)]


class SentenceSplittingTests(unittest.TestCase):
    def setUp(self):
        self.settings = SegmentationSettings()

    def test_splits_at_punctuation(self):
        words = make_words([
            ("azi", 0.0, 0.4), ("plouă.", 0.5, 0.9),
            ("mâine", 3.2, 3.6), ("ninge", 3.7, 4.0), ("mult.", 4.1, 4.5),
        ])
        windows = build_sentence_windows(words, one_region(words), self.settings)
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0].text, "AZI PLOUĂ")
        self.assertEqual(windows[0].boundary_end_type, BOUNDARY_PUNCTUATION)
        self.assertEqual(windows[1].text, "MÂINE NINGE MULT")

    def test_splits_at_long_silence_without_punctuation(self):
        words = make_words([
            ("am", 0.0, 0.3), ("zis", 0.4, 0.8), ("că", 0.9, 1.2),
            # 2-second silence: speaker restarted without finishing
            ("vremea", 3.2, 3.6), ("e", 3.7, 3.8), ("bună", 3.9, 4.3),
        ])
        settings = SegmentationSettings(target_min_duration=0.5)
        windows = build_sentence_windows(words, one_region(words), settings)
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0].boundary_end_type, BOUNDARY_SILENCE)

    def test_merges_short_sentences(self):
        words = make_words([
            ("da.", 0.0, 0.4),
            ("sigur.", 0.7, 1.2),
            ("mergem.", 1.5, 2.1),
        ])
        windows = build_sentence_windows(words, one_region(words), self.settings)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].text, "DA SIGUR MERGEM")

    def test_never_merges_across_long_pause(self):
        words = make_words([
            ("da.", 0.0, 0.4),
            # 3-second pause > merge_gap_max — unrelated utterances
            ("sigur.", 3.4, 3.9),
        ])
        windows = build_sentence_windows(words, one_region(words), self.settings)
        self.assertEqual(len(windows), 0)  # both under hard_min after no merge

    def test_overlong_run_on_splits_at_largest_gap_not_mid_word(self):
        # SAFETY-NET path: with an explicit max duration set, 22 seconds of
        # continuous speech (no punctuation, no pause above the 0.5 s
        # threshold) must split at the LARGEST pause — 0.4 s at 12.0→12.4 —
        # never at a blind duration cut.
        self.settings = SegmentationSettings(hard_max_duration=15.0)
        specs = []
        t = 0.0
        while t < 11.7:
            specs.append((f"w{len(specs)}", t, t + 0.3))
            t += 0.4  # 0.1 s gaps
        specs.append(("pauza", 11.7, 12.0))
        t = 12.4  # the 0.4 s pause — largest in the run, still < threshold
        while t < 22.0:
            specs.append((f"w{len(specs)}", t, t + 0.3))
            t += 0.4
        words = make_words(specs)
        windows = build_sentence_windows(words, one_region(words), self.settings)

        self.assertTrue(all(w.duration <= self.settings.hard_max_duration + 0.5
                            for w in windows))
        # The split must land exactly in the 12.0–12.7 pause
        boundary_types = [w.boundary_end_type for w in windows[:-1]]
        self.assertIn(BOUNDARY_WORD_GAP, boundary_types)
        first = windows[0]
        self.assertAlmostEqual(first.speech_end, 12.0, delta=0.01)
        # No window may cut inside any word
        for window in windows:
            for word in words:
                inside_start = window.start > word.start + 1e-6 and window.start < word.end - 1e-6
                inside_end = window.end > word.start + 1e-6 and window.end < word.end - 1e-6
                self.assertFalse(inside_start or inside_end,
                                 f"window edge cuts word {word.raw_text}")

    def test_drops_hallucinated_words_outside_vad(self):
        words = make_words([
            ("muzică", 0.0, 0.5),      # hallucination over intro music
            ("bună", 10.0, 10.4), ("ziua", 10.5, 10.9), ("tuturor.", 11.0, 11.6),
        ])
        speech_regions = [(9.8, 12.0)]  # VAD only saw the greeting
        windows = build_sentence_windows(words, speech_regions,
                                         SegmentationSettings(target_min_duration=0.5))
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].text, "BUNĂ ZIUA TUTUROR")

    def test_neighbouring_windows_do_not_overlap(self):
        words = make_words([
            ("unu", 0.0, 1.0), ("doi", 1.1, 2.0), ("trei.", 2.1, 3.0),
            ("patru", 3.2, 4.2), ("cinci", 4.3, 5.3), ("șase.", 5.4, 6.4),
        ])
        windows = build_sentence_windows(words, one_region(words), self.settings)
        for earlier, later in zip(windows, windows[1:]):
            self.assertLessEqual(earlier.end, later.start + 1e-9)

    def test_empty_input(self):
        self.assertEqual(build_sentence_windows([], [], self.settings), [])


class BoundaryPaddingTests(unittest.TestCase):
    def test_padding_is_applied_and_clamped_at_zero(self):
        words = make_words([("start", 0.05, 0.5), ("imediat", 0.6, 1.0),
                            ("acum.", 1.1, 2.6)])
        windows = build_sentence_windows(words, one_region(words),
                                         SegmentationSettings())
        self.assertGreaterEqual(windows[0].start, 0.0)
        self.assertLess(windows[0].start, words[0].start)
        self.assertGreater(windows[0].end, words[-1].end)


class SpeakerAwareTests(unittest.TestCase):
    def _words_two_voices(self):
        """Fast exchange, NO pause between replies, no punctuation:
        reporter (S0) asks, guest (S1) answers immediately."""
        specs = [
            ("și", 0.0, 0.3, "SPEAKER_00"), ("atunci", 0.35, 0.7, "SPEAKER_00"),
            ("ce", 0.75, 0.9, "SPEAKER_00"), ("faceți", 0.95, 1.4, "SPEAKER_00"),
            # 0.1 s gap only — below the silence threshold
            ("păi", 1.5, 1.8, "SPEAKER_01"), ("lucrăm", 1.85, 2.3, "SPEAKER_01"),
            ("mult", 2.35, 2.7, "SPEAKER_01"),
        ]
        return [
            TimedWord(raw, raw.upper(), start, end, 0.9, speaker)
            for raw, start, end, speaker in specs
        ]

    def test_splits_at_speaker_change_without_pause(self):
        words = self._words_two_voices()
        settings = SegmentationSettings(target_min_duration=0.5)
        windows = SentenceSegmenter(settings).build_windows(
            words, [(0.0, 3.0)])
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0].boundary_end_type, BOUNDARY_SPEAKER_CHANGE)
        self.assertEqual(windows[0].audio_speaker_label, "SPEAKER_00")
        self.assertEqual(windows[1].audio_speaker_label, "SPEAKER_01")
        self.assertEqual(windows[0].text, "ȘI ATUNCI CE FACEȚI")
        self.assertEqual(windows[1].text, "PĂI LUCRĂM MULT")

    def test_short_sentences_never_merge_across_voices(self):
        # Two very short replies from different voices, close together —
        # duration-wise they would merge; the voice guard must forbid it.
        words = [
            TimedWord("da.", "DA", 0.0, 0.6, 0.9, "SPEAKER_00"),
            TimedWord("sigur.", "SIGUR", 0.8, 1.4, 0.9, "SPEAKER_01"),
        ]
        settings = SegmentationSettings(target_min_duration=2.0,
                                        hard_min_duration=0.3)
        windows = SentenceSegmenter(settings).build_windows(words, [(0.0, 2.0)])
        self.assertEqual(len(windows), 2)

    def test_no_length_limit_by_default_keeps_run_on_whole(self):
        # 22 s of one voice, no punctuation, no real pause: with the default
        # hard_max_duration=None the run-on stays ONE clip.
        specs = []
        t = 0.0
        while t < 22.0:
            specs.append((f"w{len(specs)}", t, t + 0.3))
            t += 0.4
        words = make_words(specs)
        windows = build_sentence_windows(
            words, one_region(words), SegmentationSettings())
        self.assertEqual(len(windows), 1)
        self.assertGreater(windows[0].duration, 20.0)

    def test_unlabeled_words_never_force_boundaries(self):
        # Diarization off (speaker=None everywhere) → behaves exactly like
        # the punctuation+pause segmenter.
        words = make_words([("azi", 0.0, 0.4), ("plouă.", 0.5, 0.9),
                            ("mâine", 1.2, 1.6), ("nu.", 1.7, 2.1)])
        windows = build_sentence_windows(
            words, one_region(words), SegmentationSettings(target_min_duration=0.5))
        self.assertTrue(all(w.audio_speaker_label == "" for w in windows))


if __name__ == "__main__":
    unittest.main(verbosity=2)
