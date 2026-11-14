"""Unit tests for Romanian text normalization helpers.

Run from the repo root:
    python backend/tests/test_text_normalization.py
"""

import sys
import unittest
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_BACKEND_DIR), str(_BACKEND_DIR / "shared")]

from vsr_shared.text_normalization import (  # noqa: E402
    clean_word,
    ends_sentence,
    normalize_diacritics,
)


class DiacriticsTests(unittest.TestCase):
    def test_cedilla_forms_become_comma_below(self):
        self.assertEqual(normalize_diacritics("aşa şi ţara"), "așa și țara")
        self.assertEqual(normalize_diacritics("Şcoala Ţării"), "Școala Țării")

    def test_correct_forms_unchanged(self):
        self.assertEqual(normalize_diacritics("așa și țara"), "așa și țara")

    def test_empty(self):
        self.assertEqual(normalize_diacritics(""), "")


class CleanWordTests(unittest.TestCase):
    def test_strips_punctuation_and_uppercases(self):
        self.assertEqual(clean_word("casa."), "CASA")
        self.assertEqual(clean_word("«bine»,"), "BINE")

    def test_keeps_internal_hyphen_and_apostrophe(self):
        self.assertEqual(clean_word("într-o"), "ÎNTR-O")
        self.assertEqual(clean_word("s-a."), "S-A")

    def test_normalizes_diacritics(self):
        self.assertEqual(clean_word("gătéşti"), "GĂTÉȘTI".upper())
        self.assertEqual(clean_word("ţară!"), "ȚARĂ")


class SentenceEndTests(unittest.TestCase):
    def test_basic_terminators(self):
        for word in ("gata.", "unde?", "nu!", "păi…"):
            self.assertTrue(ends_sentence(word), word)

    def test_non_terminators(self):
        for word in ("gata", "unde,", "nu;", "1.5"):
            self.assertFalse(ends_sentence(word), word)

    def test_terminator_before_closing_quote(self):
        self.assertTrue(ends_sentence('gata."'))
        self.assertTrue(ends_sentence("gata.”"))
        self.assertTrue(ends_sentence("gata.»"))

    def test_ellipsis_ascii(self):
        self.assertTrue(ends_sentence("păi..."))


if __name__ == "__main__":
    unittest.main(verbosity=2)
