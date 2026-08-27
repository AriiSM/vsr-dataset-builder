"""
Romanian Text Normalization

Shared helpers for cleaning Whisper output consistently across the pipeline:
- diacritics normalization (legacy cedilla forms → correct comma-below forms)
- word cleaning for transcripts (uppercase, punctuation stripped)
- sentence-boundary detection on raw word text (trailing . ? ! …)

Kept dependency-free (stdlib only) so it can be unit-tested anywhere.
"""

import unicodedata
from typing import Iterable

# Windows-era Romanian text often uses S/T with CEDILLA (U+015F, U+0163).
# Correct Romanian orthography uses COMMA BELOW (U+0219, U+021B).
# Mixing the two silently duplicates vocabulary entries ("aşa" vs "așa"),
# so every text entering the dataset goes through this mapping.
_CEDILLA_TO_COMMA = str.maketrans({
    "ş": "ș",  # ş → ș
    "Ş": "Ș",  # Ş → Ș
    "ţ": "ț",  # ţ → ț
    "Ţ": "Ț",  # Ţ → Ț
})

# Default sentence-ending punctuation (configurable via segmentation config).
DEFAULT_SENTENCE_END_CHARS = (".", "?", "!", "…")  # … = U+2026

# Characters that may trail a word AFTER its sentence-ending punctuation,
# e.g. »Unde?« or "Da." — stripped before checking for sentence end.
_CLOSING_MARKS = "\"'”„»’)]}"


def normalize_diacritics(text: str) -> str:
    """Normalize to NFC and replace cedilla s/t with comma-below forms."""
    if not text:
        return text
    return unicodedata.normalize("NFC", text).translate(_CEDILLA_TO_COMMA)


def clean_word(raw_word: str) -> str:
    """Raw Whisper word → transcript form: diacritics fixed, uppercased,
    surrounding punctuation stripped. Word-internal hyphens/apostrophes are
    kept ("într-o", "s-a")."""
    text = normalize_diacritics(raw_word).strip().upper()
    # Strip leading/trailing non-word characters, keep internal ones.
    start = 0
    end = len(text)
    while start < end and not (text[start].isalnum()):
        start += 1
    while end > start and not (text[end - 1].isalnum()):
        end -= 1
    return text[start:end]


def ends_sentence(
    raw_word: str,
    sentence_end_chars: Iterable[str] = DEFAULT_SENTENCE_END_CHARS,
) -> bool:
    """True if the raw (punctuated) word ends a sentence.

    Trailing quotes/brackets are ignored, so `casa."` and `casa.` both end
    a sentence. An ellipsis counts both as '…' and as '...'.
    """
    if not raw_word:
        return False
    text = raw_word.strip().rstrip(_CLOSING_MARKS)
    if not text:
        return False
    return text[-1] in tuple(sentence_end_chars)
