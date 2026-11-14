"""
Word Error Rate utilities.

Used by the pipeline (recompute WER on sync-from-disk), the frontend
(recompute WER after a manual edit / trim), and scripts/backfill_wer.py
(populate WER for the existing dataset).

Reference = Whisper original transcript captured at export time
            (the `Original:` line in the LRS2 annotation file).
Hypothesis = current `Text:` line (possibly edited by a reviewer).

If jiwer is installed it is used (canonical implementation). Otherwise
we fall back to a tokenised Levenshtein distance — same answer for
typical cases, no extra dependency.
"""

from __future__ import annotations

from typing import Optional, Tuple


def _normalise(text: str) -> list[str]:
    """Tokenise on whitespace, uppercase, strip basic punctuation.

    Whisper export uses uppercase tokens; we mirror that so a reviewer
    typing in mixed case doesn't inflate WER artificially.
    """
    if not text:
        return []
    out: list[str] = []
    for raw in text.split():
        cleaned = raw.strip(".,;:!?\"'()[]").upper()
        if cleaned:
            out.append(cleaned)
    return out


def _wer_levenshtein(ref_tokens: list[str], hyp_tokens: list[str]) -> float:
    """Word-level Levenshtein distance / len(ref). 0.0 if both empty."""
    n_ref = len(ref_tokens)
    n_hyp = len(hyp_tokens)
    if n_ref == 0 and n_hyp == 0:
        return 0.0
    if n_ref == 0:
        return 1.0  # all insertions
    # Standard DP edit distance with substitution cost = 1.
    prev = list(range(n_hyp + 1))
    for i in range(1, n_ref + 1):
        curr = [i] + [0] * n_hyp
        for j in range(1, n_hyp + 1):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost, # substitution
            )
        prev = curr
    return prev[n_hyp] / n_ref


def compute_wer(
    original: Optional[str],
    current: Optional[str],
) -> Tuple[Optional[float], Optional[int]]:
    """WER between Whisper-original and current text.

    Returns:
        (wer, ref_word_count) — both None if original is missing/empty,
        in which case there is no reference to score against.
    """
    if original is None or not original.strip():
        return None, None

    ref = _normalise(original)
    hyp = _normalise(current or "")
    ref_word_count = len(ref)

    try:
        import jiwer  # type: ignore
        wer_value = float(jiwer.wer(" ".join(ref), " ".join(hyp))) if ref else 0.0
    except ImportError:
        wer_value = _wer_levenshtein(ref, hyp)

    return round(wer_value, 4), ref_word_count
