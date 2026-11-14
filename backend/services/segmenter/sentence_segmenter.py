"""
Sentence Segmenter

Builds sentence-level cutting windows from full-video Whisper word timestamps
combined with Silero VAD speech regions. This replaces silence-only splitting:
clip boundaries follow LINGUISTIC sentence ends (punctuation) and real speech
pauses, and a too-long sentence is split at its LARGEST inter-word pause —
never mid-word, never at a blind duration limit.

Pure logic module: no I/O, no ML imports — fully unit-testable.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from vsr_shared.text_normalization import DEFAULT_SENTENCE_END_CHARS, ends_sentence


# Boundary provenance markers, recorded per window edge and exported as
# metadata (a "forced" edge caps the segment's quality tier later):
BOUNDARY_PUNCTUATION = "punctuation"        # sentence-ending punctuation
BOUNDARY_SILENCE = "silence"                # VAD pause >= split threshold
BOUNDARY_SPEAKER_CHANGE = "speaker_change"  # diarization: another voice starts
BOUNDARY_WORD_GAP = "word_gap"              # largest pause inside an over-long sentence
BOUNDARY_MEDIA = "media"                    # start/end of the speech in the video


@dataclass
class TimedWord:
    """One word from full-video transcription, in source-video seconds."""
    raw_text: str                      # as Whisper produced it, punctuation kept
    clean_text: str                    # transcript form (upper, no punctuation)
    start: float
    end: float
    confidence: Optional[float] = None
    speaker: Optional[str] = None      # diarization voice label (SPEAKER_00…);
                                       # None = diarization off or no coverage


@dataclass
class SentenceWindow:
    """A cutting window covering one sentence (or merged short sentences)."""
    start: float                       # padded cut-in point (source seconds)
    end: float                         # padded cut-out point
    words: List[TimedWord] = field(default_factory=list)
    boundary_start_type: str = BOUNDARY_MEDIA
    boundary_end_type: str = BOUNDARY_MEDIA

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def speech_start(self) -> float:
        return self.words[0].start if self.words else self.start

    @property
    def speech_end(self) -> float:
        return self.words[-1].end if self.words else self.end

    @property
    def text(self) -> str:
        return " ".join(w.clean_text for w in self.words if w.clean_text)

    @property
    def audio_speaker_label(self) -> str:
        """Dominant diarization voice of this window ("" when unlabeled)."""
        return _dominant_speaker(self.words) or ""


@dataclass
class SegmentationSettings:
    """All knobs come from config.yaml (segmentation: block)."""
    sentence_end_chars: Sequence[str] = DEFAULT_SENTENCE_END_CHARS
    split_silence_threshold: float = 0.5   # pause >= this always separates windows
    target_min_duration: float = 2.0       # merge shorter sentences up toward this
    target_max_duration: float = 12.0      # do not merge beyond this
    hard_min_duration: float = 1.0         # drop windows shorter than this
    # None = NO length limit: boundaries are purely linguistic/speaker-based.
    # A number re-enables the safety net (split at the largest pause above it).
    hard_max_duration: Optional[float] = None
    merge_gap_max: float = 1.0             # never merge across a pause longer than this
    boundary_pad: float = 0.15             # lead-in/lead-out added around speech
    vad_margin: float = 0.5                # words outside VAD regions ± margin are hallucinations


def _dominant_speaker(words) -> Optional[str]:
    """Majority diarization label of a word list (None when unlabeled)."""
    votes = {}
    for word in words:
        if word.speaker:
            votes[word.speaker] = votes.get(word.speaker, 0) + 1
    if not votes:
        return None
    return max(votes, key=votes.get)


def _compatible_speakers(left_words, right_words) -> bool:
    """Two word runs may merge only when they belong to the same voice
    (or at least one side has no diarization label at all)."""
    left = _dominant_speaker(left_words)
    right = _dominant_speaker(right_words)
    return left is None or right is None or left == right


def build_sentence_windows(
    words: List[TimedWord],
    speech_regions: List[Tuple[float, float]],
    settings: SegmentationSettings,
) -> List[SentenceWindow]:
    """Compute sentence-level cutting windows for one video.

    Steps:
      1. Drop words that fall outside VAD speech regions (± margin) —
         Whisper hallucinates text over music/silence.
      2. Split the word stream at sentence-ending punctuation AND at
         silences >= split_silence_threshold.
      3. Merge adjacent short sentences (bounded by target_max_duration and
         merge_gap_max) so 1-word answers don't become 0.4 s clips.
      4. Recursively split windows longer than hard_max_duration at their
         largest inter-word pause — never mid-word.
      5. Pad boundaries into the surrounding silence (clamped to gap midpoints
         so neighbouring clips never overlap) and drop sub-hard_min windows.
    """
    kept = _drop_words_outside_speech(words, speech_regions, settings.vad_margin)
    if not kept:
        return []

    units = _split_at_sentence_ends_and_silences(kept, settings)
    units = _merge_short_units(units, settings)
    units = _split_overlong_units(units, settings)
    windows = _finalize_boundaries(units, settings)
    return [w for w in windows if w.duration >= settings.hard_min_duration and w.words]


# ---------------------------------------------------------------- internals


@dataclass
class _Unit:
    """A run of words plus the reason it ended (and started)."""
    words: List[TimedWord]
    start_type: str
    end_type: str

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def duration(self) -> float:
        return self.end - self.start


def _drop_words_outside_speech(
    words: List[TimedWord],
    speech_regions: List[Tuple[float, float]],
    margin: float,
) -> List[TimedWord]:
    """Keep only words overlapping some VAD speech region expanded by margin."""
    if not speech_regions:
        # No VAD signal at all — trust Whisper rather than dropping everything.
        return list(words)
    kept = []
    for word in words:
        for region_start, region_end in speech_regions:
            if word.end > region_start - margin and word.start < region_end + margin:
                kept.append(word)
                break
    return kept


def _split_at_sentence_ends_and_silences(
    words: List[TimedWord],
    settings: SegmentationSettings,
) -> List["_Unit"]:
    units: List[_Unit] = []
    current: List[TimedWord] = []
    start_type = BOUNDARY_MEDIA

    for i, word in enumerate(words):
        current.append(word)

        is_last = i == len(words) - 1
        gap_after = (words[i + 1].start - word.end) if not is_last else None

        next_word = words[i + 1] if not is_last else None

        end_type = None
        if is_last:
            end_type = BOUNDARY_MEDIA
        elif ends_sentence(word.raw_text, settings.sentence_end_chars):
            end_type = BOUNDARY_PUNCTUATION
        elif (
            next_word.speaker is not None
            and word.speaker is not None
            and next_word.speaker != word.speaker
        ):
            # Another voice takes over (diarization) — a reply boundary,
            # even in fast exchanges with no pause at all.
            end_type = BOUNDARY_SPEAKER_CHANGE
        elif gap_after is not None and gap_after >= settings.split_silence_threshold:
            # The speaker stopped mid-sentence (restart, interruption) —
            # a real pause still separates clips even without punctuation.
            end_type = BOUNDARY_SILENCE

        if end_type is not None:
            units.append(_Unit(current, start_type, end_type))
            # The next unit starts for the same reason this one ended.
            start_type = end_type
            current = []

    return units


def _merge_short_units(
    units: List["_Unit"],
    settings: SegmentationSettings,
) -> List["_Unit"]:
    """Greedily merge consecutive short sentences into training-friendly clips.

    A merge happens only when the current window is still below
    target_min_duration, the merged result stays within target_max_duration,
    and the pause between the two units is small (<= merge_gap_max) — merging
    across a long silence would glue unrelated utterances together.
    """
    if not units:
        return []

    merged: List[_Unit] = [units[0]]
    for unit in units[1:]:
        previous = merged[-1]
        gap = unit.start - previous.end
        combined_duration = unit.end - previous.start
        same_voice = _compatible_speakers(previous.words, unit.words)
        if (
            same_voice
            and previous.duration < settings.target_min_duration
            and gap <= settings.merge_gap_max
            and combined_duration <= settings.target_max_duration
        ):
            merged[-1] = _Unit(
                previous.words + unit.words,
                previous.start_type,
                unit.end_type,
            )
        else:
            merged.append(unit)
    return merged


def _split_overlong_units(
    units: List["_Unit"],
    settings: SegmentationSettings,
) -> List["_Unit"]:
    if settings.hard_max_duration is None:
        return units  # no length limit — never force-split a sentence
    result: List[_Unit] = []
    for unit in units:
        result.extend(_split_unit_at_largest_gap(unit, settings))
    return result


def _split_unit_at_largest_gap(
    unit: "_Unit",
    settings: SegmentationSettings,
) -> List["_Unit"]:
    """Recursively split an over-long unit at its largest inter-word pause.

    The boundary always lands between two words (largest pause = most natural
    break available). Single-word units longer than hard_max cannot be split
    and are returned as-is (flagged by duration downstream).
    """
    if unit.duration <= settings.hard_max_duration or len(unit.words) < 2:
        return [unit]

    gaps = [
        (unit.words[i + 1].start - unit.words[i].end, i)
        for i in range(len(unit.words) - 1)
    ]
    # Prefer the largest pause; ties resolve toward the middle of the unit so
    # the two halves stay balanced.
    mid = (len(unit.words) - 1) / 2.0
    _, split_index = max(gaps, key=lambda g: (g[0], -abs(g[1] - mid)))

    left = _Unit(unit.words[: split_index + 1], unit.start_type, BOUNDARY_WORD_GAP)
    right = _Unit(unit.words[split_index + 1:], BOUNDARY_WORD_GAP, unit.end_type)
    return (
        _split_unit_at_largest_gap(left, settings)
        + _split_unit_at_largest_gap(right, settings)
    )


def _finalize_boundaries(
    units: List["_Unit"],
    settings: SegmentationSettings,
) -> List[SentenceWindow]:
    """Pad each unit into the surrounding silence without overlapping neighbours.

    The cut-in point is speech_start - boundary_pad, clamped to the midpoint of
    the pause before the unit (same on the other side), and never negative.
    """
    windows: List[SentenceWindow] = []
    for i, unit in enumerate(units):
        pad_before = settings.boundary_pad
        pad_after = settings.boundary_pad

        if i > 0:
            half_gap_before = max(0.0, (unit.start - units[i - 1].end) / 2.0)
            pad_before = min(pad_before, half_gap_before)
        if i < len(units) - 1:
            half_gap_after = max(0.0, (units[i + 1].start - unit.end) / 2.0)
            pad_after = min(pad_after, half_gap_after)

        windows.append(SentenceWindow(
            start=max(0.0, unit.start - pad_before),
            end=unit.end + pad_after,
            words=list(unit.words),
            boundary_start_type=unit.start_type,
            boundary_end_type=unit.end_type,
        ))
    return windows


class SentenceSegmenter:
    """Decides the cutting windows for one video.

    Input : the full-video word stream (with optional diarization labels)
            + the VAD speech regions.
    Output: SentenceWindow list — boundaries at punctuation ∪ real pauses ∪
            speaker changes; short sentences merged (never across voices);
            no blind duration cut (hard_max_duration=None by default).

    Thin, stateless orchestration over the pure functions in this module —
    configured once from config.yaml's `segmentation:` block.
    """

    def __init__(self, settings: SegmentationSettings):
        self.settings = settings

    def build_windows(
        self,
        words: List[TimedWord],
        speech_regions: List[Tuple[float, float]],
    ) -> List[SentenceWindow]:
        return build_sentence_windows(words, speech_regions, self.settings)
