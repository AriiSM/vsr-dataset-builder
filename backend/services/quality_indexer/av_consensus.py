"""
Voice↔face consensus — the diarization cross-check.

Two independent systems name the speaker of every segment:
  - pyannote hears a VOICE   → audio_speaker_label (SPEAKER_00, ...)
  - ArcFace sees a FACE      → speaker_id (md_001_spk0, ...)

Per video, the majority vote decides which face each voice belongs to.
Segments that contradict their voice's majority face get flagged
av_speaker_mismatch=True — classic voice-over / B-roll / wrong-track cases,
poison for lip reading. Nothing is deleted: the flag caps the quality tier
at B and surfaces the segment for review.

Pure logic (stdlib only) — needs the whole video by nature (a majority
exists only across all segments), but costs microseconds and touches no
files. Runs at end-of-video right after speaker clustering.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class ConsensusResult:
    """Outcome of the per-video voice↔face majority vote."""
    voice_to_face: Dict[str, str] = field(default_factory=dict)
    # segment_id → True (contradicts majority) / False (agrees).
    # Segments missing either label are absent — they were not judged.
    mismatch_by_segment: Dict[str, bool] = field(default_factory=dict)

    @property
    def num_judged(self) -> int:
        return len(self.mismatch_by_segment)

    @property
    def num_mismatched(self) -> int:
        return sum(1 for flag in self.mismatch_by_segment.values() if flag)


def compute_av_consensus(
    segments: List[Tuple[str, str, str]],
) -> ConsensusResult:
    """Majority-vote the voice→face mapping and flag the contradictions.

    Args:
        segments: (segment_id, audio_speaker_label, speaker_id) per exported
            segment. Entries with either label blank are skipped (no
            diarization, or no identity evidence) — never flagged.

    Returns:
        ConsensusResult with the winning mapping and per-segment verdicts.
    """
    result = ConsensusResult()

    votes: Dict[str, Dict[str, int]] = {}
    judged: List[Tuple[str, str, str]] = []
    for segment_id, voice, face in segments:
        if not voice or not face:
            continue
        judged.append((segment_id, voice, face))
        votes.setdefault(voice, {}).setdefault(face, 0)
        votes[voice][face] += 1

    # Majority face per voice; deterministic tie-break by name so reruns
    # produce identical flags.
    for voice, face_counts in votes.items():
        result.voice_to_face[voice] = max(
            sorted(face_counts), key=lambda f: face_counts[f],
        )

    for segment_id, voice, face in judged:
        result.mismatch_by_segment[segment_id] = (
            face != result.voice_to_face[voice]
        )
    return result
