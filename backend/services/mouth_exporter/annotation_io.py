"""
Annotation I/O — the segment's TEXT artifact (LRS2 format).

write_annotation() produces the .txt every exported segment ships with:
    Text:      current transcript (reviewers may edit it later)
    Original:  the Whisper transcript at export time (WER reference)
    Conf:      1/2/3 from min+mean word confidence
    WORD START END ASDSCORE  — every word, timed relative to the segment

parse_lrs2_annotation() reads it back (pipeline recovery, review tools).
"""

from pathlib import Path
from typing import Any, Dict, List

from loguru import logger


def write_annotation(
    output_path: Path,
    transcription: Any,  # TranscribedSegment
    asd_scores: List[float]
):
    """
    Export annotation in LRS2 format.

    Format:
        Text:      TRANSCRIBED TEXT
        Original:  TRANSCRIBED TEXT
        Conf:      2
        WORD START END ASDSCORE
        WORD1 0.00 0.25 10.5
        ...

    Conf is derived from BOTH min and mean word confidence (see
    transcribe._conf_level) so a single weak word drags the segment
    down even if the rest of the words are very confident.

    `Original:` captures the Whisper raw transcript at export time.
    Reviewers may later edit `Text:` — the WER is then computed against
    `Original:` as the reference. At first export, Original = Text.
    """
    from services.segmenter.transcription import _conf_level

    text = transcription.text.upper()
    conf_level = _conf_level(transcription.confidence_min, transcription.confidence)

    lines = []
    lines.append(f"Text:  {text}")
    lines.append(f"Original:  {text}")
    lines.append(f"Conf:  {conf_level}")
    lines.append("WORD START END ASDSCORE")

    for i, word in enumerate(transcription.words):
        rel_start = word.start - transcription.start
        rel_end = word.end - transcription.start
        asd_score = asd_scores[i] if i < len(asd_scores) else 8.0
        lines.append(f"{word.text.upper()} {rel_start:.2f} {rel_end:.2f} {asd_score:.1f}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.debug(f"Exported annotation: {output_path}")


def parse_lrs2_annotation(annotation_path: Path) -> Dict[str, Any]:
    """
    Parse an LRS2-format annotation file.

    Returns dict with:
        - text:        current (possibly edited) transcript
        - original:    Whisper raw transcript at export time, or None if absent
                       (legacy annotations without an `Original:` line)
        - confidence:  1/2/3
        - words:       list of {word, start, end, asd_score}
    """
    with open(annotation_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    result: Dict[str, Any] = {
        'text': '',
        'original': None,
        'confidence': 0,
        'words': []
    }

    for line in lines:
        stripped = line.strip()

        # Order matters: check 'Original:' before 'Text:' so we don't capture
        # the longer prefix into the wrong field.
        if stripped.startswith('Original:'):
            result['original'] = stripped[len('Original:'):].strip()
        elif stripped.startswith('Text:'):
            result['text'] = stripped[len('Text:'):].strip()
        elif stripped.startswith('Conf:'):
            try:
                result['confidence'] = int(stripped[len('Conf:'):].strip())
            except ValueError:
                pass
        elif stripped.startswith('WORD START END'):
            continue  # Header line
        elif stripped:
            # Word line: WORD START END ASDSCORE
            parts = stripped.split()
            if len(parts) >= 4:
                result['words'].append({
                    'word': parts[0],
                    'start': float(parts[1]),
                    'end': float(parts[2]),
                    'asd_score': float(parts[3])
                })

    return result
