"""
Audio slicer — the segment's AUDIO artifact.

Cuts audio/{segment_id}.wav out of the clip's analysis wav by pure sample
math (stdlib `wave` — PCM in, PCM out, no external process, no new
dependency). The clip wav and the exported segment share the same clip-file
time coordinates, so slicing is exact by construction.
"""

import wave
from pathlib import Path

from loguru import logger


def write_segment_audio(
    clip_wav_path: Path,
    output_wav_path: Path,
    start_time: float,
    end_time: float,
) -> bool:
    """Write [start_time, end_time] of the clip wav to output_wav_path.

    Times are clip-file seconds. Frame indices are clamped to the available
    range. Returns False (with a log line) instead of raising — a missing
    segment audio must never kill the export of the video crops.
    """
    try:
        with wave.open(str(clip_wav_path), "rb") as source:
            sample_rate = source.getframerate()
            total_frames = source.getnframes()

            start_frame = max(0, int(start_time * sample_rate))
            end_frame = min(total_frames, int(end_time * sample_rate))
            if end_frame <= start_frame:
                logger.warning(
                    f"Empty audio window for {output_wav_path.name} "
                    f"[{start_time:.2f}-{end_time:.2f}s]"
                )
                return False

            source.setpos(start_frame)
            payload = source.readframes(end_frame - start_frame)

            with wave.open(str(output_wav_path), "wb") as target:
                target.setnchannels(source.getnchannels())
                target.setsampwidth(source.getsampwidth())
                target.setframerate(sample_rate)
                target.writeframes(payload)
        return True
    except Exception as e:
        logger.warning(f"Segment audio export failed for {output_wav_path.name}: {e}")
        return False
