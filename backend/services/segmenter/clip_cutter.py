"""
Clip Cutter (ffmpeg)

Cuts one clip out of the source video: ONE ffmpeg invocation, ONE decode,
TWO outputs produced together —
    clip.mp4  — VIDEO ONLY (H.264 CFR 25 fps), for face tracking and the
                crop export;
    clip.wav  — AUDIO ONLY (16 kHz mono PCM) — the single audio source for
                ASD/SyncNet MFCCs and for muxing into the exported segments.

One modality per file: no duplicated audio, no repeated extraction from mp4.

Cutting both modalities from the same source in the same invocation with the
same window means audio/video can never drift apart.
"""

import subprocess
from pathlib import Path
from typing import Optional

from loguru import logger


class ClipCutter:
    """Frame-accurate clip extraction with configurable H.264 quality.

    clip_crf / clip_preset come from config (segmentation.clip_crf /
    clip_preset). Keep CRF low (16): these clips are the source for the
    final crop export — every generation of loss lands on the lips.
    """

    OUTPUT_FPS = 25          # pipeline-wide constant frame rate

    def __init__(
        self,
        clip_crf: int,
        clip_preset: Optional[str],
        seek_start_padding: float,
        audio_sample_rate: int = 16000,
    ):
        self.clip_crf = clip_crf
        self.clip_preset = clip_preset
        # Small lead-in before the speech start so word onsets are never
        # chopped by an imprecise seek.
        self.seek_start_padding = seek_start_padding
        # From config (audio.sample_rate) — the same rate VAD/Whisper/MFCC use.
        self.audio_sample_rate = audio_sample_rate

    def cut(
        self,
        source_video: Path,
        window_start: float,
        window_end: float,
        output_video: Path,
        output_audio: Path,
    ) -> Optional[float]:
        """Cut [window_start, window_end] (source-video seconds) into
        output_video + output_audio.

        Returns content_start — the actual second of the source video where
        the produced files begin (= window_start − pre-roll padding) — or
        None when cutting failed. content_start is the anchor for every
        absolute↔clip time conversion downstream.
        """
        content_start = max(0.0, window_start - self.seek_start_padding)
        duration = window_end - content_start

        video_options = [
            "-map", "0:v:0", "-an",     # video only — audio lives in the .wav
            "-c:v", "libx264", "-crf", str(self.clip_crf),
        ]
        if self.clip_preset:
            video_options += ["-preset", self.clip_preset]
        video_options += [
            # Force CFR at the pipeline fps: 30/60 fps sources would otherwise
            # produce more frames than the audio duration accounts for.
            "-r", str(self.OUTPUT_FPS),
            str(output_video),
        ]
        audio_options = [
            "-map", "0:a:0", "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(self.audio_sample_rate), "-ac", "1",
            str(output_audio),
        ]

        # ONE invocation, ONE decode, both outputs (video seek is input-side
        # for speed; ffmpeg writes every "-map ... output" group it is given).
        command = [
            self._find_ffmpeg(), "-y",
            "-ss", str(content_start),
            "-t", str(duration),
            "-i", str(source_video),
            *video_options,
            *audio_options,
        ]

        try:
            subprocess.run(command, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            stderr_tail = e.stderr.decode(errors="replace")[-200:]
            logger.error(
                f"ffmpeg failed cutting [{window_start:.2f}-{window_end:.2f}s]: "
                f"{stderr_tail}"
            )
            return None

        produced_ok = (
            output_video.exists() and output_video.stat().st_size > 0
            and output_audio.exists() and output_audio.stat().st_size > 0
        )
        return content_start if produced_ok else None

    @staticmethod
    def _find_ffmpeg() -> str:
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            return "ffmpeg"
