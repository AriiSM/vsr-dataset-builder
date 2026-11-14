"""
Video encoding for exported crops.

FfmpegPipeWriter — the single-encode path (default): BGR frames stream
through stdin straight into the final H.264 file, audio muxed in the same
invocation. Verified live (Phase 2 tests): exact CFR, exact frame count.

reencode_with_audio — the legacy rollback path (export.use_ffmpeg_pipe:
false): mp4v temp file re-encoded with audio. One extra lossy generation;
kept only as an escape hatch.
"""

import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger

# Process-wide x264 thread cap (performance.encoder_threads in config.yaml,
# applied by VSRPipeline at startup); 0 = ffmpeg's default (all cores).
# Uncapped x264 grabs every core per encode and starves the analysis thread.
ENCODER_THREADS = 0


def _thread_args() -> List[str]:
    return ["-threads", str(ENCODER_THREADS)] if ENCODER_THREADS > 0 else []


class FfmpegPipeWriter:
    """Stream BGR frames straight into ONE final H.264 encode (+ audio mux).

    Replaces the legacy two-step export (OpenCV mp4v temp file → ffmpeg
    re-encode), which added an extra lossy generation to every crop. Frames
    go through stdin as rawvideo; audio is muxed from the source clip in the
    same ffmpeg invocation.

    Same write() interface as cv2.VideoWriter so the export loop is shared.
    """

    def __init__(
        self,
        output_path: Path,
        frame_size: Tuple[int, int],   # (width, height)
        fps: float,
        codec: str,
        crf: int,
        preset: str,
        audio_source: Optional[Path] = None,
        audio_start: float = 0.0,
        audio_duration: Optional[float] = None,
    ):
        self.output_path = Path(output_path)
        self._failed = False

        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            ffmpeg = "ffmpeg"

        width, height = frame_size
        cmd = [
            ffmpeg, "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "pipe:0",
        ]
        if audio_source is not None:
            cmd += [
                "-ss", str(audio_start),
                "-t", str(audio_duration if audio_duration is not None else 0),
                "-i", str(audio_source),
                "-map", "0:v", "-map", "1:a",
                "-c:a", "aac", "-ar", "16000", "-ac", "1",
                # Video (pipe) is the timing master; stop at the shorter stream
                # so a rounding mismatch never produces trailing frozen audio.
                "-shortest",
            ]
        else:
            cmd += ["-an"]
        cmd += [
            "-c:v", codec,
            "-preset", preset,
            "-crf", str(crf),
            *_thread_args(),
            "-pix_fmt", "yuv420p",
            "-r", str(fps),
            str(self.output_path),
        ]

        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame_bgr: np.ndarray) -> None:
        if self._failed:
            return
        try:
            self._process.stdin.write(np.ascontiguousarray(frame_bgr).tobytes())
        except (BrokenPipeError, OSError):
            self._failed = True

    def release(self) -> bool:
        """Finish the encode. Returns True when the output file is valid."""
        # Signal end-of-stream; a broken pipe here means ffmpeg already died.
        if self._process.stdin and not self._process.stdin.closed:
            try:
                self._process.stdin.close()
            except (BrokenPipeError, OSError):
                self._failed = True

        # Drain stderr until EOF (= process exit) — avoids pipe-buffer
        # deadlock — then reap the process.
        stderr = b""
        if self._process.stderr:
            try:
                stderr = self._process.stderr.read() or b""
            finally:
                self._process.stderr.close()
        try:
            returncode = self._process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
            logger.error(f"ffmpeg pipe encode timed out for {self.output_path}")
            self.output_path.unlink(missing_ok=True)
            return False

        if self._failed or returncode != 0:
            tail = stderr.decode(errors="replace")[-300:]
            logger.error(f"ffmpeg pipe encode failed for {self.output_path}: {tail}")
            self.output_path.unlink(missing_ok=True)
            return False
        return self.output_path.exists() and self.output_path.stat().st_size > 0


def reencode_with_audio(
    input_path: Path,
    output_path: Path,
    audio_source: Path,
    audio_start: float,
duration: float,
video_codec: str,
video_preset: str,
video_crf: int,
include_audio: bool,
) -> bool:
    """Re-encode video with ffmpeg, optionally adding audio."""
    try:
        try:
            import imageio_ffmpeg
            _ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            _ffmpeg = 'ffmpeg'

        if include_audio:
            # Mux audio from source.
            # -t duration on input 1 limits how much audio is read.
            cmd = [
                _ffmpeg, '-y',
                '-i', str(input_path),
                '-ss', str(audio_start),
                '-t', str(duration),
                '-i', str(audio_source),
                '-map', '0:v',
                '-map', '1:a',
                '-c:v', video_codec,
                '-preset', video_preset,
                '-crf', str(video_crf),
                *_thread_args(),
                '-c:a', 'aac',
                '-ar', '16000',
                '-ac', '1',
                str(output_path)
            ]
        else:
            # Video only
            cmd = [
                _ffmpeg, '-y',
                '-i', str(input_path),
                '-c:v', video_codec,
                '-preset', video_preset,
                '-crf', str(video_crf),
                *_thread_args(),
                '-an',
                str(output_path)
            ]
        
        subprocess.run(cmd, capture_output=True, check=True)
        input_path.unlink(missing_ok=True)  # remove temp file
        return output_path.exists() and output_path.stat().st_size > 0

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg encoding failed: {e.stderr.decode()}")
        input_path.unlink(missing_ok=True)
        return False
