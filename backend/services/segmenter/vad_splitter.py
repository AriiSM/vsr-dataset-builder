"""
VAD Splitter — the segmenter service's orchestrator

Composes the segmenter's building blocks:
    SileroVAD    (voice_activity.py)  — where speech and pauses are
    ClipCutter   (clip_cutter.py)     — one-invocation ffmpeg cutting
    ClipManifest (clip_manifest.py)   — clips.json persistence

Two strategies (config: segmentation.strategy):
    "sentence" — the pipeline computes SentenceWindows (punctuation ∪ pauses
                 ∪ speaker changes) and calls cut_sentence_windows();
    "vad"      — legacy silence-only splitting (split_video()), kept for
                 comparison runs.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
from loguru import logger

from services.segmenter.clip_cutter import ClipCutter
from services.segmenter.clip_manifest import ClipManifest, VideoClip
from services.segmenter.sentence_segmenter import SentenceWindow
from services.segmenter.voice_activity import SileroVAD

__all__ = ["VADSplitter", "VideoClip"]


class VADSplitter:
    """Segmentation orchestrator for one video at a time.

    All parameters come from config.yaml — nothing tunable is hardcoded.
    """

    def __init__(
        self,
        vad_threshold: float,
        min_speech_duration_ms: int,
        min_silence_duration_ms: int,
        window_size_samples: int,
        speech_pad_ms: int,
        split_threshold: float,
        min_clip_duration: float,
        max_clip_duration: float,
        sample_rate: int,
        seek_start_padding: float,
        clip_crf: int = 16,
        clip_preset: Optional[str] = "veryfast",
    ):
        self.split_threshold = split_threshold
        self.min_clip_duration = min_clip_duration
        self.max_clip_duration = max_clip_duration

        self._vad = SileroVAD(
            threshold=vad_threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            window_size_samples=window_size_samples,
            speech_pad_ms=speech_pad_ms,
            sample_rate=sample_rate,
        )
        self._cutter = ClipCutter(
            clip_crf=clip_crf,
            clip_preset=clip_preset,
            seek_start_padding=seek_start_padding,
            audio_sample_rate=sample_rate,
        )

    # ------------------------------------------------------ sentence strategy

    def extract_full_audio(self, video_path: Path, video_id: str, temp_dir: Path) -> Path:
        """Whole-video audio → temp WAV (16 kHz mono).

        The same WAV feeds VAD, WhisperX and diarization. The caller deletes
        it once segmentation is done.
        """
        temp_dir = Path(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        audio_path = temp_dir / f"{video_id}_full_audio.wav"
        self._vad.extract_audio_from_video(video_path, audio_path)
        return audio_path

    def detect_speech_regions(self, audio_path: Path) -> List[Tuple[float, float]]:
        """Raw VAD speech regions (absolute seconds) — pauses live between them."""
        segments = self._vad.detect_speech(audio_path)
        return [(s.start, s.end) for s in segments]

    def cut_sentence_windows(
        self,
        video_path: Path,
        video_id: str,
        windows: List[SentenceWindow],
        clips_dir: Path,
    ) -> List[VideoClip]:
        """Cut one clip per sentence window and persist the v2 manifest.

        Every clip carries its own words (absolute timestamps), boundary
        types and diarization voice label — nothing downstream ever needs to
        re-transcribe or re-diarize.
        """
        clip_dir = Path(clips_dir) / video_id
        clip_dir.mkdir(parents=True, exist_ok=True)

        clips: List[VideoClip] = []
        for window_index, window in enumerate(windows):
            clip_id = f"{video_id}_clip_{window_index:03d}"
            output_video = clip_dir / f"{clip_id}.mp4"
            output_audio = clip_dir / f"{clip_id}.wav"

            content_start = self._cutter.cut(
                video_path, window.start, window.end, output_video, output_audio,
            )
            if content_start is None:
                logger.warning(
                    f"Failed to cut clip {clip_id} "
                    f"[{window.start:.2f}-{window.end:.2f}s]"
                )
                continue

            clips.append(VideoClip(
                clip_id=clip_id,
                video_id=video_id,
                clip_index=window_index,
                start=window.start,
                end=window.end,
                video_path=output_video,
                audio_path=output_audio,
                content_start=content_start,
                words=list(window.words),
                boundary_start_type=window.boundary_start_type,
                boundary_end_type=window.boundary_end_type,
                audio_speaker_label=window.audio_speaker_label,
            ))

        ClipManifest.save(clip_dir, video_id, clips)
        logger.info(
            f"Cut {video_id} into {len(clips)} sentence clips "
            f"({len(windows) - len(clips)} failed)"
        )
        return clips

    # ------------------------------------------------------------- resume

    def load_existing_clips(self, video_id: str, clip_dir: Path) -> List[VideoClip]:
        """Rebuild VideoClips from disk without re-running any model.

        Manifest v2 restores words + voice labels exactly; a legacy directory
        without a manifest falls back to file-duration timing WITH a loud
        warning (source-video timestamps unrecoverable there).
        """
        clip_dir = Path(clip_dir)
        clips = ClipManifest.load(clip_dir, video_id)
        if clips is not None:
            return clips

        logger.warning(
            f"No clips manifest for {video_id} in {clip_dir} — falling back "
            f"to file-duration timing. start/end times will be relative to "
            f"each clip, NOT to the source video."
        )
        return self._rebuild_clips_from_files(video_id, clip_dir)

    def _rebuild_clips_from_files(self, video_id: str, clip_dir: Path) -> List[VideoClip]:
        """Last-resort recovery: only the media files exist (pre-manifest runs)."""
        clips: List[VideoClip] = []
        for video_file in sorted(clip_dir.glob(f"{video_id}_clip_*.mp4")):
            audio_file = video_file.with_suffix(".wav")
            if not audio_file.exists():
                logger.warning(f"Missing WAV for {video_file.name} — skipping")
                continue
            try:
                clip_index = int(video_file.stem.split("_clip_")[-1])
            except ValueError:
                logger.warning(f"Cannot parse clip index from {video_file.name} — skipping")
                continue

            duration = self._media_duration(video_file)
            if duration < self.min_clip_duration:
                continue

            clips.append(VideoClip(
                clip_id=f"{video_id}_clip_{clip_index:03d}",
                video_id=video_id,
                clip_index=clip_index,
                start=0.0,
                end=duration,
                video_path=video_file,
                audio_path=audio_file,
                content_start=0.0,
            ))
        return clips

    @staticmethod
    def _media_duration(video_file: Path) -> float:
        capture = cv2.VideoCapture(str(video_file))
        try:
            fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        return frame_count / fps if fps > 0 else 0.0

    # ------------------------------------------------------- legacy strategy

    def split_video(
        self,
        video_path: Path,
        video_id: str,
        clips_dir: Path,
        temp_dir: Path,
    ) -> List[VideoClip]:
        """LEGACY (strategy "vad"): silence-only splitting.

        Boundaries are VAD gaps >= split_threshold; groups longer than
        max_clip_duration are sub-split at internal gaps, and a continuous
        no-gap run is hard-cut at the limit (the exact behaviour the
        sentence strategy replaces). Kept for comparison runs.
        """
        clip_dir = Path(clips_dir) / video_id
        clip_dir.mkdir(parents=True, exist_ok=True)

        full_audio = self.extract_full_audio(video_path, video_id, temp_dir)
        try:
            windows = self._merge_vad_regions_into_windows(full_audio)
        finally:
            full_audio.unlink(missing_ok=True)
        logger.info(f"VAD: {video_id} produced {len(windows)} clips to cut")

        clips: List[VideoClip] = []
        for window_index, (start, end) in enumerate(windows):
            clip_id = f"{video_id}_clip_{window_index:03d}"
            output_video = clip_dir / f"{clip_id}.mp4"
            output_audio = clip_dir / f"{clip_id}.wav"

            content_start = self._cutter.cut(
                video_path, start, end, output_video, output_audio,
            )
            if content_start is None:
                logger.warning(f"Failed to cut clip {clip_id} [{start:.2f}-{end:.2f}s]")
                continue

            clips.append(VideoClip(
                clip_id=clip_id,
                video_id=video_id,
                clip_index=window_index,
                start=start,
                end=end,
                video_path=output_video,
                audio_path=output_audio,
                content_start=content_start,
            ))

        ClipManifest.save(clip_dir, video_id, clips)
        logger.info(f"Split {video_id} into {len(clips)} clips")
        return clips

    def _merge_vad_regions_into_windows(self, audio_path: Path) -> List[Tuple[float, float]]:
        """Legacy merging: VAD regions → clip windows bounded by duration."""
        import math

        segments = self._vad.detect_speech(audio_path)
        if not segments:
            return []

        # Group regions separated by less than split_threshold
        groups: List[List] = [[segments[0]]]
        for segment in segments[1:]:
            if segment.start - groups[-1][-1].end < self.split_threshold:
                groups[-1].append(segment)
            else:
                groups.append([segment])

        # Bound each group by max_clip_duration (sub-split at internal gaps,
        # then hard-cut continuous runs — legacy behaviour)
        windows: List[Tuple[float, float]] = []
        for group in groups:
            if group[-1].end - group[0].start <= self.max_clip_duration:
                windows.append((group[0].start, group[-1].end))
                continue
            sub_start, sub_end = group[0].start, group[0].end
            for segment in group[1:]:
                if segment.end - sub_start <= self.max_clip_duration:
                    sub_end = segment.end
                else:
                    windows.append((sub_start, sub_end))
                    sub_start, sub_end = segment.start, segment.end
            windows.append((sub_start, sub_end))

        bounded: List[Tuple[float, float]] = []
        for start, end in windows:
            if end - start <= self.max_clip_duration:
                bounded.append((start, end))
                continue
            chunk_count = math.ceil((end - start) / self.max_clip_duration)
            chunk_length = (end - start) / chunk_count
            for i in range(chunk_count):
                chunk_start = start + i * chunk_length
                chunk_end = start + (i + 1) * chunk_length if i < chunk_count - 1 else end
                bounded.append((chunk_start, chunk_end))

        kept = [(s, e) for s, e in bounded if e - s >= self.min_clip_duration]
        logger.info(
            f"VAD merge: {len(segments)} regions → {len(kept)} clips "
            f"({len(bounded) - len(kept)} too short dropped)"
        )
        return kept
