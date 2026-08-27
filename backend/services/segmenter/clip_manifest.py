"""
Clip Manifest (clips.json)

The VideoClip record plus its persistence. clips.json lives next to the cut
clips and lets resume runs recover EVERYTHING the segmentation stage produced
— source-video timings, the words of every clip (with timestamps), boundary
types and diarization voice labels — without re-running VAD, Whisper or
pyannote.

Versions:
    v1 — VAD timings only (legacy runs; clips get words=None and fall back
         to per-clip transcription downstream)
    v2 — adds per-clip words + boundary types + audio_speaker_label
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from loguru import logger

from services.segmenter.sentence_segmenter import TimedWord


MANIFEST_FILENAME = "clips.json"
MANIFEST_VERSION = 2


@dataclass
class VideoClip:
    """One sentence/reply-level clip cut from the source video."""
    clip_id: str          # e.g. "md_001_clip_000"
    video_id: str
    clip_index: int
    start: float          # speech start — seconds in SOURCE video
    end: float            # speech end   — seconds in SOURCE video
    video_path: Path      # the cut .mp4 (video + audio)
    audio_path: Path      # the cut .wav (analysis audio)
    # The second of the source video where the FILES begin
    # (= start − pre-roll padding). Anchor for absolute↔clip conversions:
    # clip_time = absolute_time − content_start.
    content_start: float = 0.0

    # Sentence strategy only — from the full-video Whisper pass
    # (absolute source-video seconds). None → legacy VAD strategy.
    words: Optional[List[TimedWord]] = None
    boundary_start_type: str = ""   # punctuation | silence | speaker_change | word_gap | media
    boundary_end_type: str = ""
    audio_speaker_label: str = ""   # dominant diarization voice (SPEAKER_00…)

    @property
    def duration(self) -> float:
        return self.end - self.start


class ClipManifest:
    """Reads/writes clips.json for one video's clip directory."""

    @staticmethod
    def save(clip_dir: Path, video_id: str, clips: List[VideoClip]) -> None:
        payload = {
            "manifest_version": MANIFEST_VERSION,
            "video_id": video_id,
            "clips": [ClipManifest._clip_to_entry(c) for c in clips],
        }
        manifest_path = Path(clip_dir) / MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.debug(f"Wrote clips manifest: {manifest_path}")

    @staticmethod
    def load(clip_dir: Path, video_id: str) -> Optional[List[VideoClip]]:
        """Rebuild VideoClip objects from clips.json.

        Returns None when the manifest is missing/unreadable/mismatched —
        the caller decides on a fallback. Clips whose media files vanished
        are skipped.
        """
        manifest_path = Path(clip_dir) / MANIFEST_FILENAME
        if not manifest_path.exists():
            return None
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Cannot parse {manifest_path}: {e}")
            return None
        if payload.get("video_id") != video_id:
            logger.warning(
                f"{manifest_path} belongs to {payload.get('video_id')!r}, "
                f"expected {video_id!r}"
            )
            return None

        clips = []
        for entry in payload.get("clips", []):
            clip = ClipManifest._entry_to_clip(entry, video_id, Path(clip_dir))
            if clip is not None:
                clips.append(clip)
        logger.info(f"Loaded {len(clips)} existing clips from manifest for {video_id}")
        return clips

    # ------------------------------------------------------- serialization

    @staticmethod
    def _clip_to_entry(clip: VideoClip) -> dict:
        entry = {
            "clip_id": clip.clip_id,
            "clip_index": clip.clip_index,
            "start": clip.start,
            "end": clip.end,
            "content_start": clip.content_start,
        }
        if clip.words is not None:
            entry["boundary_start_type"] = clip.boundary_start_type
            entry["boundary_end_type"] = clip.boundary_end_type
            entry["audio_speaker_label"] = clip.audio_speaker_label
            entry["words"] = [
                {
                    "raw": word.raw_text,
                    "clean": word.clean_text,
                    "start": round(word.start, 3),
                    "end": round(word.end, 3),
                    "conf": word.confidence,
                    "spk": word.speaker,
                }
                for word in clip.words
            ]
        return entry

    @staticmethod
    def _entry_to_clip(entry: dict, video_id: str, clip_dir: Path) -> Optional[VideoClip]:
        clip_id = entry["clip_id"]
        video_path = clip_dir / f"{clip_id}.mp4"
        audio_path = clip_dir / f"{clip_id}.wav"
        if not video_path.exists() or not audio_path.exists():
            logger.debug(f"Manifest clip {clip_id} missing files on disk — skipping")
            return None

        words = None
        if "words" in entry:
            words = [
                TimedWord(
                    raw_text=w["raw"],
                    clean_text=w["clean"],
                    start=w["start"],
                    end=w["end"],
                    confidence=w.get("conf"),
                    speaker=w.get("spk"),
                )
                for w in entry["words"]
            ]

        return VideoClip(
            clip_id=clip_id,
            video_id=video_id,
            clip_index=entry["clip_index"],
            start=entry["start"],
            end=entry["end"],
            video_path=video_path,
            audio_path=audio_path,
            content_start=entry.get("content_start", entry["start"]),
            words=words,
            boundary_start_type=entry.get("boundary_start_type", ""),
            boundary_end_type=entry.get("boundary_end_type", ""),
            audio_speaker_label=entry.get("audio_speaker_label", ""),
        )
