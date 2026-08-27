"""
Clip processor — the per-clip cascade, one clip at a time.

For each clip: face tracking → visibility gate → transcript (from the
manifest words; per-clip Whisper only on the legacy strategy) → active
speaker selection → ASD gate → SyncNet → export (crops+audio+text) →
quality tier + ArcFace identity evidence.

Every failure path returns a ClipResult with a reason (→ dropped_clips);
only environment errors (missing models) escape — the video loop aborts
loudly instead of failing every clip identically.
"""

import json
from typing import List, Optional

import cv2
import numpy as np
from loguru import logger

from orchestrator.checkpoint_store import CheckpointStore
from orchestrator.pipeline_config import PipelineConfig
from orchestrator.processing_results import ClipResult
from orchestrator.service_container import ServiceContainer
from services.face_tracker.video_processor import FaceTrack
from services.quality_indexer.quality_tiers import compute_quality_tier
from services.segmenter.clip_manifest import VideoClip
from services.speaker_detector.active_speaker import ASDResult
from services.segmenter.transcription import TranscribedSegment, Word


class ClipProcessor:
    """Runs the full cascade for one clip and reports a ClipResult."""

    def __init__(
        self,
        config: PipelineConfig,
        services: ServiceContainer,
        checkpoints: CheckpointStore,
    ):
        self.config = config
        self.services = services
        self.checkpoints = checkpoints

    def process(
        self,
        clip: VideoClip,
        video_id: str,
        segment_index: int,
    ) -> ClipResult:
        """
        Process a single VAD clip through the quality pipeline.

        Steps (ordered cheapest-first to exit early):
          a. Face detection + tracking on clip video
          b. Face visibility check — drop if ratio < threshold  (free)
          c. Whisper transcription — drop if empty/low-conf     (cheap)
          d. Pre-filter tracks by overlap + face size           (free)
          e. ASD on candidate tracks only                       (expensive)
          f. Pick best track
          g. SyncNet verification
          h. LRS2 export
        """

        # a. Face detection + tracking
        logger.info(f"    [{clip.clip_id}] face detection...")
        face_tracks = self.services.video_processor.process_video(
            clip.video_path,
            detection_interval=self.config.detection_interval,
        )

        if not face_tracks:
            return ClipResult(clip, dropped=True, drop_reason="no_face_track")

        # Get clip metadata
        cap = cv2.VideoCapture(str(clip.video_path))
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        finally:
            cap.release()

        # b. Face visibility check (zero cost — early exit before expensive steps)
        visibility = self._compute_face_visibility(face_tracks, total_frames)

        if visibility < self.config.face_visibility_threshold:
            return ClipResult(
                clip,
                dropped=True,
                drop_reason=f"face_visibility={visibility:.2f}",
                face_visibility_ratio=visibility,
            )

        # c. Transcription for this clip.
        # Sentence strategy: the words were produced by the single full-video
        # Whisper pass and travel with the clip (manifest v2) — no per-clip
        # transcription needed. Legacy VAD strategy: transcribe the clip now.
        if clip.words is not None:
            merged = self._transcription_from_manifest_words(clip)
        else:
            logger.info(f"    [{clip.clip_id}] transcribing...")
            try:
                transcriptions = self.services.transcriber.transcribe(clip.audio_path)
            except Exception as e:
                logger.warning(f"Whisper failed on {clip.clip_id}: {e}")
                return ClipResult(
                    clip,
                    dropped=True,
                    drop_reason="whisper_error",
                    face_visibility_ratio=visibility,
                )

            if not transcriptions:
                return ClipResult(
                    clip,
                    dropped=True,
                    drop_reason="no_transcription",
                    face_visibility_ratio=visibility,
                )

            # Merge all Whisper sub-segments into one for this clip
            merged = self._merge_transcriptions(transcriptions, clip)

        if not merged.words or not merged.text.strip():
            return ClipResult(
                clip,
                dropped=True,
                drop_reason="empty_transcription",
                face_visibility_ratio=visibility,
            )

        # NaN means WhisperX gave no per-word scores at all — refuse to process
        # rather than silently treating it as max confidence.
        if np.isnan(merged.confidence):
            return ClipResult(
                clip,
                dropped=True,
                drop_reason="unknown_confidence",
                face_visibility_ratio=visibility,
            )

        if merged.confidence < self.config.min_whisper_confidence:
            return ClipResult(
                clip,
                dropped=True,
                drop_reason=f"low_confidence={merged.confidence:.2f}",
                face_visibility_ratio=visibility,
            )

        # d–f. Pick the best ASD speaker track overlapping the transcription.
        selection = self.services.speaker_selector.select_active_speaker(
            video_path=clip.video_path,
            audio_path=clip.audio_path,
            face_tracks=face_tracks,
            speech_start=merged.start,
            speech_end=merged.end,
            fps=fps,
        )
        if selection is None:
            return ClipResult(
                clip, dropped=True,
                drop_reason="no_track_overlaps_speech",
                face_visibility_ratio=visibility,
            )
        best_track, best_asd = selection.track, selection.asd_result

        # ASD gate (clip_filters.min_asd_score): when even the WINNER barely
        # correlates with the audio, no visible face is speaking — voice-over,
        # B-roll, reaction shot. That's not low-quality lip data, it's not lip
        # data at all (text misaligned with lips by construction). Dropping
        # HERE also skips SyncNet + export + embedding on poison clips.
        if best_asd.mean_score < self.config.min_asd_score:
            return ClipResult(
                clip, dropped=True,
                drop_reason=f"low_asd_score={best_asd.mean_score:.1f}",
                face_visibility_ratio=visibility,
            )

        # Trim transcription to match the track's actual frame range so the
        # export only contains frames where the speaker's face is visible.
        track_start_sec = best_track.start_frame / fps
        track_end_sec = (best_track.end_frame + 1) / fps

        trim_start = max(merged.start, track_start_sec)
        trim_end = min(merged.end, track_end_sec)

        if trim_end - trim_start < self.config.min_track_speech_overlap:
            return ClipResult(
                clip, dropped=True,
                drop_reason=f"track_too_short={trim_end - trim_start:.2f}s",
                face_visibility_ratio=visibility,
            )

        # Keep only words within the trimmed range
        trimmed_words = [
            Word(
                text=w.text,
                start=max(0.0, w.start - trim_start),
                end=min(trim_end - trim_start, w.end - trim_start),
                confidence=w.confidence,
            )
            for w in merged.words
            if w.end > trim_start and w.start < trim_end
        ]

        if not trimmed_words:
            return ClipResult(
                clip, dropped=True,
                drop_reason="no_words_in_track_range",
                face_visibility_ratio=visibility,
            )

        trimmed_text = " ".join(w.text for w in trimmed_words)
        # Recompute mean confidence from the trimmed word set so that the
        # exported annotation reflects only kept words.
        valid_trim_confs = [w.confidence for w in trimmed_words if w.confidence is not None]
        trimmed_conf = float(np.mean(valid_trim_confs)) if valid_trim_confs else float("nan")
        merged = TranscribedSegment(
            start=trim_start,
            end=trim_end,
            text=trimmed_text,
            words=trimmed_words,
            language=merged.language,
            confidence=trimmed_conf,
        )

        # g–h. SyncNet verification (optional) + LRS2 export.
        return self._export(
            clip=clip,
            video_id=video_id,
            segment_index=segment_index,
            best_track=best_track,
            best_asd=best_asd,
            merged=merged,
            fps=fps,
            visibility=visibility,
        )

    def _export(
        self, clip, video_id, segment_index, best_track, best_asd, merged, fps, visibility,
    ) -> ClipResult:
        """SyncNet-verify (if enabled) and export one clip to LRS2 format.

        If both output files already exist on disk (interrupted run with no
        checkpoint) the segment is loaded instead of re-exported. Returns a
        ClipResult (dropped only when the export itself fails).
        """
        # g. SyncNet (optional — very slow, ~100s per clip)
        if self.config.syncnet_enabled:
            logger.info(f"    [{clip.clip_id}] SyncNet verification...")
            try:
                sync_result = self.services.syncnet.verify_sync(
                    video_path=clip.video_path,
                    audio_path=clip.audio_path,
                    track=best_track,
                    speech_start=merged.start,
                    speech_end=merged.end,
                    fps=fps,
                )
                sync_conf = sync_result.confidence
                sync_method = sync_result.method
            except RuntimeError:
                # Missing/broken weights without allow_fallback — configuration
                # problem, not a per-clip hiccup. Abort instead of writing 0.0
                # confidences for the whole run.
                raise
            except Exception as e:
                logger.warning(f"SyncNet failed on {clip.clip_id}: {e}")
                sync_conf = 0.0
                sync_method = "error"
        else:
            sync_conf = 0.0
            sync_method = "disabled"

        # h. LRS2 export
        logger.info(f"    [{clip.clip_id}] exporting LRS2 format...")
        # Output combines clip origin + global export index for clear ordering:
        # e.g. md_001_clip_042_00003.mp4
        segment_name = f"{clip.clip_id}_{segment_index:05d}"

        # Idempotency check: if both output files already exist (e.g. after an
        # interrupted run that had no checkpoint), load from disk rather than
        # re-exporting and avoid overwriting good data.
        expected_video = (
            self.config.processed_dir / video_id / "face_crop" / f"{segment_name}.mp4"
        )
        expected_anno = (
            self.config.processed_dir / video_id / "text" / f"{segment_name}.txt"
        )
        if expected_video.exists() and expected_anno.exists():
            logger.debug(
                f"Segment {segment_name} already on disk — loading instead of re-exporting"
            )
            seg = self.checkpoints.segment_from_disk(
                video_id,
                segment_name,
                expected_video,
                expected_anno,
            )
            if seg is not None:
                return ClipResult(
                    clip=clip,
                    dropped=False,
                    face_visibility_ratio=visibility,
                    exported_segment=seg,
                )

        word_asd_scores = self._distribute_asd_scores(best_asd, merged)

        exported_seg = self.services.exporter.export_segment(
            source_video=clip.video_path,
            video_id=video_id,
            segment_index=segment_index,
            clip_id=clip.clip_id,   # bare clip id; exporter appends _{index:05d}
            track=best_track,
            transcription=merged,
            asd_scores=word_asd_scores,
            syncnet_confidence=sync_conf,
            fps=fps,
            face_visibility_ratio=visibility,
            asd_method=best_asd.method if best_asd else "",
            syncnet_method=sync_method,
            source_time_offset=clip.content_start,
            audio_speaker_label=clip.audio_speaker_label,
            audio_source=clip.audio_path,
        )

        if exported_seg is not None and self.config.save_face_tracks:
            self._save_track_trajectory(video_id, exported_seg, best_track, merged, fps)

        identity_record = None
        if exported_seg is not None:
            # Quality tier from everything measured for this segment.
            exported_seg.quality_tier = compute_quality_tier(
                {
                    "whisper_conf": exported_seg.whisper_confidence,
                    "whisper_conf_min": exported_seg.whisper_conf_min,
                    "face_visibility_ratio": exported_seg.face_visibility_ratio,
                    "mouth_landmark_fail_rate": exported_seg.mouth_landmark_fail_rate,
                    "asd_method": exported_seg.asd_method,
                    "syncnet_method": exported_seg.syncnet_method,
                    "mouth_roi_method": exported_seg.mouth_roi_method,
                    "boundary_start_type": clip.boundary_start_type,
                    "boundary_end_type": clip.boundary_end_type,
                    "duration": exported_seg.duration,
                },
                self.config.quality_tiers,
            )

            # ArcFace identity evidence (CPU) from the face-crop frames the
            # exporter sampled while writing — no re-decode of the fresh mp4.
            # Consumed by the end-of-video speaker clustering.
            identity_frames = exported_seg.identity_frames
            exported_seg.identity_frames = None  # free RAM before checkpointing
            if self.config.speaker_identity_enabled and identity_frames:
                try:
                    record = self.services.speaker_identifier.embed_frames(
                        identity_frames
                    )
                    identity_record = record.to_json_dict() if record else None
                except Exception as e:
                    logger.warning(
                        f"Speaker identity sampling failed on {clip.clip_id}: {e}"
                    )

        return ClipResult(
            clip=clip,
            dropped=(exported_seg is None),
            drop_reason="export_failed" if exported_seg is None else None,
            face_visibility_ratio=visibility,
            exported_segment=exported_seg,
            identity=identity_record,
        )

    # helpers
    def _save_track_trajectory(self, video_id, exported_seg, track, merged, fps):
        """Optional (face_tracking.save_tracks): persist the winning track's
        per-frame bboxes for the exported range — enables re-exporting crops
        with different settings without re-running detection (~50 KB/segment)."""
        try:
            start_frame = int(merged.start * fps)
            end_frame = int(merged.end * fps)
            trajectory = {}
            for frame_idx in range(start_frame, end_frame):
                bbox = track.interpolate_bbox(frame_idx)
                if bbox is not None:
                    trajectory[frame_idx] = [bbox.x, bbox.y, bbox.width, bbox.height]

            tracks_dir = self.config.processed_dir / video_id / "tracks"
            tracks_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "segment_id": exported_seg.segment_id,
                "track_id": track.track_id,
                "fps": fps,
                "bboxes_by_frame": trajectory,
            }
            (tracks_dir / f"{exported_seg.segment_id}.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
        except Exception as e:
            logger.warning(f"Could not save track trajectory: {e}")

    def _compute_face_visibility(
        self,
        face_tracks: List[FaceTrack],
        total_frames: int,
    ) -> float:
        """Fraction of frames within the tracked window that have a face.

        The denominator is the span of the earliest-to-latest track frame,
        not the full clip length.  A clip with a long silent intro before the
        speaker appears should not be penalised for those pre-speech frames.
        Falls back to total_frames if no tracks have valid frame ranges.
        """
        if not face_tracks or total_frames <= 0:
            return 0.0
        covered: set = set()
        for track in face_tracks:
            covered.update(range(track.start_frame, track.end_frame + 1))
        if not covered:
            return 0.0
        window = max(covered) - min(covered) + 1
        return len(covered) / window

    def _transcription_from_manifest_words(self, clip: VideoClip) -> TranscribedSegment:
        """Build the clip's TranscribedSegment from manifest words (sentence
        strategy).

        Manifest word times are absolute source-video seconds; the exported
        clip file starts at clip.content_start, so times shift into
        clip-file coordinates here (t=0 at file start).
        """
        words = [
            Word(
                text=w.clean_text,
                start=max(0.0, w.start - clip.content_start),
                end=max(0.0, w.end - clip.content_start),
                confidence=w.confidence,
            )
            for w in (clip.words or [])
            if w.clean_text
        ]

        valid_confs = [w.confidence for w in words if w.confidence is not None]
        confidence = float(np.mean(valid_confs)) if valid_confs else float("nan")

        return TranscribedSegment(
            start=words[0].start if words else 0.0,
            end=words[-1].end if words else clip.duration,
            text=" ".join(w.text for w in words),
            words=words,
            language=self.config.language,
            confidence=confidence,
        )

    def _merge_transcriptions(
        self,
        transcriptions: List[TranscribedSegment],
        clip: VideoClip,
    ) -> TranscribedSegment:
        """
        Merge multiple Whisper sub-segments into one TranscribedSegment
        that spans the whole clip.

        Word timings remain relative to clip start (0.0).
        """
        all_words: List[Word] = []
        for seg in transcriptions:
            all_words.extend(seg.words)

        if not all_words:
            return TranscribedSegment(
                start=0.0,
                end=clip.duration,
                text="",
                words=[],
                language=self.config.language,
                confidence=0.0,
            )

        text = " ".join(w.text for w in all_words)
        # Mean over words that actually have a confidence score; NaN if all
        # are None — caller treats that as `unknown_confidence` and drops.
        valid_confs = [w.confidence for w in all_words if w.confidence is not None]
        confidence = float(np.mean(valid_confs)) if valid_confs else float("nan")
        start = all_words[0].start
        end = all_words[-1].end

        return TranscribedSegment(
            start=start,
            end=end,
            text=text,
            words=all_words,
            language=self.config.language,
            confidence=confidence,
        )

    def _distribute_asd_scores(
        self,
        asd_result: Optional[ASDResult],
        trans: TranscribedSegment,
    ) -> List[float]:
        """Distribute ASD scores across words by index ratio."""
        n_words = len(trans.words)
        if not asd_result or not asd_result.scores:
            return [0.0] * n_words
        scores = asd_result.scores
        return [
            scores[min(int(i * len(scores) / n_words), len(scores) - 1)]
            for i in range(n_words)
        ]
