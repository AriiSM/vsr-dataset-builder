"""
VSR Pipeline — the top-level orchestrator.

Thin by design: composes the collaborators and owns only the PER-VIDEO
story — download → sentence segmentation → the per-clip cascade (delegated
to ClipProcessor) → speaker identity clustering → catalog updates → cleanup.

Collaborators (one file each, one responsibility each):
    PipelineConfig    (pipeline_config.py)    — every knob, from config.yaml
    ServiceContainer  (service_container.py)  — lazy access to all services
    CheckpointStore   (checkpoint_store.py)   — resume state + recovery
    CatalogWriter     (catalog_writer.py)     — videos_master + segments_index
    ClipProcessor     (clip_processor.py)     — the per-clip cascade
"""

import time
import traceback
from pathlib import Path
from typing import List, Optional

from loguru import logger

from orchestrator.catalog_writer import CatalogWriter
from orchestrator.checkpoint_store import CheckpointStore
from orchestrator.clip_processor import ClipProcessor
from orchestrator.pipeline_config import PipelineConfig
from orchestrator.processing_results import (
    ClipResult,
    PipelineCancelled,
    ProcessingResult,
)
from orchestrator.service_container import ServiceContainer
from services.mouth_exporter.segment_record import ExportedSegment
from services.quality_indexer.av_consensus import compute_av_consensus
from services.quality_indexer.identity_records import SegmentIdentityRecord
from services.segmenter.clip_manifest import VideoClip
from services.segmenter.sentence_segmenter import (
    SegmentationSettings,
    SentenceSegmenter,
)
from vsr_shared.excel_schema import ProcessingStatus
from vsr_shared.speakers_registry import (
    ensure_speaker_exists,
    recompute_aggregates,
    upsert_speaker,
)

__all__ = ["VSRPipeline", "PipelineConfig"]


class VSRPipeline:
    """One video at a time: download → segment → per-clip cascade → identity."""

    def __init__(self, config: PipelineConfig):
        self.config = config

        for directory in [
            config.raw_dir, config.clips_dir,
            config.processed_dir, config.metadata_dir, config.temp_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        self.services = ServiceContainer(config)
        self.checkpoints = CheckpointStore(config)
        self.catalog = CatalogWriter(config, self.checkpoints)
        self.clip_processor = ClipProcessor(config, self.services, self.checkpoints)

        # Job-queue hooks (both optional — CLI runs leave them unset):
        #   on_progress(dict)      called at stage changes + after every clip
        #   should_cancel() -> bool  checked between stages and between clips;
        #     True → PipelineCancelled (video stays resumable via checkpoint)
        self.on_progress = None
        self.should_cancel = None

    @classmethod
    def from_config(cls, config_path) -> "VSRPipeline":
        return cls(PipelineConfig.from_yaml(Path(config_path)))

    def _report_progress(self, **payload) -> None:
        if self.on_progress is not None:
            try:
                self.on_progress(payload)
            except Exception as e:
                logger.debug(f"progress callback failed: {e}")

    def _check_cancel(self, video_id: str) -> None:
        if self.should_cancel is not None and self.should_cancel():
            logger.info(f"Cancel requested — stopping cleanly at {video_id}")
            raise PipelineCancelled(video_id)

    # lazy properties
    # main entry point
    def process_video(
        self,
        video_id: str,
        youtube_url: str,
        skip_download: bool = False,
        skip_if_exists: bool = True,
        verify_cc: bool = True,
    ) -> ProcessingResult:
        """Process one video end-to-end."""
        start_time = time.time()

        try:
            # Mark as in-progress so interrupted runs are identifiable
            self.catalog.set_video_status(video_id, ProcessingStatus.PROCESSING)

            # 1. Download
            self._check_cancel(video_id)
            self._report_progress(video_id=video_id, stage="download")
            video_path = self.config.raw_dir / f"{video_id}.mp4"
            if not video_path.exists() or not skip_download:
                logger.info(f"Step 1: Downloading video: {video_id}")
                video_path = self.services.downloader.download(
                    youtube_url, video_id, verify_cc=verify_cc
                )
                if video_path is None:
                    failed = ProcessingResult(
                        video_id=video_id,
                        status=ProcessingStatus.FAILED,
                        error_message="Download failed",
                        processing_time=time.time() - start_time,
                    )
                    self.catalog.update_for_result(failed)
                    return failed
            else:
                logger.info(f"Step 1: Video already exists: {video_path}")

            # 2. VAD split
            self._check_cancel(video_id)
            self._report_progress(video_id=video_id, stage="segmentation")
            # Re-use existing clips if they are still on disk
            clip_dir = self.config.clips_dir / video_id
            existing_clip_files = sorted(clip_dir.glob("*.mp4")) if clip_dir.exists() else []

            if existing_clip_files:
                logger.info(
                    f"Step 2: {len(existing_clip_files)} existing clips found in {clip_dir}, "
                    "skipping segmentation"
                )
                # Manifest v2 restores word timestamps too (sentence strategy);
                # v1 clips fall back to per-clip transcription downstream.
                clips = self.services.splitter.load_existing_clips(video_id, clip_dir)
            elif self.config.segmentation_strategy == "sentence":
                clips = self._segment_video_into_sentences(video_path, video_id)
            else:
                logger.info("Step 2: Splitting video by VAD (legacy strategy)...")
                clips = self.services.splitter.split_video(
                    video_path,
                    video_id,
                    clips_dir=self.config.clips_dir,
                    temp_dir=self.config.temp_dir,
                )

            if not clips:
                failed = ProcessingResult(
                    video_id=video_id,
                    status=ProcessingStatus.FAILED,
                    error_message="No speech clips after VAD splitting",
                    processing_time=time.time() - start_time,
                )
                self.catalog.update_for_result(failed)
                return failed

            logger.info(f"  {len(clips)} clips to process")

            # 3. Per-clip processing
            logger.info("Step 3: Processing clips...")

            # Read checkpoint — tells us which clips were already processed
            checkpoint = self.checkpoints.read(video_id)
            processed_clip_ids: set = set(checkpoint.get("processed_clips", {}).keys())
            segment_index: int = checkpoint.get("segment_count", 0)

            # Recover exported segments that are already on disk
            exported_segments: List[ExportedSegment] = \
                self.checkpoints.recover_segments(video_id, checkpoint)

            if checkpoint:
                logger.info(
                    f"  Resuming: {len(exported_segments)} segments already done, "
                    f"{len(processed_clip_ids)} clips already processed"
                )
            else:
                logger.info(f"  Starting fresh ({len(clips)} clips)")

            for clip in clips:
                # Skip clips already processed in a previous (interrupted) run
                if clip.clip_id in processed_clip_ids:
                    logger.debug(f"  Skipping already-processed clip: {clip.clip_id}")
                    continue

                self._check_cancel(video_id)
                try:
                    clip_result = self.clip_processor.process(clip, video_id, segment_index)
                except RuntimeError:
                    # Environment/config problems (missing models, tokens) —
                    # every clip would fail identically; abort loudly instead.
                    raise
                except Exception as e:
                    logger.error(f"Clip {clip.clip_id} failed: {e}")
                    clip_result = ClipResult(
                        clip, dropped=True,
                        drop_reason=f"processing_error: {type(e).__name__}: {str(e)[:150]}",
                    )

                # Record outcome in checkpoint immediately
                seg = clip_result.exported_segment
                checkpoint.setdefault("processed_clips", {})[clip.clip_id] = {
                    "result": "dropped" if clip_result.dropped else "exported",
                    "reason": clip_result.drop_reason,
                    "segment_id": seg.segment_id if seg else None,
                    "start_time": seg.start_time if seg else None,
                    "end_time": seg.end_time if seg else None,
                    "duration": seg.duration if seg else None,
                    "asd_score": seg.asd_score if seg else None,
                    "syncnet_confidence": seg.syncnet_confidence if seg else None,
                    "whisper_confidence": seg.whisper_confidence if seg else None,
                    # Quality metadata (Phase 0) — read back with .get() so
                    # checkpoints written before this change still resume fine.
                    "face_visibility_ratio": seg.face_visibility_ratio if seg else None,
                    "whisper_conf_min": seg.whisper_conf_min if seg else None,
                    "whisper_conf_p25": seg.whisper_conf_p25 if seg else None,
                    "asd_method": seg.asd_method if seg else None,
                    "syncnet_method": seg.syncnet_method if seg else None,
                    "face_bbox": seg.face_bbox if seg else None,
                    "mouth_landmark_fail_rate": seg.mouth_landmark_fail_rate if seg else None,
                    "mouth_roi_method": seg.mouth_roi_method if seg else None,
                    "head_pose_avg": seg.head_pose_avg if seg else None,
                    "quality_tier": seg.quality_tier if seg else None,
                    "audio_speaker_label": seg.audio_speaker_label if seg else None,
                    "identity": clip_result.identity,
                }
                checkpoint["segment_count"] = segment_index + (
                    1 if clip_result.exported_segment else 0
                )
                checkpoint["video_id"] = video_id
                self.checkpoints.write(video_id, checkpoint)

                clip_num = len(checkpoint.get("processed_clips", {}))
                total_clips = len(clips)
                self._report_progress(
                    video_id=video_id, stage="clips",
                    clip_num=clip_num, total_clips=total_clips,
                    segments_exported=segment_index,
                )

                if clip_result.dropped:
                    logger.info(
                        f"  CLIP {clip_num}/{total_clips} dropped"
                        f" ({clip_result.drop_reason}): {clip.clip_id}"
                    )
                    # Persist the rejection WITH its reason — survives cleanup
                    self.catalog.record_dropped_clip(
                        video_id, clip, clip_result.drop_reason or "",
                        face_visibility=clip_result.face_visibility_ratio,
                    )
                    continue

                if clip_result.exported_segment:
                    exported_segments.append(clip_result.exported_segment)
                    segment_index += 1
                    logger.info(
                        f"  CLIP {clip_num}/{total_clips} exported:"
                        f" {clip_result.exported_segment.segment_id}"
                    )
                    # Transaction per clip into dataset.db (+ CSV mirror) so
                    # the Review tab sees the segment immediately
                    self.catalog.append_segment(clip_result.exported_segment)
                    if clip_result.identity and clip_result.identity.get("embedding"):
                        self.catalog.store_segment_embedding(
                            clip_result.exported_segment.segment_id,
                            clip_result.identity["embedding"],
                        )

            # 4. Cleanup
            if self.config.cleanup_clips:
                import shutil
                if clip_dir.exists():
                    shutil.rmtree(clip_dir)
                    logger.debug(f"Cleaned up clips dir: {clip_dir}")
            else:
                # Remove checkpoint — processing completed successfully
                cp = self.checkpoints.path_for(video_id)
                if cp.exists():
                    cp.unlink()
                    logger.debug("Checkpoint removed (processing complete)")

            total_duration = sum(s.duration for s in exported_segments)
            logger.info(
                f"Exported {len(exported_segments)} segments "
                f"({total_duration:.1f}s total)"
            )

            # First-pass speaker identity: cluster ArcFace evidence from all
            # exported segments so a recurring speaker gets ONE id, then write
            # per-cluster demographics (gender + numeric age) to the registry.
            # MUST run before _update_excel — that's what rewrites
            # the segments table with the final speaker_id values.
            if self.config.speaker_identity_enabled:
                try:
                    self._assign_speaker_identities(
                        video_id, exported_segments, checkpoint
                    )
                except Exception as e:
                    logger.warning(
                        f"Speaker identity assignment failed for {video_id}: {e}"
                    )

            final_result = ProcessingResult(
                video_id=video_id,
                status=ProcessingStatus.COMPLETED,
                segments=exported_segments,
                total_duration=total_duration,
                processing_time=time.time() - start_time,
            )
            self.catalog.update_for_result(final_result)

            # Registry upkeep — demographics were already written per cluster
            # by _assign_speaker_identities.
            try:
                seen_speakers = {
                    seg.speaker_id or f"{seg.video_id}_spk0"
                    for seg in exported_segments
                }
                for sid in seen_speakers:
                    ensure_speaker_exists(self.config.metadata_dir, sid)
                if seen_speakers:
                    recompute_aggregates(self.config.metadata_dir)
            except Exception as e:
                logger.warning(f"Speakers registry update failed for {video_id}: {e}")

            return final_result

        except PipelineCancelled:
            # Deliberate stop: the video keeps status 'processing' + its
            # checkpoint — resume-batch picks it up later. Propagate so the
            # worker marks the JOB cancelled (not the video failed).
            raise
        except Exception as e:
            logger.error(f"Pipeline failed for {video_id}: {e}")
            logger.debug(traceback.format_exc())
            failed = ProcessingResult(
                video_id=video_id,
                status=ProcessingStatus.FAILED,
                error_message=str(e),
                processing_time=time.time() - start_time,
            )
            self.catalog.update_for_result(failed)
            return failed

    def _assign_speaker_identities(
        self,
        video_id: str,
        exported_segments: List[ExportedSegment],
        checkpoint: dict,
    ) -> None:
        """Cluster per-segment ArcFace evidence into speaker identities and
        write per-cluster demographics to the speakers registry.

        Evidence comes from the checkpoint, so segments exported by an earlier
        interrupted run participate too (their face crops were sampled when
        they were first exported).
        """
        records: Dict[str, SegmentIdentityRecord] = {}
        for info in checkpoint.get("processed_clips", {}).values():
            if info.get("result") != "exported":
                continue
            segment_id = info.get("segment_id")
            record = SegmentIdentityRecord.from_json_dict(info.get("identity") or {})
            if segment_id and record is not None:
                records[segment_id] = record

        if not records:
            logger.warning(
                f"No identity evidence for {video_id} — speaker ids stay default"
            )
            return

        mapping, profiles = self.services.speaker_identifier.assign_speakers(video_id, records)

        for segment in exported_segments:
            if segment.segment_id in mapping:
                segment.speaker_id = mapping[segment.segment_id]

        accent_region = self._video_region(video_id)
        for speaker_id, profile in profiles.items():
            fields = {
                "gender": profile.gender,
                "gender_confidence": profile.gender_confidence,
                "age_group": profile.age_group,
                "age_std": profile.age_std,
                "identity_match": profile.identity_match,
            }
            if not math.isnan(profile.age_estimate):
                fields["age_estimate"] = profile.age_estimate
            if accent_region:
                fields["accent_region"] = accent_region
            upsert_speaker(self.config.metadata_dir, speaker_id, fields)

        self._flag_av_mismatches(video_id, exported_segments)

    def _flag_av_mismatches(
        self, video_id: str, exported_segments: List[ExportedSegment]
    ) -> None:
        """Voice↔face consensus: flag segments whose diarization voice points
        at a different face than the voice's per-video majority.

        Nothing is deleted — the flag caps the quality tier at B and surfaces
        the segment for review (voice-over / B-roll / wrong-track cases).
        """
        consensus = compute_av_consensus([
            (seg.segment_id, seg.audio_speaker_label, seg.speaker_id)
            for seg in exported_segments
        ])
        if not consensus.num_judged:
            return

        for segment in exported_segments:
            verdict = consensus.mismatch_by_segment.get(segment.segment_id)
            if verdict is None:
                continue
            segment.av_speaker_mismatch = verdict
            if verdict and segment.quality_tier == "A":
                segment.quality_tier = "B"

        if consensus.num_mismatched:
            logger.warning(
                f"AV consensus [{video_id}]: {consensus.num_mismatched}/"
                f"{consensus.num_judged} segments contradict their voice's "
                f"majority face — flagged av_speaker_mismatch, tier capped at B"
            )
        else:
            logger.info(
                f"AV consensus [{video_id}]: {consensus.num_judged} segments "
                f"consistent (voices → faces: {consensus.voice_to_face})"
            )

    def _video_region(self, video_id: str) -> str:
        """The video's region from the videos table (accent default)."""
        try:
            return self.catalog.db.videos.region(video_id)
        except Exception as e:
            logger.debug(f"Could not read region for {video_id}: {e}")
            return ""

    def _segment_video_into_sentences(
        self, video_path: Path, video_id: str
    ) -> List[VideoClip]:
        """Sentence-strategy Step 2: transcribe-then-cut.

        1. Extract full audio once.
        2. Silero VAD → raw speech regions (pauses + hallucination filter).
        3. WhisperX full-video pass → every word with timestamps + punctuation.
        4. Sentence segmenter → cutting windows (never mid-word; over-long
           sentences split at their largest pause, not at a blind limit).
        5. ffmpeg cuts the clips; each clip carries its words (manifest v2).
        6. Whisper is unloaded so RetinaFace/TalkNet get the GPU.
        """
        logger.info("Step 2: Sentence segmentation (VAD + full-video Whisper)...")

        audio_path = self.services.splitter.extract_full_audio(
            video_path, video_id, self.config.temp_dir
        )
        try:
            speech_regions = self.services.splitter.detect_speech_regions(audio_path)
            words = self.services.transcriber.transcribe_full(audio_path)

            if self.config.diarization_enabled and words:
                # WHO speaks WHEN — labels enable speaker-turn boundaries.
                # A diarization failure degrades to punctuation+pauses only,
                # loudly, instead of killing the whole video.
                try:
                    self.services.diarizer.assign_speaker_labels(words, audio_path)
                except Exception as e:
                    logger.warning(
                        f"Diarization failed for {video_id} — falling back to "
                        f"punctuation+pause boundaries only: {e}"
                    )
        finally:
            audio_path.unlink(missing_ok=True)

        settings = SegmentationSettings(
            sentence_end_chars=self.config.sentence_end_chars,
            split_silence_threshold=self.config.split_threshold,
            target_min_duration=self.config.target_min_duration,
            target_max_duration=self.config.target_max_duration,
            hard_min_duration=self.config.min_clip_duration,
            hard_max_duration=self.config.segmentation_max_clip_duration,
            merge_gap_max=self.config.merge_gap_max,
            boundary_pad=self.config.boundary_pad,
            vad_margin=self.config.vad_margin,
        )
        windows = SentenceSegmenter(settings).build_windows(words, speech_regions)
        logger.info(
            f"  {len(words)} words / {len(speech_regions)} speech regions "
            f"→ {len(windows)} sentence windows"
        )

        clips = self.services.splitter.cut_sentence_windows(
            video_path, video_id, windows, self.config.clips_dir
        )

        if self.config.whisper_unload_after_segmentation:
            self.services.release_segmentation_models()

        return clips

    # resume
    def resume_video(self, video_id: str, youtube_url: str) -> "ProcessingResult":
        """
        Resume a partially-processed video.

        Reads the per-clip checkpoint (if present) to skip clips that were
        already processed in the interrupted run, then continues from where
        the pipeline stopped.  If no checkpoint exists, falls back to a full
        re-run that skips the LRS2 export for segments already on disk.
        """
        logger.info(f"Resuming video: {video_id}")
        checkpoint = self.checkpoints.read(video_id)

        if checkpoint:
            done = len(checkpoint.get("processed_clips", {}))
            segs = checkpoint.get("segment_count", 0)
            logger.info(f"  Checkpoint found: {done} clips done, {segs} segments exported")
        else:
            # No checkpoint — scan disk to report what's already there
            proc_dir = self.config.processed_dir / video_id
            existing = sorted(proc_dir.glob("*.mp4")) if proc_dir.exists() else []
            logger.info(
                f"  No checkpoint found. {len(existing)} segment file(s) already on disk "
                "— will skip re-exporting those."
            )

        # process_video now handles checkpoint reading and the idempotency
        # check automatically; skip_download avoids re-downloading the raw video.
        return self.process_video(
            video_id=video_id,
            youtube_url=youtube_url,
            skip_download=True,
            verify_cc=False,
        )

    def process_batch_resume(
        self,
        excel_path: Path,
        limit: Optional[int] = None,
        video_ids: Optional[List[str]] = None,
    ) -> "List[ProcessingResult]":
        """
        Resume all videos that were interrupted (status = 'processing' or
        'failed') AND still have clip files or partial exports on disk.

        If video_ids is given, only those rows are considered (status filter
        is bypassed) — the on-disk clip/partial check still applies.

        `excel_path` is accepted for call-site compatibility and IGNORED —
        selection reads the videos table of dataset.db (storage v2).
        """
        eligible = self.catalog.db.videos.select_for_batch(
            status_filter=["processing", "failed"], video_ids=video_ids)

        resumable_rows = []
        for row in eligible:
            vid = str(row["video_id"])
            has_clips = (self.config.clips_dir / vid).exists() and any(
                (self.config.clips_dir / vid).glob("*.mp4")
            )
            has_partial = (self.config.processed_dir / vid / "face_crop").exists() and any(
                (self.config.processed_dir / vid / "face_crop").glob("*.mp4")
            )
            if has_clips or has_partial:
                resumable_rows.append(row)

        if limit:
            resumable_rows = resumable_rows[:limit]

        logger.info(f"Found {len(resumable_rows)} resumable video(s)")
        results = []
        for i, row in enumerate(resumable_rows, 1):
            vid = str(row["video_id"])
            url = str(row.get("youtube_url") or "")
            logger.info(f"Resuming {i}/{len(resumable_rows)}: {vid}")
            results.append(self.resume_video(vid, url))

        return results

    # batch processing
    def process_batch(
        self,
        excel_path: Path,
        status_filter: Optional[List[str]] = None,
        limit: Optional[int] = None,
        video_ids: Optional[List[str]] = None,
    ) -> List[ProcessingResult]:
        """Process a selection of videos from the catalog (videos table).

        If video_ids is given, only those rows are processed (status_filter
        is ignored). `excel_path` is accepted for call-site compatibility
        and IGNORED — the DB is the source of truth (storage v2).
        """
        rows = self.catalog.db.videos.select_for_batch(
            status_filter=status_filter, video_ids=video_ids, limit=limit)

        results: List[ProcessingResult] = []
        total = len(rows)

        for i, row in enumerate(rows, start=1):
            video_id = str(row["video_id"])
            url = str(row.get("youtube_url") or "")

            license_val = str(row.get("license") or "unverified").strip()
            verify_cc = license_val in ("", "unverified")

            logger.info(f"Processing video {i}/{total}: {video_id}")
            self._check_cancel(video_id)
            self._report_progress(
                video_id=video_id, stage="batch", video_num=i, total_videos=total,
            )

            result = self.process_video(
                video_id,
                url,
                verify_cc=verify_cc,
            )
            results.append(result)

        return results

    def sync_excel_from_disk(self, *args, **kwargs):
        """CLI-facing delegate — the implementation lives in CatalogWriter."""
        return self.catalog.sync_from_disk(*args, **kwargs)
