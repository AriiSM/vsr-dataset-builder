"""
Catalog writer — everything the pipeline records about videos and segments.

Storage v2, final form: dataset.db is the ONLY store — written automatically,
one transaction per clip. CSVs exist exclusively as on-demand exports
(backend/tools/export_catalog.py); nothing here writes or reads them.

Public API (call sites in pipeline.py unchanged):
    set_video_status · update_for_result · append_segment ·
    record_dropped_clip · store_segment_embedding · rewrite_segments_for ·
    sync_from_disk
"""

import json
import math
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
from loguru import logger

from orchestrator.checkpoint_store import CheckpointStore
from orchestrator.pipeline_config import PipelineConfig
from orchestrator.processing_results import ProcessingResult
from services.mouth_exporter.segment_record import ExportedSegment
from vsr_shared.catalog_db import CatalogDatabase
from vsr_shared.excel_schema import ProcessingStatus


def _nan_to_none(value):
    """DB representation for optional floats: NaN → NULL."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


class CatalogWriter:
    """Persists per-video status and per-segment rows into dataset.db."""

    def __init__(self, config: PipelineConfig, checkpoints: CheckpointStore):
        self.config = config
        # sync_from_disk / append_segment read annotation files back
        self.checkpoints = checkpoints
        self.db = CatalogDatabase(config.catalog_db_path)

    # ------------------------------------------------------------ video rows

    def set_video_status(self, video_id: str, status: ProcessingStatus):
        """Update a video's status (DB authoritative + CSV mirror)."""
        self.db.videos.set_status(video_id, status.value)

    def update_for_result(self, result: ProcessingResult):
        """Record a finished video: stats on the video row + segment rows."""
        fields = {
            "status": result.status.value,
            "processed_date": datetime.now().strftime("%Y-%m-%d"),
            "total_segments": len(result.segments),
            "total_duration_extracted": result.total_duration,
            "error_message": result.error_message or "",
        }
        if result.segments:
            fields["avg_asd_score"] = float(
                np.mean([s.asd_score for s in result.segments]))
            fields["avg_syncnet_conf"] = float(
                np.mean([s.syncnet_confidence for s in result.segments]))
        self.db.videos.update_fields(result.video_id, fields)

        self.rewrite_segments_for(result)
        self._backup_database()

    # ---------------------------------------------------------- segment rows

    @staticmethod
    def segment_to_db_row(seg: ExportedSegment) -> dict:
        """DB-shaped row: NaN → NULL, verdicts as 0/1/NULL, no path columns."""
        return {
            "segment_id": seg.segment_id,
            "video_id": seg.video_id,
            "clip_id": "_".join(seg.segment_id.split("_")[:-1]),
            "speaker_id": seg.speaker_id or f"{seg.video_id}_spk0",
            "track_id": seg.track_id,
            "start_time": seg.start_time,
            "end_time": seg.end_time,
            "duration": seg.duration,
            "text": seg.text,
            "original_text": seg.text,
            "num_words": seg.num_words,
            "num_chars": len(seg.text) if seg.text else 0,
            "asd_score": seg.asd_score,
            "asd_method": seg.asd_method,
            "syncnet_conf": seg.syncnet_confidence,
            "syncnet_method": seg.syncnet_method,
            "whisper_conf": seg.whisper_confidence,
            "whisper_conf_min": _nan_to_none(seg.whisper_conf_min),
            "whisper_conf_p25": _nan_to_none(seg.whisper_conf_p25),
            "face_bbox": seg.face_bbox,
            "face_visibility_ratio": round(seg.face_visibility_ratio, 4),
            "head_pose_avg": seg.head_pose_avg,
            "mouth_landmark_fail_rate": round(seg.mouth_landmark_fail_rate, 4),
            "mouth_roi_method": seg.mouth_roi_method,
            "quality_tier": seg.quality_tier,
            "audio_speaker_label": seg.audio_speaker_label,
            "av_speaker_mismatch": (
                None if seg.av_speaker_mismatch is None
                else int(seg.av_speaker_mismatch)
            ),
            "wer": 0.0,
            "wer_word_count_ref": seg.num_words,
        }

    def _words_from_annotation(self, seg: ExportedSegment) -> Optional[List[dict]]:
        """Per-word rows from the segment's annotation file (the format's
        single source of truth). None when the file cannot be parsed."""
        try:
            parsed = self.checkpoints.parse_annotation(Path(seg.annotation_path))
            return [
                {"word": w.get("word", ""), "start": w.get("start"),
                 "end": w.get("end"), "score": None,
                 "asd_score": w.get("score")}
                for w in parsed.get("words", [])
            ]
        except Exception as e:
            logger.debug(f"No words for {seg.segment_id}: {e}")
            return None

    def append_segment(self, seg: ExportedSegment):
        """Record ONE exported segment (called per clip, right after export).

        DB transaction inserts the segment + its words — the row becomes
        visible only after the segment's files are complete on disk.
        """
        self.db.videos.ensure_exists(seg.video_id)
        self.db.segments.upsert(
            self.segment_to_db_row(seg),
            words=self._words_from_annotation(seg),
        )

    def record_dropped_clip(self, video_id: str, clip, reason: str,
                            face_visibility: Optional[float] = None,
                            whisper_conf: Optional[float] = None,
                            asd_score: Optional[float] = None):
        """Persist a rejection WITH its reason — survives clips/ cleanup.

        Scores measured before the drop travel along (None = never computed)
        so gate thresholds can be calibrated from real distributions.
        """
        try:
            self.db.videos.ensure_exists(video_id)
            self.db.dropped.record(
                video_id, getattr(clip, "clip_id", str(clip)), reason,
                start_time=getattr(clip, "start_time", None),
                end_time=getattr(clip, "end_time", None),
                face_visibility=face_visibility,
                whisper_conf=whisper_conf,
                asd_score=asd_score,
            )
        except Exception as e:
            logger.debug(f"dropped_clips record skipped: {e}")

    def store_segment_embedding(self, segment_id: str, embedding) -> None:
        """ArcFace evidence per segment (identity audit / re-clustering)."""
        try:
            self.db.segments.set_embedding(segment_id, np.asarray(embedding))
        except Exception as e:
            logger.debug(f"segment embedding skipped for {segment_id}: {e}")

    def rewrite_segments_for(self, result: ProcessingResult):
        """End-of-video authoritative rewrite (final speaker_id / tier / av)."""
        if not result.segments:
            return

        self.db.segments.replace_for_video(
            result.video_id,
            [self.segment_to_db_row(seg) for seg in result.segments],
        )


    # -------------------------------------------------------------- sync

    def sync_from_disk(
        self,
        excel_path: Optional[Path] = None,
        video_ids: Optional[List[str]] = None,
    ) -> int:
        """Rebuild video stats for videos whose metadata is missing, from the
        segments table (primary) or the files on disk (fallback), writing the
        video rows in dataset.db. `excel_path` is accepted for call-site
        compatibility and IGNORED — CSVs are exports only (export_catalog.py).

        Returns the number of rows updated.
        """
        review_status: dict = {}
        review_path = self.config.metadata_dir / "review_status.json"
        if review_path.exists():
            try:
                review_status = json.loads(review_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Could not parse {review_path}: {e}")

        all_rows = self.db.videos.all()
        if video_ids:
            wanted = {str(v) for v in video_ids}
            rows_to_check = [r for r in all_rows if str(r["video_id"]) in wanted]
        else:
            rows_to_check = [r for r in all_rows
                             if not r.get("total_segments")]

        updated = 0
        for row in rows_to_check:
            vid = str(row["video_id"])
            proc_dir = self.config.processed_dir / vid / "face_crop"
            video_files = sorted(proc_dir.glob("*.mp4")) if proc_dir.exists() else []
            if not video_files:
                logger.debug(f"  {vid}: no segment files found, skipping")
                continue

            stats = self.db.segments.video_stats(vid)
            if stats:
                total_segments = stats["num_segments"]
                total_duration = stats["total_duration_s"] or 0.0
                avg_asd = stats["avg_asd"]
                avg_sync = stats["avg_syncnet"]
            else:
                total_segments, total_duration, avg_asd = \
                    self._stats_from_disk(vid, video_files)
                avg_sync = None

            new_status = self._validation_status(vid, review_status)

            fields = {
                "status": new_status,
                "total_segments": total_segments,
                "total_duration_extracted": round(total_duration, 2),
            }
            if avg_asd is not None:
                fields["avg_asd_score"] = round(avg_asd, 3)
            if avg_sync is not None:
                fields["avg_syncnet_conf"] = round(avg_sync, 3)
            self.db.videos.update_fields(vid, fields)

            logger.info(
                f"  {vid}: {total_segments} segments, {total_duration:.1f}s"
                + (f", asd={avg_asd:.2f}" if avg_asd is not None else "")
            )
            updated += 1

        if updated:
            logger.info(f"sync: updated {updated} video row(s) in dataset.db")
        return updated

    def _stats_from_disk(self, vid: str, video_files: List[Path]):
        """Fallback aggregation from mp4 durations + annotation ASD scores."""
        import cv2

        total_duration = 0.0
        asd_scores: List[float] = []
        text_dir = self.config.processed_dir / vid / "text"

        for vf in video_files:
            cap = cv2.VideoCapture(str(vf))
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.release()
            total_duration += frames / fps if fps > 0 else 0.0

            anno_file = text_dir / vf.with_suffix(".txt").name
            if anno_file.exists():
                try:
                    parsed = self.checkpoints.parse_annotation(anno_file)
                    asd_scores.extend(
                        float(w["score"]) for w in parsed.get("words", [])
                        if w.get("score") is not None
                    )
                except Exception:
                    pass

        avg_asd = float(np.mean(asd_scores)) if asd_scores else None
        return len(video_files), total_duration, avg_asd

    def _validation_status(self, vid: str, review_status: dict) -> str:
        """`validated` only when EVERY surviving segment has a review verdict."""
        segments = self.db.segments.for_video(vid)
        if segments and review_status:
            if all(
                review_status.get(s["segment_id"], {}).get("status")
                in ("approved", "rejected")
                for s in segments
            ):
                return "validated"
        return "completed"

    def _backup_database(self):
        """Online backup after each finished video, keep the last 5."""
        try:
            backups_dir = self.config.metadata_dir / "backups"
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.db.backup_to(backups_dir / f"dataset_{stamp}.db")
            existing = sorted(backups_dir.glob("dataset_*.db"))
            for old in existing[:-5]:
                old.unlink(missing_ok=True)
        except Exception as e:
            logger.debug(f"catalog backup skipped: {e}")
