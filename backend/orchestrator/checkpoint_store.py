"""
Checkpoint store — resume state + reconstruction of finished work.

The per-video checkpoint (data/clips/{video_id}/.checkpoint.json) records
every processed clip's outcome the moment it completes. On resume, done
clips are skipped and their ExportedSegments are rebuilt from disk without
re-running any model. (Storage v2 will fold this into dataset.db — the
INSERT becomes the checkpoint.)
"""

import json
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from loguru import logger

from orchestrator.pipeline_config import PipelineConfig
from services.mouth_exporter.annotation_io import parse_lrs2_annotation
from services.mouth_exporter.segment_record import ExportedSegment


class CheckpointStore:
    """Read/write resume state for one videos directory layout."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    @staticmethod
    def parse_annotation(path: Path) -> dict:
        """Annotation file → the legacy key shape the orchestrator uses.

        Thin adapter over annotation_io.parse_lrs2_annotation (the single
        implementation of the format) — this replaced a full duplicate
        parser that used to live here.
        """
        parsed = parse_lrs2_annotation(path)
        return {
            "text": parsed["text"],
            "original": parsed["original"],
            "conf": parsed["confidence"] or 2,
            "words": [
                {"word": w["word"], "start": w["start"], "end": w["end"],
                 "score": w["asd_score"]}
                for w in parsed["words"]
            ],
        }

    def path_for(self, video_id: str) -> Path:
        return self.config.clips_dir / video_id / ".checkpoint.json"

    def read(self, video_id: str) -> dict:
        path = self.path_for(video_id)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Could not read checkpoint for {video_id}: {e}")
            return {}

    def write(self, video_id: str, data: dict):
        path = self.path_for(video_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not write checkpoint for {video_id}: {e}")

    def segment_from_disk(
        self,
        video_id: str,
        segment_id: str,
        video_file: Path,
        anno_file: Path,
        checkpoint_info: Optional[dict] = None,
    ) -> Optional["ExportedSegment"]:
        """Reconstruct an ExportedSegment from files already on disk.

        checkpoint_info, if provided, supplies the precise timing and score
        values that were saved when the segment was first exported.  Without
        it, duration is estimated from the video file and scores default to 0.
        """
        try:
            ann = self.parse_annotation(anno_file)
            word_scores = [w["score"] for w in ann["words"]]

            if checkpoint_info:
                duration          = checkpoint_info.get("duration") or 0.0
                start_time        = checkpoint_info.get("start_time") or 0.0
                end_time          = checkpoint_info.get("end_time") or duration
                asd_score         = checkpoint_info.get("asd_score") or 0.0
                syncnet_confidence= checkpoint_info.get("syncnet_confidence") or 0.0
                whisper_confidence= checkpoint_info.get("whisper_confidence") or 0.0
                # Phase 0 quality metadata — absent in pre-upgrade checkpoints
                face_visibility_ratio = checkpoint_info.get("face_visibility_ratio") or 0.0
                whisper_conf_min  = checkpoint_info.get("whisper_conf_min")
                whisper_conf_p25  = checkpoint_info.get("whisper_conf_p25")
                asd_method        = checkpoint_info.get("asd_method") or ""
                syncnet_method    = checkpoint_info.get("syncnet_method") or ""
                face_bbox         = checkpoint_info.get("face_bbox") or ""
                mouth_fail_rate   = checkpoint_info.get("mouth_landmark_fail_rate") or 0.0
                mouth_roi_method  = checkpoint_info.get("mouth_roi_method") or ""
                head_pose_avg     = checkpoint_info.get("head_pose_avg") or ""
                quality_tier      = checkpoint_info.get("quality_tier") or ""
                audio_speaker     = checkpoint_info.get("audio_speaker_label") or ""
            else:
                cap = cv2.VideoCapture(str(video_file))
                try:
                    fps_v    = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                finally:
                    cap.release()
                duration           = n_frames / fps_v if fps_v > 0 else 0.0
                start_time         = 0.0
                end_time           = duration
                asd_score          = float(np.mean(word_scores)) if word_scores else 0.0
                syncnet_confidence = 0.0
                whisper_confidence = 0.0
                face_visibility_ratio = 0.0
                whisper_conf_min   = None
                whisper_conf_p25   = None
                asd_method         = ""
                syncnet_method     = ""
                face_bbox          = ""
                mouth_fail_rate    = 0.0
                mouth_roi_method   = ""
                head_pose_avg      = ""
                quality_tier       = ""
                audio_speaker      = ""

            return ExportedSegment(
                segment_id=segment_id,
                video_id=video_id,
                video_path=video_file,
                annotation_path=anno_file,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                text=ann["text"],
                num_words=len(ann["words"]),
                track_id=0,
                asd_score=asd_score,
                syncnet_confidence=syncnet_confidence,
                whisper_confidence=whisper_confidence,
                face_visibility_ratio=face_visibility_ratio,
                whisper_conf_min=(float(whisper_conf_min) if whisper_conf_min is not None
                                  else float("nan")),
                whisper_conf_p25=(float(whisper_conf_p25) if whisper_conf_p25 is not None
                                  else float("nan")),
                asd_method=asd_method,
                syncnet_method=syncnet_method,
                face_bbox=face_bbox,
                mouth_landmark_fail_rate=float(mouth_fail_rate),
                mouth_roi_method=mouth_roi_method,
                head_pose_avg=head_pose_avg,
                quality_tier=quality_tier,
                audio_speaker_label=audio_speaker,
            )
        except Exception as e:
            logger.warning(f"Could not reconstruct segment {segment_id}: {e}")
            return None

    def recover_segments(
        self, video_id: str, checkpoint: dict
    ) -> List["ExportedSegment"]:
        """
        Load already-exported segments listed in the checkpoint.
        Returns them sorted by segment_id so segment_index stays consistent.
        """
        segments = []
        for _clip_id, info in checkpoint.get("processed_clips", {}).items():
            if info.get("result") == "rejected":
                continue
            if info.get("result") != "exported":
                continue
            seg_id = info.get("segment_id")
            if not seg_id:
                continue
            video_file = self.config.processed_dir / video_id / "face_crop" / f"{seg_id}.mp4"
            anno_file = self.config.processed_dir / video_id / "text" / f"{seg_id}.txt"
            if not (video_file.exists() and anno_file.exists()):
                logger.warning(
                    f"Checkpoint references missing segment {seg_id} — will re-process"
                )
                continue
            seg = self.segment_from_disk(
                video_id, seg_id, video_file, anno_file,
                checkpoint_info=info,
            )
            if seg:
                segments.append(seg)
        segments.sort(key=lambda s: s.segment_id)
        return segments
