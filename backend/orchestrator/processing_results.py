"""
Processing results — the orchestrator's per-clip and per-video outcomes.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from services.mouth_exporter.segment_record import ExportedSegment
from services.segmenter.clip_manifest import VideoClip
from vsr_shared.excel_schema import ProcessingStatus


class PipelineCancelled(Exception):
    """Raised when a cancel was requested (job queue) — the video stays in
    'processing' with its checkpoint intact, resumable later. Deliberately
    NOT a RuntimeError subclass so error paths don't swallow it."""

    def __init__(self, video_id: str):
        self.video_id = video_id
        super().__init__(f"cancelled while processing {video_id}")


@dataclass
class ClipResult:
    """Processing result for one VAD clip."""
    clip: VideoClip
    dropped: bool = False
    drop_reason: Optional[str] = None
    face_visibility_ratio: float = 0.0
    exported_segment: Optional[ExportedSegment] = None
    # ArcFace identity evidence (JSON-able dict) for speaker clustering
    identity: Optional[dict] = None

@dataclass
class ProcessingResult:
    """Result of processing one full video."""
    video_id: str
    status: ProcessingStatus
    segments: List[ExportedSegment] = field(default_factory=list)
    total_duration: float = 0.0
    error_message: Optional[str] = None
    processing_time: float = 0.0
