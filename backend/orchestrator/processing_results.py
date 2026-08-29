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
    # Scores measured BEFORE the drop decision (None = never computed).
    # Recorded on dropped_clips rows so thresholds calibrate on real data.
    whisper_conf: Optional[float] = None
    asd_score: Optional[float] = None


@dataclass
class AnalyzedClip:
    """GPU-phase result for one clip that passed every gate.

    Everything the CPU-only export phase needs (mouth crop + encode + tier +
    ArcFace identity) — hand-off payload of the GPU→CPU conveyor: while the
    export lane consumes this, the GPU thread analyzes the next clip.
    """
    clip: VideoClip
    best_track: object          # FaceTrack of the chosen speaker
    best_asd: object            # ASDResult of the winning track
    merged: object              # TranscribedSegment, trimmed to track range
    fps: float
    visibility: float
    syncnet_confidence: float   # measured on the GPU thread (or 0.0/disabled)
    syncnet_method: str

@dataclass
class ProcessingResult:
    """Result of processing one full video."""
    video_id: str
    status: ProcessingStatus
    segments: List[ExportedSegment] = field(default_factory=list)
    total_duration: float = 0.0
    error_message: Optional[str] = None
    processing_time: float = 0.0
