"""
Segment records — the data types this service produces.

ExportedSegment is the future catalog row: everything measured about one
exported segment travels in this object to the checkpoint and the index.
ExportQC carries the per-export quality-control metrics.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ExportQC:
    """Quality-control metadata produced while exporting one segment."""
    mouth_landmark_fail_rate: float = 0.0   # frames without a fresh landmark hit
    mouth_roi_method: str = ""              # mediapipe | retinaface | retinaface_fallback
    head_pose_avg: str = ""                 # JSON "[yaw_proxy, pitch_proxy, roll_deg]"
    # Face-crop frames sampled during the write loop (BGR, output_size) —
    # handed to speaker identity so the fresh mp4 is never re-decoded.
    identity_frames: List = field(default_factory=list)


@dataclass
class ExportedSegment:
    """Record of an exported segment."""
    segment_id: str
    video_id: str
    video_path: Path           # face_crop video
    annotation_path: Path
    start_time: float
    end_time: float
    duration: float
    text: str
    num_words: int
    track_id: int
    asd_score: float
    syncnet_confidence: float
    whisper_confidence: float
    mouth_video_path: Optional[Path] = None  # mouth_crop video

    # Quality metadata (written to segments_index.csv so the dataset can be
    # filtered/tiered without reprocessing):
    face_visibility_ratio: float = 0.0       # fraction of frames with a face
    whisper_conf_min: float = float("nan")   # worst word confidence
    whisper_conf_p25: float = float("nan")   # 25th percentile word confidence
    asd_method: str = ""                     # "talknet" | "fallback_motion"
    syncnet_method: str = ""                 # "syncnet" | "fallback_correlation" | "disabled" | "error"
    face_bbox: str = ""                      # median track bbox, JSON "[x, y, w, h]"
    mouth_landmark_fail_rate: float = 0.0    # fraction of frames without fresh lip landmarks
    mouth_roi_method: str = ""               # mediapipe | retinaface | retinaface_fallback
    head_pose_avg: str = ""                  # JSON "[yaw_proxy, pitch_proxy, roll_deg]"
    speaker_id: str = ""                     # assigned by speaker identity clustering
    quality_tier: str = ""                   # A | B | C (see quality_tiers.py)
    audio_speaker_label: str = ""            # diarization voice (SPEAKER_00…) — consensus input
    av_speaker_mismatch: Optional[bool] = None  # voice↔face consensus verdict (None = not judged)

    # Transient handoff (NOT a catalog column): face-crop frames sampled
    # during export, consumed by speaker identity, then dropped.
    identity_frames: Optional[List] = field(default=None, repr=False, compare=False)
