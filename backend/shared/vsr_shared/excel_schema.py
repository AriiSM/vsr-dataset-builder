"""
CSV schemas — column documentation + the storage-v1 export shapes.

Storage v2 made dataset.db the source of truth; this module keeps two
jobs: (1) ProcessingStatus — the video state machine used across the
pipeline; (2) the three *_SCHEMA dicts — self-documenting column
definitions that export_catalog.py uses to emit CSVs in the stable v1
shapes; Each schema value documents type / required / description /
example for humans reading the exports.
"""

from enum import Enum


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    VALIDATED = "validated"   # all clips reviewed and post-processed via GUI


# ===== CSV export shapes (column documentation) =====

VIDEOS_MASTER_SCHEMA = {
    # === Identification ===
    "video_id": {
        "type": "string",
        "required": True,
        "description": "Unique identifier (e.g., ro_001, md_042)",
        "example": "ro_001"
    },
    "youtube_url": {
        "type": "string",
        "required": True,
        "description": "Full YouTube URL",
        "example": "https://www.youtube.com/watch?v=XXXXXXXXXXX"
    },

    # === Source & Licensing ===
    "source": {
        "type": "category",
        "required": True,
        "options": ["TEDx", "YouTube_CC", "Interview", "Lecture", "Podcast", "News", "Other"],
        "description": "Video source type",
        "example": "TEDx"
    },
    "source_channel": {
        "type": "string",
        "required": False,
        "description": "YouTube channel name",
        "example": "TEDx Talks"
    },
    "license": {
        "type": "category",
        "required": True,
        "options": ["CC-BY", "CC-BY-SA", "CC-BY-NC", "CC-BY-ND", "CC0", "unverified"],
        "description": "License type. Use 'unverified' if not yet confirmed.",
        "example": "CC-BY"
    },

    # === Language & Region ===
    "region": {
        "type": "category",
        "required": True,
        "options": ["RO", "MD", "DIASPORA", "UNKNOWN"],
        "description": "Speaker region (RO=Romania, MD=Moldova)",
        "example": "RO"
    },

    # === Video Metadata ===
    "title": {
        "type": "string",
        "required": False,
        "description": "Original video title",
        "example": "Cum să înveți mai eficient | TEDx"
    },
    "duration_seconds": {
        "type": "float",
        "required": False,
        "description": "Total video duration in seconds",
        "example": 645.3
    },

    # === Speaker Metadata ===
    "speaker_id": {
        "type": "string",
        "required": False,
        "description": "Foreign key into speakers_registry.csv. Default {video_id}_spk0.",
        "example": "ro_001_spk0"
    },
    "num_speakers": {
        "type": "integer",
        "required": False,
        "description": "Number of speakers in video",
        "example": 1
    },
    "gender": {
        "type": "category",
        "required": False,
        "options": ["M", "F", "mixed", "unknown"],
        "description": "Speaker gender(s)",
        "example": "F"
    },
    "age_group": {
        "type": "category",
        "required": False,
        "options": ["18-30", "31-50", "51+", "mixed", "unknown"],
        "description": "Estimated age group",
        "example": "31-50"
    },

    # === Recording Conditions ===
    "environment": {
        "type": "category",
        "required": False,
        "options": ["indoor", "outdoor", "studio", "mixed", "unknown"],
        "description": "Recording environment",
        "example": "studio"
    },
    "background_noise": {
        "type": "category",
        "required": False,
        "options": ["none", "low", "moderate", "high"],
        "description": "Background noise level",
        "example": "low"
    },

    # === Processing Status ===
    "status": {
        "type": "category",
        "required": True,
        "options": [e.value for e in ProcessingStatus],
        "description": "Current processing status",
        "example": "pending"
    },

    # === Processing Results (auto-filled by pipeline) ===
    "processed_date": {
        "type": "datetime",
        "required": False,
        "description": "When processing completed (auto-filled)",
        "example": "2026-03-15 16:45:00"
    },
    "total_segments": {
        "type": "integer",
        "required": False,
        "description": "Number of extracted segments (auto-filled)",
        "example": 127
    },
    "total_duration_extracted": {
        "type": "float",
        "required": False,
        "description": "Total duration of extracted segments in seconds (auto-filled)",
        "example": 423.5
    },
    "avg_asd_score": {
        "type": "float",
        "required": False,
        "description": "Average Active Speaker Detection score (auto-filled)",
        "example": 8.7
    },
    "avg_syncnet_conf": {
        "type": "float",
        "required": False,
        "description": "Average SyncNet audio-visual sync confidence (auto-filled)",
        "example": 0.85
    },
    "error_message": {
        "type": "string",
        "required": False,
        "description": "Error message if processing failed (auto-filled)",
        "example": ""
    },
}


# ===== Segments Index Schema =====

SEGMENTS_INDEX_SCHEMA = {
    "segment_id": {
        "type": "string",
        "required": True,
        "description": "Unique segment identifier",
        "example": "ro_001_00127"
    },
    "video_id": {
        "type": "string",
        "required": True,
        "description": "Parent video identifier",
        "example": "ro_001"
    },
    "start_time": {
        "type": "float",
        "required": True,
        "description": "Segment start time in source video (seconds)",
        "example": 45.230
    },
    "end_time": {
        "type": "float",
        "required": True,
        "description": "Segment end time in source video (seconds)",
        "example": 48.750
    },
    "duration": {
        "type": "float",
        "required": True,
        "description": "Segment duration (seconds)",
        "example": 3.520
    },
    "text": {
        "type": "string",
        "required": True,
        "description": "Transcribed text",
        "example": "ACESTE ZILE CÂND GĂTEȘTI"
    },
    "num_words": {
        "type": "integer",
        "required": True,
        "description": "Number of words in transcription",
        "example": 4
    },
    "num_chars": {
        "type": "integer",
        "required": True,
        "description": "Number of characters in transcription",
        "example": 24
    },
    "speaker_id": {
        "type": "string",
        "required": False,
        "description": "Speaker identifier within video",
        "example": "spk_01"
    },
    "track_id": {
        "type": "integer",
        "required": False,
        "description": "Face track identifier",
        "example": 0
    },
    "asd_score": {
        "type": "float",
        "required": True,
        "description": "Active Speaker Detection score",
        "example": 9.2
    },
    "syncnet_conf": {
        "type": "float",
        "required": True,
        "description": "SyncNet confidence score",
        "example": 0.85
    },
    "whisper_conf": {
        "type": "float",
        "required": True,
        "description": "Whisper transcription confidence — mean across words",
        "example": 0.92
    },
    "whisper_conf_min": {
        "type": "float",
        "required": False,
        "description": "Whisper confidence — minimum word (penalises a single weak word)",
        "example": 0.41
    },
    "whisper_conf_p25": {
        "type": "float",
        "required": False,
        "description": "Whisper confidence — 25th percentile across words",
        "example": 0.78
    },
    "asd_method": {
        "type": "string",
        "required": False,
        "description": "ASD scoring path: talknet | fallback_motion. Fallback rows are NOT comparable to talknet rows.",
        "example": "talknet"
    },
    "syncnet_method": {
        "type": "string",
        "required": False,
        "description": "Sync scoring path: syncnet | fallback_correlation | insufficient_data | disabled | error.",
        "example": "syncnet"
    },
    "mouth_landmark_fail_rate": {
        "type": "float",
        "required": False,
        "description": "Fraction of exported frames without fresh MediaPipe lip landmarks (carried/fallback instead).",
        "example": 0.04
    },
    "mouth_roi_method": {
        "type": "string",
        "required": False,
        "description": "Mouth localization path: mediapipe | retinaface | retinaface_fallback.",
        "example": "mediapipe"
    },
    "quality_tier": {
        "type": "category",
        "required": False,
        "options": ["A", "B", "C"],
        "description": "Derived quality tier (see quality_tiers.py): A = trustworthy, B = usable with soft issues, C = dubious. Recomputable offline.",
        "example": "A"
    },
    "audio_speaker_label": {
        "type": "string",
        "required": False,
        "description": "Diarization voice label (per-video: SPEAKER_00…). Input for the per-video voice↔face consensus check.",
        "example": "SPEAKER_00"
    },
    "av_speaker_mismatch": {
        "type": "boolean",
        "required": False,
        "description": "Voice↔face consensus verdict: True = this segment contradicts its voice's per-video majority face (voice-over/B-roll suspect; tier capped at B). Blank = not judged (no diarization or no identity evidence).",
        "example": False
    },
    "text_largev3": {
        "type": "string",
        "required": False,
        "description": "Second-opinion transcript from Whisper large-v3 (transcript_refiner service).",
        "example": "ACESTE ZILE CÂND GĂTEȘTI"
    },
    "wer_medium_vs_large": {
        "type": "float",
        "required": False,
        "description": "Disagreement between the pipeline (medium) and large-v3 transcripts. High = transcript probably wrong.",
        "example": 0.08
    },
    "needs_review": {
        "type": "boolean",
        "required": False,
        "description": "True when wer_medium_vs_large exceeds the review threshold — prioritize these in the review GUI.",
        "example": False
    },
    "original_text": {
        "type": "string",
        "required": False,
        "description": "Whisper raw transcription captured at export — reference for WER when text is later edited manually.",
        "example": "ACESTE ZILE CAND GATESTI"
    },
    "wer": {
        "type": "float",
        "required": False,
        "description": "Word Error Rate between original_text and current text. 0 = unedited; null = no reference.",
        "example": 0.125
    },
    "wer_word_count_ref": {
        "type": "integer",
        "required": False,
        "description": "Word count in original_text (denominator context for WER).",
        "example": 4
    },
    "face_bbox": {
        "type": "string",
        "required": False,
        "description": "Face bounding box [x,y,w,h] (JSON)",
        "example": "[120, 80, 180, 220]"
    },
    "face_visibility_ratio": {
        "type": "float",
        "required": False,
        "description": "Ratio of frames with visible face",
        "example": 0.98
    },
    "head_pose_avg": {
        "type": "string",
        "required": False,
        "description": "Average head pose [yaw, pitch, roll] (JSON)",
        "example": "[5.2, -3.1, 1.5]"
    },
    "video_path": {
        "type": "string",
        "required": True,
        "description": "Relative path to segment video",
        "example": "processed/ro_001/ro_001_00127.mp4"
    },
    "annotation_path": {
        "type": "string",
        "required": True,
        "description": "Relative path to annotation file",
        "example": "annotations/ro_001/ro_001_00127.txt"
    },
    "split": {
        "type": "category",
        "required": False,
        "options": ["train", "val", "test"],
        "description": "Dataset split assignment",
        "example": "train"
    }
}


# ===== Speakers Registry Schema =====
#
# One row per speaker. Aggregate columns are recomputed from
# segments_index.csv by vsr_shared/speakers_registry.py:recompute_aggregates().

SPEAKERS_REGISTRY_SCHEMA = {
    "speaker_id": {
        "type": "string",
        "required": True,
        "description": "Unique identifier — typically `{video_id}_spk0` for single-speaker videos.",
        "example": "ro_001_spk0",
    },
    "speaker_name": {
        "type": "string",
        "required": False,
        "description": "Human-readable name (curator-supplied).",
        "example": "Andreea Esca",
    },
    "gender": {
        "type": "category",
        "required": False,
        "options": ["M", "F", "mixed", "unknown"],
        "description": "Speaker gender.",
        "example": "F",
    },
    "age_group": {
        "type": "category",
        "required": False,
        "options": ["18-30", "31-50", "51+", "mixed", "unknown"],
        "description": "Estimated age group.",
        "example": "31-50",
    },
    "age_estimate": {
        "type": "float",
        "required": False,
        "description": "Approximate numeric age — median of buffalo_l estimates over all sampled frames of this speaker's cluster (typical error ±5-8 years).",
        "example": 43.0,
    },
    "age_std": {
        "type": "float",
        "required": False,
        "description": "Spread of the per-frame age estimates. Large values mean the estimate is unreliable — review manually.",
        "example": 4.2,
    },
    "gender_confidence": {
        "type": "float",
        "required": False,
        "description": "Fraction of sampled frames voting for the majority gender (0.5 = coin flip, 1.0 = unanimous).",
        "example": 0.97,
    },
    "identity_match": {
        "type": "category",
        "required": False,
        "options": ["new", "auto", "manual"],
        "description": "How this speaker id was assigned: new cluster, automatic cross-video match (audit these), or manual correction.",
        "example": "auto",
    },
    "accent_region": {
        "type": "category",
        "required": False,
        "options": ["RO", "MD", "DIASPORA", "UNKNOWN"],
        "description": "Speaker accent region.",
        "example": "RO",
    },
    "num_videos": {
        "type": "integer",
        "required": False,
        "description": "Auto-aggregated: distinct videos featuring this speaker.",
        "example": 3,
    },
    "num_segments": {
        "type": "integer",
        "required": False,
        "description": "Auto-aggregated: total segments attributed to this speaker.",
        "example": 412,
    },
    "total_duration_s": {
        "type": "float",
        "required": False,
        "description": "Auto-aggregated: total spoken duration in seconds.",
        "example": 1845.7,
    },
    "avg_asd": {
        "type": "float",
        "required": False,
        "description": "Auto-aggregated: mean ASD score across segments.",
        "example": 8.4,
    },
    "avg_wer": {
        "type": "float",
        "required": False,
        "description": "Auto-aggregated: mean WER across segments with a reference.",
        "example": 0.034,
    },
}



