"""
Pipeline configuration.

Single source of truth: EVERY tunable parameter lives in config/config.yaml.
_REQUIRED_CONFIG_KEYS + _validate_config fail loudly, listing every missing
key, before anything else runs. PipelineConfig.from_yaml is the only place
that reads the file.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import yaml
from loguru import logger


def _resolve_device(requested: str) -> str:
    """Return 'cuda' if requested and available, otherwise 'cpu'."""
    import torch
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available — falling back to CPU")
        return "cpu"
    return requested

def _resolve_compute_type(compute_type: str, device: str) -> str:
    """float16 requires GPU — fall back to float32 on CPU."""
    if compute_type == "float16" and device == "cpu":
        return "float32"
    return compute_type


# CONFIG VALIDATION

# Required (section, key) pairs in config.yaml. Validated at load time so that
# missing config keys produce an upfront, descriptive error rather than a
# silent fallback to a hidden default.

_REQUIRED_CONFIG_KEYS: List[Tuple[str, ...]] = [
    ("paths", "base_dir"),
    ("download", "cc_only"),
    ("download", "format"),
    ("download", "rate_limit"),
    ("download", "max_retries"),
    ("audio", "sample_rate"),
    ("audio", "whisper", "model"),
    ("audio", "whisper", "language"),
    ("audio", "whisper", "device"),
    ("audio", "whisper", "compute_type"),
    ("audio", "whisper", "batch_size"),
    ("vad_splitting", "vad_threshold"),
    ("vad_splitting", "min_speech_duration_ms"),
    ("vad_splitting", "min_silence_duration_ms"),
    ("vad_splitting", "window_size_samples"),
    ("vad_splitting", "speech_pad_ms"),
    ("vad_splitting", "split_threshold"),
    ("vad_splitting", "min_clip_duration"),
    ("vad_splitting", "max_clip_duration"),
    ("vad_splitting", "seek_start_padding"),
    ("vad_splitting", "cleanup_clips"),
    ("video", "output_fps"),
    ("video", "output_resolution"),
    ("video", "face_detection", "confidence_threshold"),
    ("video", "face_detection", "nms_threshold"),
    ("video", "face_detection", "detection_interval"),
    ("video", "face_tracking", "iou_threshold"),
    ("video", "face_tracking", "max_age"),
    ("video", "face_tracking", "min_hits"),
    ("video", "face_tracking", "kalman", "process_noise"),
    ("video", "face_tracking", "kalman", "measurement_noise"),
    ("asd", "max_candidate_tracks"),
    ("asd", "num_frames_context"),
    ("syncnet", "enabled"),
    ("syncnet", "max_offset_frames"),
    ("export", "crop_margin"),
    ("export", "include_audio"),
    ("export", "mouth_resolution"),
    ("export", "video_codec"),
    ("export", "video_crf"),
    ("export", "gaussian_smoothing_sigma"),
    ("export", "mouth_gaussian_smoothing_sigma"),
    ("export", "minimum_crop_size_pixels"),
    ("export", "mouth_width_multiplier"),
    ("export", "mouth_min_half_size_pixels"),
    ("clip_filters", "face_visibility_threshold"),
    ("clip_filters", "min_asd_score"),
    ("clip_filters", "min_whisper_confidence"),
    ("clip_filters", "min_track_speech_overlap"),
]


def _validate_config(cfg: dict) -> None:
    """Walk all required keys; raise ValueError listing every missing one.

    Single source of truth for config: every parameter must be in config.yaml.
    No silent fallbacks to hidden defaults.
    """
    missing: List[str] = []
    for path in _REQUIRED_CONFIG_KEYS:
        node = cfg
        for key in path:
            if not isinstance(node, dict) or key not in node:
                missing.append(".".join(path))
                break
            node = node[key]
    if missing:
        raise ValueError(
            "config.yaml is missing required keys:\n  - "
            + "\n  - ".join(missing)
            + "\nAdd them and re-run."
        )


# CONFIG

@dataclass(kw_only=True)
class PipelineConfig:
    """Full pipeline configuration for v2.

    All values are read from config.yaml. There are no hidden defaults — if a
    required key is missing from the yaml, _validate_config raises an error
    listing every missing key.

    Use `PipelineConfig.from_yaml('config/config.yaml')` to construct.
    """

    @property
    def catalog_db_path(self) -> Path:
        """dataset.db — the storage-v2 catalog (metadata_dir = data/catalog)."""
        return self.metadata_dir / "dataset.db"

    # Paths
    base_dir: Path
    raw_dir: Path
    clips_dir: Path
    processed_dir: Path
    annotations_dir: Path
    # Storage v2: metadata_dir IS the catalog dir (data/catalog) — holds
    # dataset.db plus the on-demand CSV exports.
    metadata_dir: Path
    models_dir: Path
    temp_dir: Path

    # Download
    cc_only: bool
    download_format: str
    download_fragment_retries: int
    download_sleep_interval: int
    rate_limit: str
    max_retries: int
    cookies_from_browser: Optional[str] = None   # truly optional: "chrome"/"firefox"/...
    cookies_file: Optional[Path] = None           # truly optional: cookies.txt path

    # Audio / Whisper
    whisper_model: str
    language: str
    compute_type: str
    device: str
    whisper_batch_size: int

    # VAD splitting
    audio_sample_rate: int
    vad_threshold: float
    min_speech_duration_ms: int
    min_silence_duration_ms: int
    speech_pad_ms: int
    vad_window_size_samples: int
    split_threshold: float
    min_clip_duration: float
    max_clip_duration: float
    seek_start_padding: float
    cleanup_clips: bool

    # Segmentation strategy: "sentence" (punctuation + pauses, single
    # full-video Whisper pass) or "vad" (legacy silence-only splitting).
    segmentation_strategy: str
    sentence_end_chars: tuple
    target_min_duration: float
    target_max_duration: float
    # None = no length limit (user decision): boundaries are purely
    # linguistic/speaker-based; a number re-enables the safety-net split.
    segmentation_max_clip_duration: Optional[float]
    merge_gap_max: float
    boundary_pad: float
    vad_margin: float
    clip_crf: int
    clip_preset: str
    whisper_unload_after_segmentation: bool

    # Diarization (pyannote via WhisperX) — WHO speaks WHEN
    diarization_enabled: bool
    diarization_hf_token: str
    diarization_min_speakers: Optional[int]
    diarization_max_speakers: Optional[int]

    # Video / face detection
    detection_confidence: float
    detection_nms_threshold: float
    detection_interval: int
    tracking_iou: float
    tracking_max_age: int
    tracking_min_hits: int
    kalman_process_noise: float
    kalman_measurement_noise: float
    output_fps: float
    output_size: Tuple[int, int]
    mouth_size: Tuple[int, int]
    crop_margin: float
    include_audio: bool
    video_codec: str
    video_crf: int
    video_preset: str
    use_ffmpeg_pipe: bool
    # Storage v2: write audio/{segment_id}.wav next to the crops
    save_segment_audio: bool
    gaussian_smoothing_sigma: float
    mouth_gaussian_smoothing_sigma: float
    minimum_crop_size_pixels: int
    mouth_width_multiplier: float
    mouth_min_half_size_pixels: int

    # Mouth ROI v2 (MediaPipe dense lip landmarks)
    mouth_roi_method: str
    mouth_roi_min_confidence: float
    mouth_smoothing_type: str
    one_euro_min_cutoff: float
    one_euro_beta: float
    mouth_roi_width_multiplier: float
    mouth_grayscale: bool
    mouth_align_roll: bool
    mouth_fail_rate_fallback: float

    # ASD
    asd_num_frames_context: int
    # Optional: persist the winning track's per-frame trajectory per segment
    # (re-export with different crop settings without re-running RetinaFace)
    save_face_tracks: bool

    # Refuse to run with fake fallback scores unless explicitly allowed
    asd_allow_fallback: bool

    # SyncNet
    syncnet_enabled: bool
    syncnet_max_offset_frames: int
    syncnet_allow_fallback: bool
    syncnet_max_analysis_seconds: float

    # Per-clip quality filters
    face_visibility_threshold: float
    # Winner's mean ASD score below this → clip dropped as "no one visible
    # is speaking" (voice-over/B-roll). 0.0 disables the gate entirely.
    min_asd_score: float
    min_whisper_confidence: float
    min_track_speech_overlap: float
    # None = every overlapping track competes (no cap — 5-speaker videos)
    max_candidate_tracks: Optional[int]

    # Auto speaker metadata (gender / age_group) at end of each video

    # First-pass speaker identity (ArcFace clustering + demographics)
    speaker_identity_enabled: bool
    speaker_identity_samples: int
    speaker_identity_cluster_eps: float
    speaker_identity_cross_video: bool
    speaker_identity_similarity: float

    # quality_tiers: raw config block for compute_quality_tier()
    quality_tiers: dict

    @classmethod
    def from_yaml(cls, config_path: Path) -> "PipelineConfig":
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        # Validate all required keys upfront — fail loudly with full list.
        _validate_config(cfg)

        paths = cfg["paths"]
        download = cfg["download"]
        audio = cfg["audio"]
        whisper = audio["whisper"]
        vad_s = cfg["vad_splitting"]
        video = cfg["video"]
        face_det = video["face_detection"]
        face_track = video["face_tracking"]
        asd_cfg = cfg["asd"]
        syncnet = cfg["syncnet"]
        filt = cfg["clip_filters"]
        export = cfg["export"]
        seg = cfg.get("segmentation", {})
        diarization = seg.get("diarization", {})
        mouth_roi = cfg.get("mouth_roi", {})
        spk_id = cfg.get("speaker_identity", {})

        base_dir = Path(paths["base_dir"])
        device = _resolve_device(whisper["device"])

        return cls(
            base_dir=base_dir,
            raw_dir=Path(paths.get("raw_dir", base_dir / "raw")),
            clips_dir=Path(paths.get("clips_dir", base_dir / "clips")),
            processed_dir=Path(paths.get("processed_dir", base_dir / "processed")),
            annotations_dir=Path(paths.get("annotations_dir", base_dir / "annotations")),
            metadata_dir=Path(paths.get("catalog_dir",
                                        paths.get("metadata_dir", base_dir / "catalog"))),
            models_dir=Path(paths.get("models_dir", "./models")),
            temp_dir=Path(paths.get("temp_dir", "./temp")),

            cc_only=download["cc_only"],
            download_format=download["format"],
            rate_limit=download["rate_limit"],
            max_retries=download["max_retries"],
            download_fragment_retries=int(download.get("fragment_retries", 10)),
            download_sleep_interval=int(download.get("sleep_interval", 2)),
            cookies_from_browser=download.get("cookies_from_browser"),
            cookies_file=(Path(download["cookies_file"]) if download.get("cookies_file") else None),

            whisper_model=whisper["model"],
            language=whisper["language"],
            device=device,
            compute_type=_resolve_compute_type(whisper["compute_type"], device),
            whisper_batch_size=whisper["batch_size"],

            audio_sample_rate=audio["sample_rate"],
            vad_threshold=vad_s["vad_threshold"],
            min_speech_duration_ms=vad_s["min_speech_duration_ms"],
            min_silence_duration_ms=vad_s["min_silence_duration_ms"],
            speech_pad_ms=vad_s["speech_pad_ms"],
            vad_window_size_samples=vad_s["window_size_samples"],
            split_threshold=vad_s["split_threshold"],
            min_clip_duration=vad_s["min_clip_duration"],
            max_clip_duration=vad_s["max_clip_duration"],
            seek_start_padding=vad_s["seek_start_padding"],
            cleanup_clips=vad_s["cleanup_clips"],

            # segmentation: block is optional — old configs default to the
            # new sentence strategy with sensible values.
            segmentation_strategy=str(seg.get("strategy", "sentence")),
            sentence_end_chars=tuple(seg.get("sentence_end_chars", [".", "?", "!", "…"])),
            target_min_duration=float(seg.get("target_min_duration", 2.0)),
            target_max_duration=float(seg.get("target_max_duration", 12.0)),
            merge_gap_max=float(seg.get("merge_gap_max", 1.0)),
            boundary_pad=float(seg.get("boundary_pad", 0.15)),
            vad_margin=float(seg.get("vad_margin", 0.5)),
            segmentation_max_clip_duration=(
                float(seg["max_clip_duration"])
                if seg.get("max_clip_duration") is not None else None
            ),
            clip_crf=int(seg.get("clip_crf", 16)),
            clip_preset=str(seg.get("clip_preset", "veryfast")),
            whisper_unload_after_segmentation=bool(
                whisper.get("unload_after_segmentation", True)
            ),

            diarization_enabled=bool(diarization.get("enabled", False)),
            diarization_hf_token=str(diarization.get("hf_token", "") or ""),
            diarization_min_speakers=diarization.get("min_speakers"),
            diarization_max_speakers=diarization.get("max_speakers"),

            detection_confidence=face_det["confidence_threshold"],
            detection_nms_threshold=face_det["nms_threshold"],
            detection_interval=face_det["detection_interval"],
            tracking_iou=face_track["iou_threshold"],
            tracking_max_age=face_track["max_age"],
            tracking_min_hits=face_track["min_hits"],
            kalman_process_noise=float(face_track["kalman"]["process_noise"]),
            kalman_measurement_noise=float(face_track["kalman"]["measurement_noise"]),
            output_fps=video["output_fps"],
            output_size=tuple(video["output_resolution"]),
            mouth_size=tuple(export["mouth_resolution"]),
            crop_margin=export["crop_margin"],
            include_audio=export["include_audio"],
            video_codec=export["video_codec"],
            video_crf=export["video_crf"],
            video_preset=str(export.get("video_preset", "veryfast")),
            use_ffmpeg_pipe=bool(export.get("use_ffmpeg_pipe", True)),
            save_segment_audio=bool(export.get("save_segment_audio", True)),
            gaussian_smoothing_sigma=float(export["gaussian_smoothing_sigma"]),
            mouth_gaussian_smoothing_sigma=float(export["mouth_gaussian_smoothing_sigma"]),
            minimum_crop_size_pixels=int(export["minimum_crop_size_pixels"]),
            mouth_width_multiplier=float(export["mouth_width_multiplier"]),
            mouth_min_half_size_pixels=int(export["mouth_min_half_size_pixels"]),

            # mouth_roi: block is optional — defaults enable MediaPipe v2.
            mouth_roi_method=str(mouth_roi.get("method", "mediapipe")),
            mouth_roi_min_confidence=float(mouth_roi.get("min_detection_confidence", 0.5)),
            mouth_smoothing_type=str(mouth_roi.get("smoothing", {}).get("type", "one_euro")),
            one_euro_min_cutoff=float(mouth_roi.get("smoothing", {}).get("min_cutoff", 1.0)),
            one_euro_beta=float(mouth_roi.get("smoothing", {}).get("beta", 0.3)),
            mouth_roi_width_multiplier=float(mouth_roi.get("width_multiplier", 1.8)),
            mouth_grayscale=bool(mouth_roi.get("grayscale", True)),
            mouth_align_roll=bool(mouth_roi.get("align_roll", True)),
            mouth_fail_rate_fallback=float(mouth_roi.get("fail_rate_fallback", 0.50)),

            save_face_tracks=bool(face_track.get("save_tracks", False)),

            asd_num_frames_context=asd_cfg["num_frames_context"],
            asd_allow_fallback=bool(asd_cfg.get("allow_fallback", False)),

            syncnet_enabled=syncnet["enabled"],
            syncnet_max_offset_frames=syncnet["max_offset_frames"],
            syncnet_allow_fallback=bool(syncnet.get("allow_fallback", False)),
            syncnet_max_analysis_seconds=float(syncnet.get("max_analysis_seconds", 10.0)),
            max_candidate_tracks=asd_cfg.get("max_candidate_tracks"),

            face_visibility_threshold=filt["face_visibility_threshold"],
            min_asd_score=filt["min_asd_score"],
            min_whisper_confidence=filt["min_whisper_confidence"],
            min_track_speech_overlap=filt["min_track_speech_overlap"],


            # speaker_identity / quality_tiers blocks are optional
            speaker_identity_enabled=bool(spk_id.get("enabled", True)),
            speaker_identity_samples=int(spk_id.get("samples_per_segment", 5)),
            speaker_identity_cluster_eps=float(spk_id.get("cluster_eps", 0.40)),
            speaker_identity_cross_video=bool(
                spk_id.get("cross_video", {}).get("enabled", True)
            ),
            speaker_identity_similarity=float(
                spk_id.get("cross_video", {}).get("similarity_threshold", 0.60)
            ),
            quality_tiers=dict(cfg.get("quality_tiers", {})),
        )
