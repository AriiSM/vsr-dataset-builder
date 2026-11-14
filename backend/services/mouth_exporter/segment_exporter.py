"""
Segment Exporter — the mouth_exporter service's orchestrator.

Turns one selected clip (winning track + words + scores) into the four
artifacts of an exported segment:
    face_crop/{segment_id}.mp4   — head, 256×256 (review + identity)
    mouth_crop/{segment_id}.mp4  — mouth, 96×96 grayscale, roll-aligned
                                   (the training data)
    audio/{segment_id}.wav       — the segment's audio (storage v2)
    {segment_id}.txt             — transcript + per-word timing (annotation)

Two passes over the clip (deliberate — memory stays at one frame):
    pass 1: MediaPipe lip landmarks on the stabilized face ROI
    pass 2: write face+mouth crops simultaneously (single-encode pipe)

Delegates to: crop_trajectories (WHERE to cut), video_encoder (HOW to
encode), annotation_io (text), audio_slicer (audio), mouth_landmarks
(lip geometry). All formulas unchanged from the Phase 2-3 implementation.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from loguru import logger
import json

from services.mouth_exporter.annotation_io import write_annotation
from services.mouth_exporter.audio_slicer import write_segment_audio
from services.mouth_exporter.crop_trajectories import (
    build_face_trajectory,
    build_mouth_trajectory_v1,
    build_mouth_trajectory_v2,
)
from services.mouth_exporter.mouth_landmarks import (
    LipLandmarks,
    MouthLandmarker,
)
from services.mouth_exporter.segment_record import ExportQC, ExportedSegment
from services.mouth_exporter.video_encoder import FfmpegPipeWriter, reencode_with_audio


class SegmentExporter:
    """
    Export segments to LRS2-compatible format.

    Output structure (storage v2 — the per-video folder is self-contained):
        processed/{video_id}/face_crop/{segment_id}.mp4   — face at video.output_resolution (256×256)
        processed/{video_id}/mouth_crop/{segment_id}.mp4  — 96×96 mouth
        processed/{video_id}/audio/{segment_id}.wav       — segment audio
        processed/{video_id}/text/{segment_id}.txt        — annotation

    Annotation format:
        Text:  TRANSCRIBED TEXT HERE
        Conf:  2
        WORD START END ASDSCORE
        WORD1 0.00 0.25 10.5
        WORD2 0.25 0.50 11.2
        ...
    """

    def __init__(
        self,
        processed_dir: Path,
        output_fps: float,
        output_size: Tuple[int, int],
        mouth_size: Tuple[int, int],
        crop_margin: float,
        include_audio: bool,
        video_codec: str,
        video_crf: int,
        gaussian_smoothing_sigma: float,
        mouth_gaussian_smoothing_sigma: float,
        minimum_crop_size_pixels: int,
        mouth_width_multiplier: float,
        mouth_min_half_size_pixels: int,
        video_preset: str = "veryfast",
        use_ffmpeg_pipe: bool = True,
        mouth_roi_method: str = "mediapipe",
        mouth_roi_min_confidence: float = 0.5,
        mouth_smoothing_type: str = "one_euro",
        one_euro_min_cutoff: float = 1.0,
        one_euro_beta: float = 0.3,
        mouth_roi_width_multiplier: float = 1.8,
        mouth_grayscale: bool = True,
        mouth_align_roll: bool = True,
        mouth_fail_rate_fallback: float = 0.5,
        save_segment_audio: bool = True,
        identity_samples: int = 0,
    ):
        self.processed_dir = Path(processed_dir)
        self.output_fps = output_fps
        self.output_size = output_size
        self.mouth_size = mouth_size
        self.crop_margin = crop_margin  # extra buffer around max face size
        self.include_audio = include_audio
        self.video_codec = video_codec
        self.video_crf = video_crf
        self.gaussian_smoothing_sigma = gaussian_smoothing_sigma
        self.mouth_gaussian_smoothing_sigma = mouth_gaussian_smoothing_sigma
        self.minimum_crop_size_pixels = minimum_crop_size_pixels
        self.mouth_width_multiplier = mouth_width_multiplier
        self.mouth_min_half_size_pixels = mouth_min_half_size_pixels
        self.video_preset = video_preset
        # True → single-encode export via _FfmpegPipeWriter (default).
        # False → legacy mp4v temp file + re-encode (rollback switch).
        self.use_ffmpeg_pipe = use_ffmpeg_pipe

        # Mouth ROI v2 (MediaPipe dense lip landmarks, every frame).
        # method "retinaface" keeps the legacy 2-point estimate; the actual
        # method used per segment is decided by _mouth_landmarker presence.
        self.mouth_smoothing_type = mouth_smoothing_type
        self.one_euro_min_cutoff = one_euro_min_cutoff
        self.one_euro_beta = one_euro_beta
        self.mouth_roi_width_multiplier = mouth_roi_width_multiplier
        self.mouth_grayscale = mouth_grayscale
        self.mouth_align_roll = mouth_align_roll
        self.mouth_fail_rate_fallback = mouth_fail_rate_fallback
        # Storage v2: write audio/{segment_id}.wav next to the crops
        self.save_segment_audio = save_segment_audio
        # >0 → collect this many evenly-spaced face-crop frames during the
        # write loop and hand them to speaker identity via the returned
        # segment (saves a full re-open + seek + decode of the fresh mp4).
        self.identity_samples = identity_samples
        self._mouth_landmarker = (
            MouthLandmarker(min_detection_confidence=mouth_roi_min_confidence)
            if mouth_roi_method == "mediapipe" else None
        )

        # Create directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def export_segment(
        self,
        source_video: Path,
        video_id: str,
        segment_index: int,
        track: Any,  # FaceTrack
        transcription: Any,  # TranscribedSegment
        asd_scores: List[float],
        syncnet_confidence: float,
        fps: float = 25.0,
        clip_id: Optional[str] = None,
        face_visibility_ratio: float = 0.0,
        asd_method: str = "",
        syncnet_method: str = "",
        source_time_offset: float = 0.0,
        audio_speaker_label: str = "",
        audio_source: Optional[Path] = None,
    ) -> Optional[ExportedSegment]:
        """
        Export a single segment with face crop and annotation.

        Args:
            source_video: Path to source video
            video_id: Video identifier
            segment_index: Segment index within video
            track: Face track for this segment
            transcription: Transcribed segment with words
            asd_scores: Per-word ASD scores
            syncnet_confidence: SyncNet confidence score
            fps: Source video FPS
            clip_id: VAD clip identifier (e.g. "md_001_clip_042"). When provided,
                     used as the output filename so the exported file is traceable
                     back to its source clip.

        Returns:
            ExportedSegment record, or None if export failed
        """
        # Name combines clip origin + global export index for clear ordering:
        # e.g. md_001_clip_042_00003.mp4
        if clip_id:
            segment_id = f"{clip_id}_{segment_index:05d}"
        else:
            segment_id = f"{video_id}_{segment_index:05d}"
        
        # Create output directories
        face_out_dir  = self.processed_dir / video_id / "face_crop"
        mouth_out_dir = self.processed_dir / video_id / "mouth_crop"
        anno_out_dir  = self.processed_dir / video_id / "text"
        face_out_dir.mkdir(parents=True, exist_ok=True)
        mouth_out_dir.mkdir(parents=True, exist_ok=True)
        anno_out_dir.mkdir(parents=True, exist_ok=True)

        face_path  = face_out_dir  / f"{segment_id}.mp4"
        mouth_path = mouth_out_dir / f"{segment_id}.mp4"
        annotation_path = anno_out_dir / f"{segment_id}.txt"

        # Export face + mouth crops in a single video read pass
        export_qc = self._export_dual_crop(
            source_video,
            face_path,
            mouth_path,
            track,
            transcription.start,
            transcription.end,
            fps,
            audio_source=audio_source,
        )

        if export_qc is None:
            logger.error(f"Failed to export crops for {segment_id}")
            return None

        if not mouth_path.exists():
            mouth_path = None

        # Export annotation (the segment's TEXT artifact)
        write_annotation(annotation_path, transcription, asd_scores)

        # The segment's AUDIO artifact — sliced from the clip wav by sample
        # math (stdlib), zero external processes.
        if self.save_segment_audio and audio_source is not None:
            audio_out_dir = self.processed_dir / video_id / "audio"
            audio_out_dir.mkdir(parents=True, exist_ok=True)
            write_segment_audio(
                audio_source,
                audio_out_dir / f"{segment_id}.wav",
                transcription.start,
                transcription.end,
            )

        avg_asd = np.mean(asd_scores) if asd_scores else 0.0

        # Median bounding box of the track — a compact spatial fingerprint of
        # where the speaker's face sits in the source clip.
        face_bbox = ""
        if getattr(track, "detections", None):
            xs = [d.bbox.x for d in track.detections]
            ys = [d.bbox.y for d in track.detections]
            ws = [d.bbox.width for d in track.detections]
            hs = [d.bbox.height for d in track.detections]
            face_bbox = json.dumps([
                int(np.median(xs)), int(np.median(ys)),
                int(np.median(ws)), int(np.median(hs)),
            ])

        return ExportedSegment(
            segment_id=segment_id,
            video_id=video_id,
            video_path=face_path,
            annotation_path=annotation_path,
            # ABSOLUTE source-video seconds (transcription times are clip-file
            # relative; source_time_offset = clip.content_start). Without this
            # offset the segment's position in the original video is lost the
            # moment clips.json is cleaned up.
            start_time=source_time_offset + transcription.start,
            end_time=source_time_offset + transcription.end,
            duration=transcription.duration,
            text=transcription.text,
            num_words=transcription.num_words,
            track_id=track.track_id,
            asd_score=avg_asd,
            syncnet_confidence=syncnet_confidence,
            whisper_confidence=transcription.confidence,
            mouth_video_path=mouth_path,
            face_visibility_ratio=face_visibility_ratio,
            whisper_conf_min=float(getattr(transcription, "confidence_min", float("nan"))),
            whisper_conf_p25=float(getattr(transcription, "confidence_p25", float("nan"))),
            asd_method=asd_method,
            syncnet_method=syncnet_method,
            face_bbox=face_bbox,
            mouth_landmark_fail_rate=export_qc.mouth_landmark_fail_rate,
            mouth_roi_method=export_qc.mouth_roi_method,
            head_pose_avg=export_qc.head_pose_avg,
            audio_speaker_label=audio_speaker_label,
            identity_frames=export_qc.identity_frames or None,
        )

    def _compute_mouth_landmarks_pass(
        self,
        source_video: Path,
        smoothed_face: Dict[int, Any],
        start_frame: int,
        end_frame: int,
    ) -> Dict[int, LipLandmarks]:
        """Pass 1 of the mouth ROI v2 export: decode the clip once and run
        MediaPipe FaceMesh on the stabilized face ROI of every frame.

        Decoding a ≤15 s clip twice (landmarks pass + write pass) costs ~1-2 s
        and avoids holding hundreds of frames in memory.
        """
        landmarks_by_frame: Dict[int, LipLandmarks] = {}
        cap = cv2.VideoCapture(str(source_video))
        if not cap.isOpened():
            return landmarks_by_frame

        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if start_frame <= frame_index < end_frame:
                roi = None
                face_box = smoothed_face.get(frame_index)
                if face_box is not None:
                    roi = (face_box.x, face_box.y, face_box.width, face_box.height)
                result = self._mouth_landmarker.detect(frame, roi=roi)
                if result is not None:
                    landmarks_by_frame[frame_index] = result
            frame_index += 1
            if frame_index >= end_frame:
                break

        cap.release()
        return landmarks_by_frame

    def _crop_mouth_v2(
        self,
        frame: np.ndarray,
        position: Tuple[int, int, int, float],
    ) -> np.ndarray:
        """Mouth crop with optional roll alignment and grayscale conversion.

        Output stays 3-channel BGR (grayscale replicated) so both writer
        paths handle it identically; the encode is yuv420 either way.
        """
        center_x, center_y, half, roll_degrees = position

        if self.mouth_align_roll and abs(roll_degrees) > 1.0:
            # Rotate the frame around the mouth center so the lip line is
            # horizontal — matches Auto-AVSR style preprocessing.
            rotation = cv2.getRotationMatrix2D((center_x, center_y), roll_degrees, 1.0)
            frame = cv2.warpAffine(
                frame, rotation, (frame.shape[1], frame.shape[0]),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
            )

        crop = self._crop_mouth(frame, (center_x, center_y, half))

        if self.mouth_grayscale:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            crop = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return crop

    def _export_dual_crop(
        self,
        source_video: Path,
        face_path: Path,
        mouth_path: Path,
        track: Any,
        start_time: float,
        end_time: float,
        source_fps: float,
        audio_source: Optional[Path] = None,
    ) -> Optional[ExportQC]:
        """Export face-crop (video.output_resolution) and mouth-crop
        (export.mouth_resolution) in a single video read.

        audio_source: the clip's .wav (the clip mp4 is video-only). Falls
        back to source_video for legacy callers whose mp4 still has audio.
        Returns ExportQC on success, None on failure.
        """
        start_frame = int(start_time * source_fps)
        end_frame   = int(end_time   * source_fps)
        duration    = end_time - start_time

        # Pre-compute smoothed trajectories
        smoothed_face = build_face_trajectory(
            track, start_frame, end_frame,
            smoothing_sigma=self.gaussian_smoothing_sigma,
            crop_margin=self.crop_margin,
            minimum_half_size_pixels=self.minimum_crop_size_pixels,
        )
        if not smoothed_face and track.detections:
            from services.face_tracker.geometry import BoundingBox
            sizes = [max(d.bbox.width, d.bbox.height) for d in track.detections]
            half  = int(max(sizes) * (1.0 + self.crop_margin) / 2.0)
            d     = track.detections[len(track.detections) // 2]
            cx    = d.bbox.x + d.bbox.width  // 2
            cy    = d.bbox.y + d.bbox.height // 2
            fb    = BoundingBox(cx - half, cy - half, half * 2, half * 2, 1.0)
            smoothed_face = {f: fb for f in range(start_frame, end_frame)}

        # Mouth trajectory: MediaPipe dense landmarks (v2) with controlled
        # fallback to the legacy RetinaFace 2-point estimate.
        qc = ExportQC()
        smoothed_mouth: Dict[int, Tuple[int, int, int, float]] = {}
        if self._mouth_landmarker is not None:
            landmarks = self._compute_mouth_landmarks_pass(
                source_video, smoothed_face, start_frame, end_frame,
            )
            trajectory, fail_rate, head_pose = build_mouth_trajectory_v2(
                landmarks, start_frame, end_frame,
                fps=self.output_fps,
                smoothing_type=self.mouth_smoothing_type,
                one_euro_min_cutoff=self.one_euro_min_cutoff,
                one_euro_beta=self.one_euro_beta,
                gaussian_sigma=self.mouth_gaussian_smoothing_sigma,
                width_multiplier=self.mouth_roi_width_multiplier,
                minimum_half_size_pixels=self.mouth_min_half_size_pixels,
            )
            qc.mouth_landmark_fail_rate = round(fail_rate, 4)
            qc.head_pose_avg = head_pose
            if trajectory and fail_rate <= self.mouth_fail_rate_fallback:
                smoothed_mouth = trajectory
                qc.mouth_roi_method = "mediapipe"
            else:
                logger.warning(
                    f"Mouth landmarks failed on {fail_rate:.0%} of frames — "
                    f"falling back to RetinaFace mouth estimate for {mouth_path.name}"
                )
                qc.mouth_roi_method = "retinaface_fallback"
        else:
            qc.mouth_roi_method = "retinaface"

        if not smoothed_mouth:
            # Legacy 2-point trajectory (roll = 0 → no alignment applied)
            legacy = build_mouth_trajectory_v1(
                track, start_frame, end_frame,
                smoothing_sigma=self.mouth_gaussian_smoothing_sigma,
                width_multiplier=self.mouth_width_multiplier,
                minimum_half_size_pixels=self.mouth_min_half_size_pixels,
            )
            smoothed_mouth = {
                f: (cx, cy, half, 0.0) for f, (cx, cy, half) in legacy.items()
            }

        # Open video once, write both crops simultaneously
        cap = cv2.VideoCapture(str(source_video))
        if not cap.isOpened():
            return None

        audio_source = (audio_source or source_video) if self.include_audio else None
        if self.use_ffmpeg_pipe:
            # Single-encode path: frames stream straight into the final H.264
            # file, audio muxed in the same invocation. One lossy generation.
            face_writer = FfmpegPipeWriter(
                face_path, self.output_size, self.output_fps,
                self.video_codec, self.video_crf, self.video_preset,
                audio_source=audio_source,
                audio_start=start_time, audio_duration=duration,
            )
            mouth_writer = FfmpegPipeWriter(
                mouth_path, self.mouth_size, self.output_fps,
                self.video_codec, self.video_crf, self.video_preset,
                audio_source=audio_source,
                audio_start=start_time, audio_duration=duration,
            )
            face_temp = mouth_temp = None
        else:
            # Legacy rollback path: mp4v temp files, re-encoded afterwards.
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            face_temp  = face_path.with_suffix('.temp.mp4')
            mouth_temp = mouth_path.with_suffix('.temp.mp4')
            face_writer  = cv2.VideoWriter(str(face_temp),  fourcc, self.output_fps, self.output_size)
            mouth_writer = cv2.VideoWriter(str(mouth_temp), fourcc, self.output_fps, self.mouth_size)

        # Identity evidence: evenly-spaced frame indices sampled while the
        # face crops are still in RAM — the encoded mp4 never gets re-read.
        identity_indices = set()
        if self.identity_samples > 0 and end_frame > start_frame:
            identity_indices = set(np.linspace(
                start_frame, end_frame - 1,
                num=min(self.identity_samples, end_frame - start_frame),
                dtype=int,
            ).tolist())

        last_face_bbox = None
        last_mouth_pos = None
        fi = 0
        fw = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if start_frame <= fi < end_frame:
                # Face crop
                bbox = smoothed_face.get(fi) or last_face_bbox
                if bbox is not None:
                    last_face_bbox = bbox
                    face_crop = self._crop_face(frame, bbox)
                    if fi in identity_indices:
                        qc.identity_frames.append(face_crop)
                else:
                    face_crop = np.zeros((*self.output_size, 3), dtype=np.uint8)
                face_writer.write(face_crop)

                # Mouth crop
                pos = smoothed_mouth.get(fi) or last_mouth_pos
                if pos is not None:
                    last_mouth_pos = pos
                    mouth_crop = self._crop_mouth_v2(frame, pos)
                else:
                    mouth_crop = np.zeros((*self.mouth_size, 3), dtype=np.uint8)
                mouth_writer.write(mouth_crop)

                fw += 1
            fi += 1
            if fi >= end_frame:
                break

        cap.release()

        if self.use_ffmpeg_pipe:
            face_ok = face_writer.release()
            mouth_ok = mouth_writer.release()
            if fw == 0:
                face_path.unlink(missing_ok=True)
                mouth_path.unlink(missing_ok=True)
                return None
        else:
            face_writer.release()
            mouth_writer.release()
            if fw == 0:
                face_temp.unlink(missing_ok=True)
                mouth_temp.unlink(missing_ok=True)
                return None
            # Re-encode both with ffmpeg (adds audio + H.264)
            face_ok = reencode_with_audio(
                face_temp, face_path, audio_source or source_video,
                start_time, duration, self.video_codec, self.video_preset,
                self.video_crf, self.include_audio,
            )
            mouth_ok = reencode_with_audio(
                mouth_temp, mouth_path, audio_source or source_video,
                start_time, duration, self.video_codec, self.video_preset,
                self.video_crf, self.include_audio,
            )

        if not face_ok:
            return None
        if not mouth_ok:
            logger.warning("Mouth crop encode failed, face crop OK")
        return qc

    def _crop_face(self, frame: np.ndarray, bbox: Any) -> np.ndarray:
        """Clip bbox to frame bounds and resize to output_size.

        The bbox already embeds the margin (built by _smooth_bbox_trajectory),
        so no additional margin is applied here.
        """
        h, w = frame.shape[:2]
        x1 = max(0, bbox.x)
        y1 = max(0, bbox.y)
        x2 = min(w, bbox.x + bbox.width)
        y2 = min(h, bbox.y + bbox.height)
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return np.zeros((*self.output_size, 3), dtype=np.uint8)
        return cv2.resize(face, self.output_size)

    def _crop_mouth(self, frame: np.ndarray,
                    pos: Tuple[int, int, int]) -> np.ndarray:
        """Crop mouth region centred at (cx, cy) with given half_size."""
        cx, cy, half = pos
        h, w = frame.shape[:2]
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half)
        y2 = min(h, cy + half)
        mouth = frame[y1:y2, x1:x2]
        if mouth.size == 0:
            return np.zeros((*self.mouth_size, 3), dtype=np.uint8)
        return cv2.resize(mouth, self.mouth_size)
