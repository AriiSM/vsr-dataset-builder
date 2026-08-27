"""
Service container — every microservice, constructed lazily, in one place.

The pipeline touches models only through this container: nothing loads
until first use (resume paths stay cheap), and shared resources
(MfccExtractor's per-clip cache, TrackCropReader) are single instances
handed to every consumer.
"""

from orchestrator.pipeline_config import PipelineConfig
from services.downloader.youtube_downloader import YouTubeDownloader
from services.face_tracker.video_processor import VideoProcessor
from services.mouth_exporter.segment_exporter import SegmentExporter
from services.quality_indexer.speaker_identifier import SpeakerIdentifier
from services.segmenter.diarization import SpeakerDiarizer
from services.segmenter.transcription import WhisperTranscriber
from services.segmenter.vad_splitter import VADSplitter
from services.speaker_detector.active_speaker import TalkNetASD
from services.speaker_detector.audio_features import MfccExtractor
from services.speaker_detector.speaker_selector import SpeakerSelector
from services.speaker_detector.sync_verifier import SyncNetVerifier
from services.speaker_detector.track_crops import TrackCropReader


class ServiceContainer:
    """Lazy factory + cache for every service the pipeline uses."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._downloader = None
        self._splitter = None
        self._transcriber = None
        self._diarizer = None
        self._video_processor = None
        self._mfcc_extractor = None
        self._crop_reader = None
        self._speaker_selector = None
        self._syncnet = None
        self._exporter = None
        self._speaker_identifier = None

    def invalidate_downloader(self) -> None:
        """Drop the cached downloader (CLI cookie overrides re-create it)."""
        self._downloader = None

    def release_segmentation_models(self) -> None:
        """Free Whisper + diarization VRAM once segmentation is done."""
        if self._transcriber is not None:
            self._transcriber.unload()
        if self._diarizer is not None:
            self._diarizer.unload()

    @property
    def downloader(self) -> YouTubeDownloader:
        if self._downloader is None:
            self._downloader = YouTubeDownloader(
                output_dir=self.config.raw_dir,
                format_string=self.config.download_format,
                rate_limit=self.config.rate_limit,
                max_retries=self.config.max_retries,
                fragment_retries=self.config.download_fragment_retries,
                sleep_interval=self.config.download_sleep_interval,
                cookies_file=self.config.cookies_file,
                cookies_from_browser=self.config.cookies_from_browser,
            )
        return self._downloader

    @property
    def splitter(self) -> VADSplitter:
        if self._splitter is None:
            self._splitter = VADSplitter(
                vad_threshold=self.config.vad_threshold,
                min_speech_duration_ms=self.config.min_speech_duration_ms,
                min_silence_duration_ms=self.config.min_silence_duration_ms,
                window_size_samples=self.config.vad_window_size_samples,
                speech_pad_ms=self.config.speech_pad_ms,
                split_threshold=self.config.split_threshold,
                min_clip_duration=self.config.min_clip_duration,
                max_clip_duration=self.config.max_clip_duration,
                sample_rate=self.config.audio_sample_rate,
                seek_start_padding=self.config.seek_start_padding,
                clip_crf=self.config.clip_crf,
                clip_preset=self.config.clip_preset,
            )
        return self._splitter

    @property
    def transcriber(self) -> WhisperTranscriber:
        if self._transcriber is None:
            self._transcriber = WhisperTranscriber(
                model_name=self.config.whisper_model,
                language=self.config.language,
                device=self.config.device,
                compute_type=self.config.compute_type,
                batch_size=self.config.whisper_batch_size,
            )
        return self._transcriber

    @property
    def video_processor(self) -> VideoProcessor:
        if self._video_processor is None:
            self._video_processor = VideoProcessor(
                detection_confidence=self.config.detection_confidence,
                detection_nms_threshold=self.config.detection_nms_threshold,
                tracking_iou=self.config.tracking_iou,
                max_track_age=self.config.tracking_max_age,
                min_track_hits=self.config.tracking_min_hits,
                kalman_process_noise=self.config.kalman_process_noise,
                kalman_measurement_noise=self.config.kalman_measurement_noise,
            )
        return self._video_processor

    @property
    def mfcc_extractor(self) -> MfccExtractor:
        """Shared per-clip MFCC cache — TalkNet and SyncNet slice from the
        same computed matrix (computed once per clip, freed on the next)."""
        if self._mfcc_extractor is None:
            self._mfcc_extractor = MfccExtractor()
        return self._mfcc_extractor

    @property
    def crop_reader(self) -> TrackCropReader:
        if self._crop_reader is None:
            self._crop_reader = TrackCropReader()
        return self._crop_reader

    @property
    def speaker_selector(self) -> SpeakerSelector:
        if self._speaker_selector is None:
            talknet = TalkNetASD(
                model_path=self.config.models_dir / "talknet_asd.pth",
                device=self.config.device,
                num_frames_context=self.config.asd_num_frames_context,
                allow_fallback=self.config.asd_allow_fallback,
            )
            self._speaker_selector = SpeakerSelector(
                talknet=talknet,
                crop_reader=self.crop_reader,
                mfcc_extractor=self.mfcc_extractor,
                min_track_speech_overlap=self.config.min_track_speech_overlap,
                max_candidate_tracks=self.config.max_candidate_tracks,
            )
        return self._speaker_selector

    @property
    def syncnet(self) -> SyncNetVerifier:
        if self._syncnet is None:
            self._syncnet = SyncNetVerifier(
                model_path=self.config.models_dir / "syncnet_v2.pth",
                device=self.config.device,
                max_offset_frames=self.config.syncnet_max_offset_frames,
                allow_fallback=self.config.syncnet_allow_fallback,
                max_analysis_seconds=self.config.syncnet_max_analysis_seconds,
                mfcc_extractor=self.mfcc_extractor,
                crop_reader=self.crop_reader,
            )
        return self._syncnet

    @property
    def exporter(self) -> SegmentExporter:
        if self._exporter is None:
            self._exporter = SegmentExporter(
                processed_dir=self.config.processed_dir,
                output_fps=self.config.output_fps,
                output_size=self.config.output_size,
                mouth_size=self.config.mouth_size,
                crop_margin=self.config.crop_margin,
                include_audio=self.config.include_audio,
                video_codec=self.config.video_codec,
                video_crf=self.config.video_crf,
                gaussian_smoothing_sigma=self.config.gaussian_smoothing_sigma,
                mouth_gaussian_smoothing_sigma=self.config.mouth_gaussian_smoothing_sigma,
                minimum_crop_size_pixels=self.config.minimum_crop_size_pixels,
                mouth_width_multiplier=self.config.mouth_width_multiplier,
                mouth_min_half_size_pixels=self.config.mouth_min_half_size_pixels,
                video_preset=self.config.video_preset,
                use_ffmpeg_pipe=self.config.use_ffmpeg_pipe,
                mouth_roi_method=self.config.mouth_roi_method,
                mouth_roi_min_confidence=self.config.mouth_roi_min_confidence,
                mouth_smoothing_type=self.config.mouth_smoothing_type,
                one_euro_min_cutoff=self.config.one_euro_min_cutoff,
                one_euro_beta=self.config.one_euro_beta,
                mouth_roi_width_multiplier=self.config.mouth_roi_width_multiplier,
                mouth_grayscale=self.config.mouth_grayscale,
                mouth_align_roll=self.config.mouth_align_roll,
                mouth_fail_rate_fallback=self.config.mouth_fail_rate_fallback,
                save_segment_audio=self.config.save_segment_audio,
                identity_samples=(
                    self.config.speaker_identity_samples
                    if self.config.speaker_identity_enabled else 0
                ),
            )
        return self._exporter

    @property
    def diarizer(self) -> SpeakerDiarizer:
        if self._diarizer is None:
            self._diarizer = SpeakerDiarizer(
                hf_token=self.config.diarization_hf_token,
                device=self.config.device,
                min_speakers=self.config.diarization_min_speakers,
                max_speakers=self.config.diarization_max_speakers,
            )
        return self._diarizer

    @property
    def speaker_identifier(self) -> SpeakerIdentifier:
        if self._speaker_identifier is None:
            self._speaker_identifier = SpeakerIdentifier(
                models_dir=self.config.models_dir,
                catalog_db_path=self.config.catalog_db_path,
                cluster_eps=self.config.speaker_identity_cluster_eps,
                cross_video_enabled=self.config.speaker_identity_cross_video,
                cross_video_similarity=self.config.speaker_identity_similarity,
            )
        return self._speaker_identifier
