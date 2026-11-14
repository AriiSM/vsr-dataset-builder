"""
Voice Activity Detection (Silero VAD)

Finds WHERE speech happens: speech regions + the real pauses between them,
in absolute source-video seconds. The sentence segmenter uses these to drop
Whisper hallucinations (words outside speech) and to know where the natural
breathing points are.

Model loads lazily via torch.hub — resume paths never pay for it.
"""

import torch
import torchaudio
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from loguru import logger


@dataclass
class SpeechSegment:
    """A detected speech segment."""
    start: float  # Start time in seconds
    end: float    # End time in seconds
    
    @property
    def duration(self) -> float:
        return self.end - self.start



class SileroVAD:
    """
    Voice Activity Detection using Silero VAD.

    Silero VAD is fast and accurate, suitable for processing large amounts of audio.

    All parameters are required — caller must read them from config.yaml.
    """

    def __init__(
        self,
        threshold: float,
        min_speech_duration_ms: int,
        min_silence_duration_ms: int,
        window_size_samples: int,
        speech_pad_ms: int,
        sample_rate: int,
    ):
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.window_size_samples = window_size_samples
        self.speech_pad_ms = speech_pad_ms
        self.sample_rate = sample_rate

        # Model loads lazily: resume paths (existing clips + manifest) never
        # run VAD, so they must not pay the torch.hub load either.
        self.model = None
        self.get_speech_timestamps = None

    def _ensure_model_loaded(self):
        """Load Silero VAD on first use.

        trust_repo=True keeps newer torch versions from stopping at an
        interactive confirmation prompt — fatal for unattended batch runs.
        """
        if self.model is not None:
            return
        try:
            self.model, vad_utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
                trust_repo=True,
            )
        except Exception as e:
            raise RuntimeError(
                "Could not load Silero VAD via torch.hub. The first use needs "
                "internet access to fetch github.com/snakers4/silero-vad "
                "(cached afterwards under ~/.cache/torch/hub). "
                f"Original error: {e}"
            ) from e
        self.get_speech_timestamps = vad_utils[0]
        logger.info("Silero VAD model loaded")
    
    def _load_audio(self, audio_path: Path) -> torch.Tensor:
        """Load and preprocess audio file using soundfile (avoids torchaudio/torchcodec issues)."""
        import soundfile as sf
        wav_np, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(wav_np.T)  # (channels, samples)

        # Convert to mono if stereo
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            wav = resampler(wav)

        return wav.squeeze()
    
    def detect_speech(self, audio_path: Path) -> List[SpeechSegment]:
        """Detect speech regions in an audio file.

        Returns SpeechSegment list with start/end in SECONDS (the only unit
        the pipeline uses anywhere).
        """
        self._ensure_model_loaded()
        wav = self._load_audio(audio_path)

        speech_timestamps = self.get_speech_timestamps(
            wav,
            self.model,
            threshold=self.threshold,
            sampling_rate=self.sample_rate,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            window_size_samples=self.window_size_samples,
            speech_pad_ms=self.speech_pad_ms,
            return_seconds=True,
        )

        segments = [
            SpeechSegment(start=ts['start'], end=ts['end'])
            for ts in speech_timestamps
        ]
        logger.info(f"Detected {len(segments)} speech segments")
        return segments

    def extract_audio_from_video(
        self,
        video_path: Path,
        output_path: Optional[Path] = None
    ) -> Path:
        """Extract audio from video file using ffmpeg."""
        import subprocess
        
        if output_path is None:
            output_path = video_path.with_suffix('.wav')
        
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            ffmpeg_exe = 'ffmpeg'

        cmd = [
            ffmpeg_exe, '-y',
            '-i', str(video_path),
            '-vn',  # No video
            '-acodec', 'pcm_s16le',
            '-ar', str(self.sample_rate),
            '-ac', '1',  # Mono
            str(output_path)
        ]

        subprocess.run(cmd, capture_output=True, check=True)
        logger.info(f"Extracted audio to: {output_path}")
        
        return output_path


