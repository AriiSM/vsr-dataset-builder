"""
Transcription (WhisperX)

Turns audio into words with exact timestamps:
- transcribe():      legacy per-clip path (strategy "vad")
- transcribe_full(): ONE pass over the whole video (strategy "sentence") —
                     every word keeps both its raw form (punctuation intact,
                     for sentence-boundary detection) and its clean transcript
                     form (uppercased, diacritics normalized).
- unload():          frees the GPU for the face/ASD stages.
"""

import os
import torch
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from loguru import logger

from vsr_shared.text_normalization import clean_word, normalize_diacritics
from services.segmenter.sentence_segmenter import TimedWord

# Ensure ffmpeg from imageio_ffmpeg is on PATH so whisperx can find it
try:
    import imageio_ffmpeg
    _ffmpeg_dir = str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
    if _ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
except ImportError:
    pass


@dataclass
class Word:
    """A transcribed word with timing.

    `confidence` is None when WhisperX (or the alignment model) does not return
    a score for this word. Treating this as None instead of defaulting to 1.0
    avoids silently labelling unknown-quality words as "perfect".
    """
    text: str
    start: float
    end: float
    confidence: Optional[float] = None


def _conf_level(confidence_min: float, confidence_mean: float) -> int:
    """LRS2 Conf 1/2/3 derived from BOTH min and mean word confidence.

    A single weak word should drag the segment down even if the rest is great.
    Rule:
      - 3 (high)   : min >= 0.7 AND mean >= 0.9
      - 2 (medium) : min >= 0.5 AND mean >= 0.7
      - 1 (low)    : everything else (including NaN — no confidence signal)
    """
    import math
    if math.isnan(confidence_min) or math.isnan(confidence_mean):
        return 1
    if confidence_min >= 0.7 and confidence_mean >= 0.9:
        return 3
    if confidence_min >= 0.5 and confidence_mean >= 0.7:
        return 2
    return 1


@dataclass
class TranscribedSegment:
    """A transcribed speech segment with word-level detail.

    `confidence` is the MEAN of available word confidences (kept for
    backwards compatibility with existing pipeline code). Use the
    `confidence_min` / `confidence_p25` properties for richer signal.

    All three return NaN if every word has confidence=None — the segment
    has no quality signal at all and should be dropped upstream as
    `unknown_confidence`.
    """
    start: float
    end: float
    text: str
    words: List[Word]
    language: str
    confidence: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def num_words(self) -> int:
        return len(self.words)

    @property
    def confidence_min(self) -> float:
        valid = [w.confidence for w in self.words if w.confidence is not None]
        return float(np.min(valid)) if valid else float("nan")

    @property
    def confidence_p25(self) -> float:
        valid = [w.confidence for w in self.words if w.confidence is not None]
        return float(np.percentile(valid, 25)) if valid else float("nan")



class WhisperTranscriber:
    """
    Speech transcription using Whisper with word-level timestamps.
    
    Uses WhisperX for accurate word-level alignment.
    """
    
    def __init__(
        self,
        model_name: str,
        language: str,
        device: str,
        compute_type: str,
        batch_size: int,
    ):
        self.model_name = model_name
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        
        self._model = None
        self._align_model = None
        self._align_metadata = None
    
    def _load_model(self):
        """Lazy load Whisper model."""
        if self._model is None:
            import whisperx
            
            logger.info(f"Loading Whisper model: {self.model_name}")
            self._model = whisperx.load_model(
                self.model_name,
                self.device,
                compute_type=self.compute_type,
                language=self.language
            )
            logger.info("Whisper model loaded")
    
    def _load_align_model(self):
        """Load alignment model for word timestamps."""
        if self._align_model is None:
            import whisperx
            
            logger.info(f"Loading alignment model for: {self.language}")
            self._align_model, self._align_metadata = whisperx.load_align_model(
                language_code=self.language,
                device=self.device
            )
            logger.info("Alignment model loaded")
    
    def transcribe(
        self,
        audio_path: Path,
    ) -> List[TranscribedSegment]:
        """
        Transcribe audio file with word-level timestamps.

        Args:
            audio_path: Path to audio file

        Returns:
            List of TranscribedSegment objects
        """
        import whisperx
        import soundfile as sf

        self._load_model()

        # Load audio as float32 numpy array (same format whisperx.load_audio produces)
        # Using soundfile avoids a direct ffmpeg subprocess call on Windows
        audio_data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)  # mix down to mono
        audio = audio_data
        
        # Transcribe
        logger.info("Transcribing audio...")
        result = self._model.transcribe(
            audio,
            batch_size=self.batch_size,
            language=self.language
        )
        
        # Align for word timestamps
        self._load_align_model()
        
        logger.info("Aligning words...")
        result = whisperx.align(
            result["segments"],
            self._align_model,
            self._align_metadata,
            audio,
            self.device,
            return_char_alignments=False
        )
        
        # Convert to our format
        transcribed = []
        for seg in result["segments"]:
            words = []
            for word_info in seg.get("words", []):
                # confidence stays None when WhisperX does not return a score
                # (some models / languages) — caller can detect and reject.
                words.append(Word(
                    text=normalize_diacritics(word_info["word"]).strip().upper(),
                    start=word_info.get("start", seg["start"]),
                    end=word_info.get("end", seg["end"]),
                    confidence=word_info.get("score"),
                ))

            if not words:
                continue

            valid_confs = [w.confidence for w in words if w.confidence is not None]
            mean_conf = float(np.mean(valid_confs)) if valid_confs else float("nan")

            transcribed.append(TranscribedSegment(
                start=seg["start"],
                end=seg["end"],
                text=" ".join(w.text for w in words),
                words=words,
                language=self.language,
                confidence=mean_conf,
            ))

        logger.info(f"Transcribed {len(transcribed)} segments")
        return transcribed

    def transcribe_full(self, audio_path: Path) -> List[TimedWord]:
        """Transcribe an ENTIRE video's audio in one pass, returning a flat
        word stream for the sentence segmenter.

        Unlike transcribe(), each word keeps BOTH forms:
        - raw_text: exactly as Whisper produced it, punctuation included —
          the sentence segmenter reads sentence ends (. ? ! …) from here;
        - clean_text: transcript form (diacritics fixed, uppercased,
          punctuation stripped) — what ends up in annotations.

        Timestamps are absolute source-video seconds.
        """
        import whisperx
        import soundfile as sf

        self._load_model()

        audio_data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        logger.info("Transcribing full audio (single pass)...")
        result = self._model.transcribe(
            audio_data,
            batch_size=self.batch_size,
            language=self.language,
        )

        self._load_align_model()
        logger.info("Aligning word timestamps...")
        result = whisperx.align(
            result["segments"],
            self._align_model,
            self._align_metadata,
            audio_data,
            self.device,
            return_char_alignments=False,
        )

        words: List[TimedWord] = []
        for seg in result["segments"]:
            for word_info in seg.get("words", []):
                raw = word_info["word"].strip()
                cleaned = clean_word(raw)
                if not cleaned:
                    continue  # pure punctuation token
                words.append(TimedWord(
                    raw_text=normalize_diacritics(raw),
                    clean_text=cleaned,
                    start=word_info.get("start", seg["start"]),
                    end=word_info.get("end", seg["end"]),
                    confidence=word_info.get("score"),
                ))

        words.sort(key=lambda w: w.start)
        logger.info(f"Full-pass transcription: {len(words)} words")
        return words

    def unload(self):
        """Release Whisper + alignment models (frees ~2-3 GB of VRAM).

        Called after full-video segmentation so RetinaFace / TalkNet get the
        GPU to themselves. The next transcribe call transparently reloads.
        """
        import gc

        released = self._model is not None or self._align_model is not None
        self._model = None
        self._align_model = None
        self._align_metadata = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if released:
            logger.info("Whisper models unloaded — GPU memory released")


