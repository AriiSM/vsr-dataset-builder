"""
Speaker Diarization (pyannote, called directly)

WHO speaks WHEN: anonymous per-video voice labels (SPEAKER_00, SPEAKER_01…)
stamped onto transcribed words, so the sentence segmenter can cut at speaker
turns. The voice↔face identity link is established later, per segment, by
TalkNet (lip-audio sync) + the per-video consensus vote in quality_indexer.

Gated model: free Hugging Face account + accept the pyannote terms once +
token in config (segmentation.diarization.hf_token). Optional — with
diarization disabled the segmenter falls back to punctuation + pauses only.
"""

import torch
from pathlib import Path
from typing import Optional

from loguru import logger


class SpeakerDiarizer:
    """WHO speaks WHEN — pyannote diarization, loaded directly.

    Produces anonymous per-video voice labels (SPEAKER_00, SPEAKER_01…) and
    stamps them onto already-transcribed words, so the sentence segmenter can
    cut at speaker turns. The identity link voice↔face is established later,
    per segment, by TalkNet (lip-audio sync) + the per-video consensus vote.

    Requires a gated Hugging Face model: free account, accept the pyannote
    terms once, put the token in config (segmentation.diarization.hf_token).
    Loads lazily; unload() frees the GPU like WhisperTranscriber does.
    """

    def __init__(
        self,
        hf_token: str,
        device: str,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ):
        self.hf_token = hf_token
        self.device = device
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self._pipeline = None

    def _ensure_pipeline_loaded(self):
        if self._pipeline is not None:
            return
        if not self.hf_token:
            raise RuntimeError(
                "Diarization is enabled but segmentation.diarization.hf_token "
                "is empty. Create a free Hugging Face account, accept the "
                "terms at huggingface.co/pyannote/speaker-diarization-3.1, "
                "generate an access token and put it in config.yaml."
            )
        try:
            # Direct pyannote — NOT the whisperx DiarizationPipeline wrapper:
            # that wrapper hardcodes use_auth_token, which newer
            # huggingface_hub renamed to token, and breaks at load time.
            import inspect

            from pyannote.audio import Pipeline as PyannotePipeline

            params = inspect.signature(
                PyannotePipeline.from_pretrained).parameters
            token_kw = "token" if "token" in params else "use_auth_token"
            pipeline = PyannotePipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                **{token_kw: self.hf_token},
            )
            if pipeline is None:
                raise RuntimeError(
                    "from_pretrained returned None — usually the HF token "
                    "has not accepted the model terms"
                )
            self._pipeline = pipeline.to(torch.device(self.device))
        except Exception as e:
            raise RuntimeError(
                f"Could not load the pyannote diarization pipeline: {e}. "
                "Check that pyannote.audio is installed and the HF token has "
                "accepted the model terms."
            ) from e
        logger.info("Diarization pipeline loaded (pyannote)")

    def assign_speaker_labels(self, words: list, audio_path: Path) -> int:
        """Run diarization and stamp each TimedWord with its voice label.

        Words falling into no diarization turn (or exactly between turns)
        keep speaker=None — the segmenter treats None as "unknown, do not
        force a boundary".

        Returns the number of distinct voices found.
        """
        self._ensure_pipeline_loaded()

        logger.info("Diarizing (who speaks when)...")
        annotation = self._pipeline(
            str(audio_path),
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
        )
        speaker_turns = [
            (turn.start, turn.end, label)
            for turn, _, label in annotation.itertracks(yield_label=True)
        ]

        labeled = self.label_words_by_overlap(words, speaker_turns)
        voices = {w.speaker for w in words if w.speaker}
        logger.info(
            f"Diarization: {len(voices)} voices, "
            f"{labeled}/{len(words)} words labeled"
        )
        return len(voices)

    @staticmethod
    def label_words_by_overlap(words: list, speaker_turns: list) -> int:
        """Assign each word the speaker whose turn overlaps it the most.

        Pure logic (unit-testable without pyannote). speaker_turns is a list
        of (start, end, label) in the same absolute seconds as the words.
        Returns how many words received a label.
        """
        labeled_count = 0
        for word in words:
            best_overlap = 0.0
            best_label = None
            for turn_start, turn_end, label in speaker_turns:
                overlap = min(word.end, turn_end) - max(word.start, turn_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_label = label
            if best_label is not None:
                word.speaker = best_label
                labeled_count += 1
        return labeled_count

    def unload(self):
        """Release the diarization models (GPU memory back to the pipeline)."""
        import gc

        if self._pipeline is not None:
            self._pipeline = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Diarization pipeline unloaded")
