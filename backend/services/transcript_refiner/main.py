"""
Transcript Refiner Service

Second-opinion pass over ALREADY EXPORTED segments using Whisper large-v3
(int8 — fits the 4 GB GPU, or runs on CPU). The pipeline itself stays on
`medium` for speed; this service runs afterwards, standalone:

1. For each segment in segments_index.csv, extract the audio from the
   exported face_crop video and transcribe it with large-v3.
2. Compare against the pipeline transcript (WER medium-vs-large).
3. Write additive columns:
     text_largev3            — the large-v3 transcript (normalized)
     wer_medium_vs_large     — disagreement between the two models
     needs_review            — True when disagreement exceeds the threshold
   High-disagreement segments are exactly the ones worth human review time.

Usage (from the repo root):
    python backend/services/transcript_refiner/main.py                # all pending
    python backend/services/transcript_refiner/main.py --video-id md_001
    python backend/services/transcript_refiner/main.py --limit 50     # smoke test
    python backend/services/transcript_refiner/main.py --force        # redo all

Requires: faster-whisper (pip install faster-whisper). Nothing else from
the GPU stack is loaded — safe to run alongside nothing on 4 GB VRAM.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
_PROJECT_ROOT = _BACKEND_DIR.parent
sys.path[:0] = [str(_BACKEND_DIR), str(_BACKEND_DIR / "shared")]

import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402

from vsr_shared.text_normalization import clean_word, normalize_diacritics  # noqa: E402
from vsr_shared.wer_utils import compute_wer  # noqa: E402


DEFAULT_MODEL = "large-v3"
DEFAULT_COMPUTE_TYPE = "int8"
DEFAULT_REVIEW_THRESHOLD = 0.15

# Additive columns this service owns in segments_index.csv
COLUMN_TEXT = "text_largev3"
COLUMN_WER = "wer_medium_vs_large"
COLUMN_REVIEW = "needs_review"


def _find_ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def _extract_audio(video_path: Path, wav_path: Path) -> bool:
    command = [
        _find_ffmpeg(), "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(wav_path),
    ]
    result = subprocess.run(command, capture_output=True)
    return result.returncode == 0 and wav_path.exists() and wav_path.stat().st_size > 0


class LargeWhisperTranscriber:
    """faster-whisper large-v3 in int8 — loaded once, reused per segment."""

    def __init__(self, model_name: str, device: str, compute_type: str,
                 language: str = "ro"):
        from faster_whisper import WhisperModel

        logger.info(f"Loading faster-whisper {model_name} ({device}, {compute_type})...")
        self._model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self.language = language
        logger.info("Model loaded")

    def transcribe(self, wav_path: Path) -> str:
        segments, _info = self._model.transcribe(
            str(wav_path), language=self.language, beam_size=5,
        )
        words = []
        for segment in segments:
            for raw_word in segment.text.split():
                cleaned = clean_word(raw_word)
                if cleaned:
                    words.append(cleaned)
        return " ".join(words)


def refine(args) -> int:
    index_path = _PROJECT_ROOT / "data" / "metadata" / "segments_index.csv"
    if not index_path.exists():
        logger.error(f"segments_index.csv not found at {index_path}")
        return 1

    frame = pd.read_csv(index_path)
    for column, default in ((COLUMN_TEXT, ""), (COLUMN_WER, None), (COLUMN_REVIEW, None)):
        if column not in frame.columns:
            frame[column] = default

    rows = frame
    if args.video_id:
        rows = rows[rows["video_id"].isin(args.video_id)]
    if not args.force:
        rows = rows[rows[COLUMN_TEXT].isna() | (rows[COLUMN_TEXT] == "")]
    if args.limit:
        rows = rows.head(args.limit)

    if rows.empty:
        logger.info("Nothing to refine (use --force to redo).")
        return 0

    logger.info(f"Refining {len(rows)} segments with {args.model}...")
    transcriber = LargeWhisperTranscriber(args.model, args.device, args.compute_type)

    refined = flagged = failed = 0
    with tempfile.TemporaryDirectory() as temp_dir:
        wav_path = Path(temp_dir) / "segment.wav"
        for index, row in rows.iterrows():
            video_path = _PROJECT_ROOT / str(row["video_path"])
            if not video_path.exists():
                video_path = Path(str(row["video_path"]))
            if not video_path.exists():
                logger.warning(f"Missing video for {row['segment_id']} — skipping")
                failed += 1
                continue

            if not _extract_audio(video_path, wav_path):
                logger.warning(f"Audio extraction failed for {row['segment_id']}")
                failed += 1
                continue

            try:
                large_text = transcriber.transcribe(wav_path)
            except Exception as e:
                logger.warning(f"Transcription failed for {row['segment_id']}: {e}")
                failed += 1
                continue

            pipeline_text = normalize_diacritics(str(row.get("text") or ""))
            wer, _ = compute_wer(pipeline_text, large_text)
            needs_review = bool(wer is not None and wer > args.review_threshold)

            frame.at[index, COLUMN_TEXT] = large_text
            frame.at[index, COLUMN_WER] = round(wer, 4) if wer is not None else None
            frame.at[index, COLUMN_REVIEW] = needs_review

            refined += 1
            flagged += int(needs_review)
            if refined % 25 == 0:
                frame.to_csv(index_path, index=False)  # checkpoint progress
                logger.info(f"  {refined}/{len(rows)} done ({flagged} flagged)")

    frame.to_csv(index_path, index=False)
    logger.info(
        f"Refinement complete: {refined} refined, {flagged} flagged for review "
        f"(WER > {args.review_threshold}), {failed} failed"
    )
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Second-opinion transcription with Whisper large-v3 (int8)."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto",
                        help="auto | cuda | cpu (int8 fits a 4GB GPU)")
    parser.add_argument("--compute-type", default=DEFAULT_COMPUTE_TYPE)
    parser.add_argument("--review-threshold", type=float,
                        default=DEFAULT_REVIEW_THRESHOLD,
                        help="WER above which a segment is flagged needs_review")
    parser.add_argument("--video-id", nargs="*", help="restrict to these videos")
    parser.add_argument("--limit", type=int, help="max segments (smoke test)")
    parser.add_argument("--force", action="store_true",
                        help="redo segments that already have text_largev3")
    args = parser.parse_args()

    if args.device == "auto":
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"

    sys.exit(refine(args))


if __name__ == "__main__":
    main()
