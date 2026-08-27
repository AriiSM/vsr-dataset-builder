"""
rebuild_segments_index.py — recover segments_index.csv rows that exist as
files on disk but went missing from the CSV.

How rows get lost:
  - An earlier cluster/cleanup/reset script crashed mid-write and
    truncated segments_index.
  - A merge / pandas operation dropped rows because of mixed dtypes.
  - A manual edit / external tool corrupted the file.

The face_crop .mp4 + annotation .txt files are usually still on disk
(pipeline writes those first, CSV last). This script walks every video
directory, finds .mp4/.txt pairs that aren't in segments_index, and
appends them with the fields we can reconstruct:

  - segment_id, video_id, video_path, annotation_path: from disk
  - duration: probed from the .mp4 via cv2
  - text, num_words, original_text, wer: parsed from the .txt
  - asd_score: mean of per-word ASD scores in the .txt
  - speaker_id: {video_id}_spk0 (1-speaker default — re-run
                 cluster_speakers.py after to split multi-speaker videos)
  - start_time / end_time / syncnet_conf / whisper_conf / etc.: 0/blank
                 (these were pipeline outputs, lost when the row was lost;
                  cosmetic, do not affect dataset usability)

Defaults are safe: dry-run unless --apply; backup of segments_index.csv
is written before any rewrite.

Usage:
    python backend/tools/rebuild_segments_index.py              # dry-run
    python backend/tools/rebuild_segments_index.py --apply      # actually rebuild
    python backend/tools/rebuild_segments_index.py --apply --video-id md_008 md_023
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path[:0] = [str(_PROJECT_ROOT / "backend"), str(_PROJECT_ROOT / "backend" / "shared")]

import cv2  # noqa: E402
import pandas as pd  # noqa: E402


def _video_duration(path: Path) -> float:
    """Return duration in seconds via cv2 (no ffmpeg call)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0.0
    try:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        return frames / fps if fps > 0 else 0.0
    finally:
        cap.release()


def _parse_annotation(path: Path) -> dict:
    """Pull text + original + word ASD scores from an LRS2 annotation file."""
    out = {"text": "", "original": None, "words": [], "asd_scores": []}
    if not path.exists():
        return out
    in_words = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Original:"):
            out["original"] = line[len("Original:"):].strip()
        elif line.startswith("Text:"):
            out["text"] = line[len("Text:"):].strip()
        elif line.startswith("WORD START END"):
            in_words = True
        elif in_words and line.strip():
            parts = line.split()
            if len(parts) >= 4:
                out["words"].append(parts[0])
                try:
                    out["asd_scores"].append(float(parts[3]))
                except ValueError:
                    pass
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data", default="./data", help="Project data directory.")
    p.add_argument("--apply", action="store_true",
                   help="Actually rebuild rows. Default is dry-run.")
    p.add_argument("--video-id", nargs="+", metavar="ID",
                   help="Restrict to these video IDs (default: every video in master).")
    args = p.parse_args()

    data_dir       = Path(args.data)
    metadata_dir   = data_dir / "metadata"
    processed_dir  = data_dir / "processed"
    annotations_dir = data_dir / "annotations"
    seg_csv = metadata_dir / "segments_index.csv"
    master_csv = metadata_dir / "videos_master.csv"

    if not master_csv.exists():
        print(f"ERROR: {master_csv} not found", file=sys.stderr)
        return 1

    master = pd.read_csv(master_csv)
    seg    = pd.read_csv(seg_csv) if seg_csv.exists() else pd.DataFrame(columns=["segment_id"])

    # Existing segment_ids — anything else we find on disk is a candidate to add back.
    existing = set(seg["segment_id"].astype(str).tolist()) if "segment_id" in seg.columns else set()

    candidate_videos = master["video_id"].astype(str).tolist()
    if args.video_id:
        candidate_videos = [v for v in candidate_videos if v in args.video_id]

    rebuilt: list[dict] = []
    n_videos_with_missing = 0

    for vid in candidate_videos:
        face_dir = processed_dir / vid / "face_crop"
        if not face_dir.exists():
            continue
        anno_dir = data_dir / "processed" / vid / "text"
        if not anno_dir.exists():
            anno_dir = annotations_dir / vid
        missing_for_this_vid = 0
        for mp4 in sorted(face_dir.glob("*.mp4")):
            seg_id = mp4.stem
            if seg_id in existing:
                continue
            txt = anno_dir / f"{seg_id}.txt"
            if not txt.exists():
                # No annotation either → can't reconstruct meaningfully.
                continue
            duration = _video_duration(mp4)
            ann = _parse_annotation(txt)
            words = ann["words"]
            asd = (sum(ann["asd_scores"]) / len(ann["asd_scores"])) if ann["asd_scores"] else 0.0
            text = ann["text"] or ""
            original = ann["original"] if ann["original"] is not None else text
            row = {
                "segment_id":            seg_id,
                "video_id":              vid,
                "start_time":            0.0,   # lost when CSV row was lost
                "end_time":              duration,
                "duration":              round(duration, 3),
                "text":                  text,
                "num_words":             len(words),
                "num_chars":             len(text),
                "speaker_id":            f"{vid}_spk0",  # re-run cluster_speakers to split
                "track_id":              0,
                "asd_score":             round(asd, 3),
                "syncnet_conf":          0.0,
                "whisper_conf":          0.0,
                "original_text":         original,
                "wer":                   0.0,
                "wer_word_count_ref":    len(original.split()) if original else 0,
                "face_bbox":             "",
                "face_visibility_ratio": "",
                "head_pose_avg":         "",
                "video_path":            str(mp4.relative_to(data_dir.parent)) if data_dir.parent in mp4.parents else str(mp4),
                "annotation_path":       str(txt.relative_to(data_dir.parent)) if data_dir.parent in txt.parents else str(txt),
                "split":                 "",
            }
            rebuilt.append(row)
            missing_for_this_vid += 1
        if missing_for_this_vid:
            n_videos_with_missing += 1
            print(f"  {vid:12s}  recovering {missing_for_this_vid} row(s)")

    if not rebuilt:
        print("\nAll disk files already represented in segments_index.csv — nothing to rebuild.")
        return 0

    print(f"\nWill add {len(rebuilt)} row(s) across {n_videos_with_missing} video(s).")

    if not args.apply:
        print("\n(dry-run by default — pass --apply to actually write)")
        return 0

    # Backup before write.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if seg_csv.exists():
        backup = seg_csv.with_name(f"{seg_csv.name}.bak.{timestamp}")
        shutil.copy2(seg_csv, backup)
        print(f"\nBackup: {backup.name}")

    # Build the new DataFrame: union of existing rows + rebuilt rows. Keep
    # column order from current CSV (or from the schema for the rebuilt cols
    # if the CSV is empty).
    new_rows_df = pd.DataFrame(rebuilt)
    if seg.empty:
        out = new_rows_df
    else:
        # Re-order rebuilt rows to match existing column order; missing cols
        # in either side are filled with "" / NaN.
        existing_cols = list(seg.columns)
        for col in existing_cols:
            if col not in new_rows_df.columns:
                new_rows_df[col] = ""
        new_rows_df = new_rows_df[existing_cols + [c for c in new_rows_df.columns
                                                   if c not in existing_cols]]
        out = pd.concat([seg, new_rows_df], ignore_index=True)

    # Atomic write
    tmp = seg_csv.with_suffix(".csv.tmp")
    out.to_csv(tmp, index=False)
    tmp.replace(seg_csv)

    print(f"segments_index.csv: {len(seg)} → {len(out)} row(s) (+{len(rebuilt)}).")
    print("\nRecommended follow-ups:")
    print("  1. python backend/tools/cluster_speakers.py --video-id <rebuilt_ids>  # split 1-speaker default")
    print("  2. python backend/tools/run_pipeline.py sync-excel data/metadata/videos_master.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
