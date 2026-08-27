"""
cleanup_orphan_segments.py — remove segments_index rows whose files are missing.

A segment is "orphan" when it appears in data/metadata/segments_index.csv
but the corresponding face_crop video and/or annotation file is gone.
Likely causes:
  - User rejected the segment in the UI; the handler deleted the files
    but the CSV cleanup failed (Excel locked, race, etc.).
  - Pipeline was interrupted between writing the CSV row and writing the
    annotation/video.
  - Files deleted manually outside the UI.

This script lists every orphan with a probable cause, then optionally
removes the row from segments_index.csv. Idempotent — running it twice
is safe; the second run finds zero orphans.

Usage:
    python backend/tools/cleanup_orphan_segments.py --dry-run      # report only
    python backend/tools/cleanup_orphan_segments.py                # clean
    python backend/tools/cleanup_orphan_segments.py --details      # full table
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path[:0] = [str(_PROJECT_ROOT / "backend"), str(_PROJECT_ROOT / "backend" / "shared")]

import pandas as pd  # noqa: E402


def _classify(seg_id: str, video_id: str, has_video: bool, has_anno: bool,
              review_entry: dict) -> str:
    """Pick a one-word cause label for an orphan row."""
    if review_entry and review_entry.get("status") == "rejected":
        return "rejected"
    if not has_video and not has_anno:
        return "both_missing"
    if not has_anno and has_video:
        return "anno_missing"
    if has_anno and not has_video:
        return "video_missing"
    return "unknown"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data", default="./data",
                   help="Project data directory.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report only — do not modify segments_index.csv.")
    p.add_argument("--apply", action="store_true",
                   help="Alias for 'no --dry-run' — provided for consistency with the "
                        "drop_failed_and_renumber.py / reset_manually_reviewed.py scripts. "
                        "Without --dry-run the script already writes; --apply is a no-op "
                        "but accepted so muscle memory works.")
    p.add_argument("--details", action="store_true",
                   help="Print every orphan row, not just summary counts.")
    p.add_argument("--also-clean-review-status", action="store_true",
                   help="Drop matching keys from review_status.json too. Off by default — "
                        "rejected entries are sometimes useful to keep as audit trail.")
    args = p.parse_args()

    data_dir = Path(args.data)
    metadata_dir = data_dir / "metadata"
    processed_dir = data_dir / "processed"
    annotations_dir = data_dir / "annotations"
    seg_csv = metadata_dir / "segments_index.csv"
    review_path = metadata_dir / "review_status.json"

    if not seg_csv.exists():
        print(f"ERROR: {seg_csv} not found", file=sys.stderr)
        return 1

    df = pd.read_csv(seg_csv)
    review = (
        json.loads(review_path.read_text(encoding="utf-8"))
        if review_path.exists() else {}
    )

    orphans: list[dict] = []
    for _, row in df.iterrows():
        seg_id = str(row["segment_id"])
        video_id = str(row["video_id"])
        face_video = processed_dir / video_id / "face_crop" / f"{seg_id}.mp4"
        anno_file = data_dir / "processed" / video_id / "text" / f"{seg_id}.txt"
        if not anno_file.exists():
            anno_file = annotations_dir / video_id / f"{seg_id}.txt"

        has_video = face_video.exists()
        has_anno = anno_file.exists()

        if has_video and has_anno:
            continue  # healthy

        rs_entry = review.get(seg_id, {}) if isinstance(review, dict) else {}
        cause = _classify(seg_id, video_id, has_video, has_anno, rs_entry)
        orphans.append({
            "segment_id": seg_id,
            "video_id": video_id,
            "cause": cause,
            "has_video": has_video,
            "has_anno": has_anno,
            "review_status": rs_entry.get("status") if isinstance(rs_entry, dict) else None,
        })

    n_total = len(df)
    n_orphan = len(orphans)

    print(f"Scan: {n_total} segments in CSV")
    print(f"  Healthy (both files present): {n_total - n_orphan}")
    print(f"  Orphan (missing file(s)):     {n_orphan}")

    if not orphans:
        print("\nNothing to clean.")
        return 0

    causes = Counter(o["cause"] for o in orphans)
    print("\nOrphan causes:")
    for cause, count in causes.most_common():
        print(f"  {cause:18s} {count:>4}")

    if args.details:
        print("\nDetails:")
        print(f"  {'segment_id':36s} {'video':10s} {'cause':14s} {'video?':6s} {'anno?':5s} {'review'}")
        for o in orphans[:200]:
            print(f"  {o['segment_id'][:36]:36s} {o['video_id'][:10]:10s} "
                  f"{o['cause']:14s} {str(o['has_video']):6s} {str(o['has_anno']):5s} "
                  f"{o['review_status'] or '-'}")
        if len(orphans) > 200:
            print(f"  ... + {len(orphans) - 200} more (re-run with output redirected to a file to see all)")

    if args.dry_run:
        print("\n(--dry-run: no files written)")
        return 0

    # Apply: drop orphan rows from segments_index.
    orphan_ids = {o["segment_id"] for o in orphans}
    cleaned = df[~df["segment_id"].astype(str).isin(orphan_ids)]

    # Sanity check #1: did we drop exactly the right count?
    expected_after = n_total - n_orphan
    if len(cleaned) != expected_after:
        print(f"\nABORT: expected {expected_after} rows after drop, got {len(cleaned)}. "
              "Refusing to write — your CSV is untouched.", file=sys.stderr)
        return 2

    # Safety: timestamped backup of the live CSV before any write.
    # Lives next to the original (data/metadata/) so you can restore by:
    #   copy segments_index.csv.bak.YYYYMMDD_HHMMSS segments_index.csv
    from datetime import datetime
    import shutil
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = seg_csv.with_name(f"{seg_csv.stem}.csv.bak.{timestamp}")
    try:
        shutil.copy2(seg_csv, backup)
    except Exception as e:
        print(f"\nABORT: backup failed ({e}). Refusing to write.", file=sys.stderr)
        return 1
    print(f"\nBackup: {backup.name}")

    # Atomic write: write to .tmp first, then replace. If we crash mid-write
    # the live file stays as it was (and the backup is also intact).
    tmp = seg_csv.with_suffix(".csv.tmp")
    try:
        cleaned.to_csv(tmp, index=False)
    except PermissionError:
        print(f"\nERROR: cannot write to {tmp.name} — disk full or read-only?",
              file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nERROR writing temp file: {e}", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return 1

    try:
        tmp.replace(seg_csv)  # atomic on Windows when same drive
    except PermissionError:
        tmp.unlink(missing_ok=True)
        print(f"\nERROR: {seg_csv.name} is open in another program — close it and retry. "
              f"No data lost; your file is unchanged.", file=sys.stderr)
        return 1

    # Sanity check #2: re-read what we just wrote and confirm the row count.
    try:
        verify = pd.read_csv(seg_csv)
        if len(verify) != expected_after:
            print(f"\nWARNING: post-write verification mismatch "
                  f"({len(verify)} rows vs expected {expected_after}). "
                  f"Restore from {backup.name} if needed.", file=sys.stderr)
            return 1
    except Exception as e:
        print(f"\nWARNING: post-write verification failed: {e}. "
              f"File written; backup at {backup.name}.", file=sys.stderr)

    print(f"Removed {n_orphan} orphan row(s) from {seg_csv.name}")
    print(f"Verified: {len(cleaned)} rows now in CSV (was {n_total}).")

    # Optional: also strip review_status.json keys.
    if args.also_clean_review_status and review_path.exists():
        before = len(review)
        for sid in orphan_ids:
            review.pop(sid, None)
        review_path.write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Removed {before - len(review)} key(s) from {review_path.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
