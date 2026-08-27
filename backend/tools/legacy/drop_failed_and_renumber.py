"""
drop_failed_and_renumber.py — remove status=failed videos from the master
CSV and (optionally) renumber the survivors so video IDs are dense again.

What "failed" means here:
  Rows in videos_master.csv whose `status` column is exactly "failed".
  Typical cause: bulk-import couldn't download the video (CC-check fail,
  bot detection, video deleted, network glitch). Those rows have NO
  on-disk artefacts (no .mp4 in data/raw/, no segments, no annotations).
  So "deletion" is just dropping the row from the CSV.

What "renumber" means:
  After dropping failed rows, surviving videos may have gaps in their
  numbering (e.g. md_001, md_002, md_005, md_007). Renumbering fills the
  gaps: md_001, md_002, md_003, md_004 (in original sort order).
  This requires renaming on-disk artefacts AND updating every CSV/JSON
  that references the old ID.

Renaming covers:
  - data/raw/{old_id}.mp4
  - data/processed/{old_id}/  (incl. face_crop/*.mp4 + mouth_crop/*.mp4 — segment_id is renamed too)
  - data/annotations/{old_id}/  (incl. *.txt — segment_id renamed)
  - data/clips/{old_id}/        (incl. clips.json + .checkpoint.json + leftover .mp4/.wav)
  - segments_index.csv          (video_id, segment_id, video_path, annotation_path)
  - speakers_registry.csv       (speaker_id starts with video_id)
  - review_status.json          (keys are segment_ids that start with video_id)

Two-phase rename for collision safety:
  Phase 1: every dir/file matching an old_id-being-renamed is moved to
           a temp "__renumbering__" suffix.
  Phase 2: temp names are renamed to their final new_id.
  This means no rename ever lands on a path that still belongs to another
  in-progress rename.

Defaults:
  --dry-run is ON unless you pass --apply.
  Both CSV and JSON files are backed up under data/metadata/<file>.bak.<ts>
  before any change. Atomic CSV writes via tmp + rename.

Usage:
    python backend/tools/drop_failed_and_renumber.py                    # dry-run with renumber plan
    python backend/tools/drop_failed_and_renumber.py --no-renumber      # only drop failed rows, keep gaps
    python backend/tools/drop_failed_and_renumber.py --apply            # actually do it
    python backend/tools/drop_failed_and_renumber.py --apply --no-renumber
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path[:0] = [str(_PROJECT_ROOT / "backend"), str(_PROJECT_ROOT / "backend" / "shared")]

import pandas as pd  # noqa: E402


# Pipeline IDs: "<prefix>_<digits>" — e.g. md_001, ro_042.
_ID_RE = re.compile(r"^(?P<prefix>[A-Za-z]+)_(?P<num>\d+)$")
_TEMP_SUFFIX = ".__renumbering__"


def _parse_id(video_id: str) -> Optional[Tuple[str, int, int]]:
    """Return (prefix, number, width) for a renumber-eligible ID; None otherwise."""
    m = _ID_RE.match(video_id)
    if not m:
        return None
    return m.group("prefix"), int(m.group("num")), len(m.group("num"))


def _make_id(prefix: str, num: int, width: int) -> str:
    return f"{prefix}_{num:0{width}d}"


def _safe_backup(path: Path, timestamp: str) -> Optional[Path]:
    if not path.exists():
        return None
    backup = path.with_name(f"{path.name}.bak.{timestamp}")
    shutil.copy2(path, backup)
    return backup


def _atomic_write_csv(df: pd.DataFrame, target: Path) -> None:
    tmp = target.with_suffix(target.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(target)


def _atomic_write_json(data: dict, target: Path) -> None:
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(target)


def _rename_paths_in_dir(directory: Path, old_id: str, new_id: str) -> int:
    """Rename every file inside `directory` whose name contains old_id.

    Returns the number of files actually renamed. Does nothing if the
    directory itself doesn't exist. Recurses one level deep (face_crop,
    mouth_crop are subdirs of data/processed/{vid}/).
    """
    if not directory.exists():
        return 0
    n = 0
    # File-rename pass — top-level + one level of subdirs.
    for entry in list(directory.iterdir()):
        if entry.is_file():
            if old_id in entry.name:
                new_name = entry.name.replace(old_id, new_id)
                target = entry.with_name(new_name)
                if target.exists() and target != entry:
                    print(f"  WARN: rename target exists, skipped: {target}", file=sys.stderr)
                    continue
                entry.rename(target)
                n += 1
        elif entry.is_dir():
            for sub in list(entry.iterdir()):
                if sub.is_file() and old_id in sub.name:
                    new_name = sub.name.replace(old_id, new_id)
                    target = sub.with_name(new_name)
                    if target.exists() and target != sub:
                        print(f"  WARN: rename target exists, skipped: {target}",
                              file=sys.stderr)
                        continue
                    sub.rename(target)
                    n += 1
    return n


def _phase1_rename_to_temp(data_dir: Path, renames: List[Tuple[str, str]]) -> None:
    """Rename every artefact named after old_id to {old_id}__renumbering__."""
    for old_id, _new_id in renames:
        # Raw video
        raw = data_dir / "raw" / f"{old_id}.mp4"
        if raw.exists():
            raw.rename(raw.with_name(raw.stem + _TEMP_SUFFIX + raw.suffix))
        # Subdirs in processed / annotations / clips
        for parent in ("processed", "annotations", "clips"):
            p = data_dir / parent / old_id
            if p.exists():
                p.rename(p.with_name(p.name + _TEMP_SUFFIX))


def _phase2_rename_temp_to_new(data_dir: Path, renames: List[Tuple[str, str]]) -> int:
    """Rename {old_id}__renumbering__ → {new_id} and rewrite filenames inside."""
    n_files = 0
    for old_id, new_id in renames:
        # Raw video
        raw_temp = data_dir / "raw" / f"{old_id}{_TEMP_SUFFIX}.mp4"
        if raw_temp.exists():
            raw_temp.rename(raw_temp.with_name(f"{new_id}.mp4"))
        # Subdirs
        for parent in ("processed", "annotations", "clips"):
            tmp_dir = data_dir / parent / f"{old_id}{_TEMP_SUFFIX}"
            if tmp_dir.exists():
                final_dir = tmp_dir.with_name(new_id)
                tmp_dir.rename(final_dir)
                # Inside this directory, rename any file whose name embeds old_id
                # (segment_id contains video_id, so face_crop/*.mp4, mouth_crop/*.mp4,
                #  *.txt all need updating).
                n_files += _rename_paths_in_dir(final_dir, old_id, new_id)
    return n_files


def _drop_segments_for_videos(seg_csv: Path, video_ids: set) -> int:
    """Remove every row in segments_index.csv whose video_id is in `video_ids`."""
    if not seg_csv.exists() or not video_ids:
        return 0
    df = pd.read_csv(seg_csv)
    if df.empty or "video_id" not in df.columns:
        return 0
    before = len(df)
    keep_mask = ~df["video_id"].astype(str).isin(video_ids)
    df = df[keep_mask]
    _atomic_write_csv(df, seg_csv)
    return before - len(df)


def _drop_speakers_for_videos(speakers_csv: Path, video_ids: set) -> int:
    """Remove every speaker row whose speaker_id starts with `{vid}_`."""
    if not speakers_csv.exists() or not video_ids:
        return 0
    df = pd.read_csv(speakers_csv)
    if df.empty or "speaker_id" not in df.columns:
        return 0
    before = len(df)
    sid_str = df["speaker_id"].fillna("").astype(str)
    drop_mask = pd.Series(False, index=df.index)
    for vid in video_ids:
        drop_mask |= sid_str.str.startswith(vid + "_")
    df = df[~drop_mask]
    _atomic_write_csv(df, speakers_csv)
    return before - len(df)


def _drop_review_keys_for_videos(review_path: Path, video_ids: set) -> int:
    """Remove every review_status.json key whose segment_id starts with `{vid}_`."""
    if not review_path.exists() or not video_ids:
        return 0
    try:
        data = json.loads(review_path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    if not isinstance(data, dict):
        return 0
    new_data = {
        seg_id: entry
        for seg_id, entry in data.items()
        if not any(seg_id.startswith(vid + "_") for vid in video_ids)
    }
    n_dropped = len(data) - len(new_data)
    if n_dropped:
        _atomic_write_json(new_data, review_path)
    return n_dropped


def _cleanup_video_artefacts(data_dir: Path, video_ids: set) -> Tuple[int, int]:
    """Delete on-disk artefacts (raw mp4 + processed/annotations/clips dirs)
    for each video_id. Returns (n_files_deleted, n_dirs_deleted).
    """
    n_files = 0
    n_dirs = 0
    for vid in video_ids:
        raw = data_dir / "raw" / f"{vid}.mp4"
        if raw.exists():
            try:
                raw.unlink()
                n_files += 1
            except OSError as e:
                print(f"  WARN: could not delete {raw}: {e}", file=sys.stderr)
        for sub in ("processed", "annotations", "clips"):
            dir_path = data_dir / sub / vid
            if dir_path.exists():
                try:
                    shutil.rmtree(dir_path)
                    n_dirs += 1
                except OSError as e:
                    print(f"  WARN: could not delete {dir_path}: {e}", file=sys.stderr)
    return n_files, n_dirs


def _rewrite_segments_csv(seg_csv: Path, mapping: Dict[str, str]) -> int:
    """Replace old_id with new_id in segments_index.csv (multiple columns)."""
    if not seg_csv.exists():
        return 0
    df = pd.read_csv(seg_csv)
    if df.empty:
        return 0

    n_rows_changed = 0
    for col in ("video_id", "segment_id", "video_path", "annotation_path"):
        if col not in df.columns:
            continue
        col_str = df[col].fillna("").astype(str)
        new_col = col_str.copy()
        for old_id, new_id in mapping.items():
            if old_id == new_id:
                continue
            mask = new_col.str.contains(old_id, regex=False, na=False)
            n_rows_changed = max(n_rows_changed, int(mask.sum()))
            # Replace text — safe because video_ids are unique-prefix.
            new_col = new_col.where(~mask, new_col.str.replace(old_id, new_id, regex=False))
        df[col] = new_col

    _atomic_write_csv(df, seg_csv)
    return n_rows_changed


def _rewrite_speakers_csv(speakers_csv: Path, mapping: Dict[str, str]) -> int:
    if not speakers_csv.exists():
        return 0
    df = pd.read_csv(speakers_csv)
    if df.empty or "speaker_id" not in df.columns:
        return 0
    n = 0
    new_col = df["speaker_id"].fillna("").astype(str).copy()
    for old_id, new_id in mapping.items():
        if old_id == new_id:
            continue
        # Speaker IDs start with "{video_id}_spk..." — safe prefix replace.
        prefix_old = old_id + "_"
        mask = new_col.str.startswith(prefix_old)
        n += int(mask.sum())
        new_col = new_col.where(~mask, new_col.str.replace(prefix_old, new_id + "_", regex=False))
    df["speaker_id"] = new_col
    _atomic_write_csv(df, speakers_csv)
    return n


def _rewrite_review_json(review_path: Path, mapping: Dict[str, str]) -> int:
    if not review_path.exists():
        return 0
    try:
        data = json.loads(review_path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    if not isinstance(data, dict):
        return 0
    new_data: dict = {}
    n = 0
    for seg_id, entry in data.items():
        replaced = seg_id
        for old_id, new_id in mapping.items():
            if old_id == new_id:
                continue
            # Segment IDs start with "{video_id}_clip_..."
            if replaced.startswith(old_id + "_"):
                replaced = new_id + replaced[len(old_id):]
                n += 1
                break
        new_data[replaced] = entry
    _atomic_write_json(new_data, review_path)
    return n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data", default="./data", help="Project data directory.")
    p.add_argument("--apply", action="store_true",
                   help="Actually do it. Without this flag the script only reports.")
    p.add_argument("--no-renumber", action="store_true",
                   help="Only drop failed rows; leave the survivors' IDs as-is (gaps OK).")
    p.add_argument("--drop-unverified", action="store_true",
                   help="Also drop rows whose license == 'unverified'. Unlike `failed` "
                        "rows, these may have on-disk artefacts (downloaded with "
                        "--no-cc-check), so the script also deletes data/raw/{id}.mp4, "
                        "data/processed/{id}/, data/annotations/{id}/, data/clips/{id}/, "
                        "and removes references from segments_index.csv, "
                        "speakers_registry.csv, review_status.json.")
    args = p.parse_args()

    data_dir = Path(args.data)
    metadata_dir = data_dir / "metadata"
    master_csv = metadata_dir / "videos_master.csv"
    seg_csv = metadata_dir / "segments_index.csv"
    speakers_csv = metadata_dir / "speakers_registry.csv"
    review_path = metadata_dir / "review_status.json"

    if not master_csv.exists():
        print(f"ERROR: {master_csv} not found", file=sys.stderr)
        return 1

    df = pd.read_csv(master_csv)
    if "status" not in df.columns or "video_id" not in df.columns:
        print("ERROR: videos_master.csv is missing required columns.", file=sys.stderr)
        return 1
    df = df.astype(object)

    failed_mask = df["status"].astype(str) == "failed"
    failed_ids = df.loc[failed_mask, "video_id"].astype(str).tolist()

    unverified_ids: List[str] = []
    if args.drop_unverified and "license" in df.columns:
        unverified_mask = (
            (df["license"].fillna("").astype(str).str.strip() == "unverified")
            & ~failed_mask  # don't double-count
        )
        unverified_ids = df.loc[unverified_mask, "video_id"].astype(str).tolist()

    drop_set = set(failed_ids) | set(unverified_ids)
    survivors = df.loc[~df["video_id"].astype(str).isin(drop_set)].copy()

    print(f"Master CSV: {len(df)} total")
    print(f"  failed:     {len(failed_ids)}  → drop (no on-disk artefacts)")
    if args.drop_unverified:
        print(f"  unverified: {len(unverified_ids)}  → drop + cleanup artefacts")
    print(f"  survivors:  {len(survivors)}")

    if failed_ids:
        print("\nFailed video_ids that will be DROPPED:")
        for vid in failed_ids:
            err_col = "error_message" if "error_message" in df.columns else None
            err = df.loc[df["video_id"].astype(str) == vid, err_col].iloc[0] if err_col else ""
            print(f"  {vid:14s}  {str(err)[:80]}")
    else:
        print("\nNothing marked as failed.")

    if unverified_ids:
        print("\nUnverified-license video_ids that will be DROPPED + ARTEFACTS DELETED:")
        for vid in unverified_ids:
            status = df.loc[df["video_id"].astype(str) == vid, "status"].iloc[0]
            url_col = "youtube_url" if "youtube_url" in df.columns else None
            url = df.loc[df["video_id"].astype(str) == vid, url_col].iloc[0] if url_col else ""
            print(f"  {vid:14s}  status={status:12s}  {str(url)[:60]}")
    elif args.drop_unverified:
        print("\nNothing marked as license=unverified.")

    # ── Build the renumber plan ────────────────────────────────────────
    renames: List[Tuple[str, str]] = []
    if not args.no_renumber and not survivors.empty:
        # Group survivors by prefix; renumber within each group.
        # Sort by old (prefix, num) so original order is preserved.
        parsed: List[Tuple[str, int, int, str]] = []  # (prefix, num, width, original_id)
        unparsed: List[str] = []
        for vid in survivors["video_id"].astype(str).tolist():
            p_ = _parse_id(vid)
            if p_ is None:
                unparsed.append(vid)
            else:
                parsed.append((*p_, vid))
        if unparsed:
            print(f"  Note: {len(unparsed)} ID(s) don't match the prefix_NNN pattern "
                  f"and will NOT be renumbered: {unparsed[:5]}")
        # Group by (prefix, width) so md_001 and ro_001 stay in their own series.
        by_group: Dict[Tuple[str, int], List[Tuple[int, str]]] = {}
        for prefix, num, width, original in parsed:
            by_group.setdefault((prefix, width), []).append((num, original))
        for (prefix, width), items in by_group.items():
            items.sort(key=lambda x: x[0])
            for new_num, (_old_num, original) in enumerate(items, start=1):
                target = _make_id(prefix, new_num, width)
                if target != original:
                    renames.append((original, target))

    if renames:
        print(f"\nRenumber plan: {len(renames)} ID(s) will be renamed:")
        for old_id, new_id in renames[:20]:
            print(f"  {old_id} → {new_id}")
        if len(renames) > 20:
            print(f"  ... + {len(renames) - 20} more")
    else:
        if args.no_renumber:
            print("\nRenumber: SKIPPED (--no-renumber)")
        else:
            print("\nRenumber: nothing to renumber (no gaps).")

    if not failed_ids and not unverified_ids and not renames:
        print("\nNothing to do.")
        return 0

    if not args.apply:
        print("\n(--dry-run by default — pass --apply to actually do it)")
        return 0

    # ── APPLY ─────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\nTimestamp for this operation: {timestamp}")

    # Backups for everything we may touch.
    for tgt in (master_csv, seg_csv, speakers_csv, review_path):
        bak = _safe_backup(tgt, timestamp)
        if bak is not None:
            print(f"  backup: {bak.name}")

    # 0. Cleanup artefacts for unverified videos BEFORE renumbering, so the
    #    rename step doesn't try to move files we're about to delete.
    if unverified_ids:
        unv_set = set(unverified_ids)
        n_files, n_dirs = _cleanup_video_artefacts(data_dir, unv_set)
        print(f"\nUnverified cleanup: removed {n_files} raw mp4(s) + {n_dirs} directory tree(s)")

        n_seg_dropped = _drop_segments_for_videos(seg_csv, unv_set)
        n_spk_dropped = _drop_speakers_for_videos(speakers_csv, unv_set)
        n_rev_dropped = _drop_review_keys_for_videos(review_path, unv_set)
        print(f"  segments_index.csv:    -{n_seg_dropped} row(s)")
        print(f"  speakers_registry.csv: -{n_spk_dropped} row(s)")
        print(f"  review_status.json:    -{n_rev_dropped} key(s)")

    # 1. Two-phase rename of on-disk artefacts.
    if renames:
        print("\nPhase 1/2: moving artefacts to temporary names...")
        _phase1_rename_to_temp(data_dir, renames)
        print("Phase 2/2: assigning final new IDs...")
        n_files = _phase2_rename_temp_to_new(data_dir, renames)
        print(f"  Renamed {n_files} file(s) inside directories.")

    # 2. Rewrite the metadata files.
    if renames:
        mapping = dict(renames)
        n_seg = _rewrite_segments_csv(seg_csv, mapping)
        n_spk = _rewrite_speakers_csv(speakers_csv, mapping)
        n_rev = _rewrite_review_json(review_path, mapping)
        print(f"  segments_index.csv  rows touched: ~{n_seg}")
        print(f"  speakers_registry.csv rows touched: {n_spk}")
        print(f"  review_status.json keys touched: {n_rev}")

    # 3. Build the new master CSV: drop failed + apply rename mapping.
    survivors = survivors.copy()
    if renames:
        rename_map = dict(renames)
        survivors["video_id"] = survivors["video_id"].astype(str).map(
            lambda v: rename_map.get(v, v)
        )
        # Re-sort by new id so rows look tidy in spreadsheets.
        survivors = survivors.sort_values(by="video_id", kind="stable").reset_index(drop=True)
    _atomic_write_csv(survivors, master_csv)
    n_dropped_total = len(failed_ids) + len(unverified_ids)
    print(f"\nvideos_master.csv: {len(df)} → {len(survivors)} row(s) "
          f"(dropped {n_dropped_total}: {len(failed_ids)} failed + "
          f"{len(unverified_ids)} unverified, renamed {len(renames)}).")

    print("\nDone. Backups remain on disk — delete them once you confirm everything is OK:")
    for fn in (master_csv.name, seg_csv.name, speakers_csv.name, review_path.name):
        print(f"  data/metadata/{fn}.bak.{timestamp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
