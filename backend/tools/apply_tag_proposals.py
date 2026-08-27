"""
Merge proposals.csv (produced by auto_tag_conditions.py) into videos_master.csv.

Safety rules:
  * Creates a timestamped backup of videos_master.csv before writing.
  * Only touches `environment` and `background_noise` columns.
  * Never overwrites an existing (non-null) value — only fills blanks.
  * Validates every proposed value against the schema enum; rejects invalid rows.

Run:
    python backend/tools/apply_tag_proposals.py            # dry-run, prints what would change
    python backend/tools/apply_tag_proposals.py --apply    # actually writes the CSV
    python backend/tools/apply_tag_proposals.py --apply --overwrite   # also overwrite existing values
"""
from __future__ import annotations

import argparse
import io
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent
CSV_PATH = ROOT / "data" / "catalog" / "videos_master.csv"
PROPOSALS_CSV = ROOT / "data" / "catalog" / "auto_tag_workdir" / "proposals.csv"

VALID_ENV = {"indoor", "outdoor", "studio", "mixed", "unknown"}
VALID_NOISE = {"none", "low", "moderate", "high"}


def backup(csv: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = csv.with_suffix(f".csv.bak.{ts}.auto-tag-apply")
    shutil.copy2(csv, bak)
    return bak


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true",
                   help="Actually write videos_master.csv. Without this flag, runs a dry-run.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing non-null env/noise values too. Default is fill-blanks-only.")
    args = p.parse_args()

    if not CSV_PATH.exists():
        print(f"ERROR: not found: {CSV_PATH}", file=sys.stderr)
        return 1
    if not PROPOSALS_CSV.exists():
        print(f"ERROR: not found: {PROPOSALS_CSV}", file=sys.stderr)
        print("Run scripts/auto_tag_conditions.py first.", file=sys.stderr)
        return 1

    master = pd.read_csv(CSV_PATH)
    proposals = pd.read_csv(PROPOSALS_CSV)

    # Build per-video proposal dict, only for rows that have something usable.
    by_id: dict[str, dict[str, str]] = {}
    invalid_rows: list[tuple[str, str]] = []
    for _, row in proposals.iterrows():
        vid = str(row["video_id"])
        env = str(row.get("suggested_environment") or "").strip().lower()
        noise = str(row.get("suggested_noise") or "").strip().lower()
        entry: dict[str, str] = {}
        if env:
            if env not in VALID_ENV:
                invalid_rows.append((vid, f"env={env!r}"))
            else:
                entry["environment"] = env
        if noise:
            if noise not in VALID_NOISE:
                invalid_rows.append((vid, f"noise={noise!r}"))
            else:
                entry["background_noise"] = noise
        if entry:
            by_id[vid] = entry

    if invalid_rows:
        print("WARNING: invalid values rejected:")
        for vid, msg in invalid_rows:
            print(f"  {vid}: {msg}")
        print()

    # Plan changes.
    changes: list[tuple[str, str, str, str]] = []  # (vid, column, before, after)
    for idx, row in master.iterrows():
        vid = str(row["video_id"])
        if vid not in by_id:
            continue
        for col, new_val in by_id[vid].items():
            cur = row.get(col)
            cur_str = str(cur) if pd.notna(cur) else ""
            if cur_str and not args.overwrite:
                continue  # skip — already filled
            if cur_str == new_val:
                continue  # no-op
            changes.append((vid, col, cur_str, new_val))

    if not changes:
        print("Nothing to change. (Either no proposals, or all targets already filled.)")
        return 0

    # Summary
    env_changes = [c for c in changes if c[1] == "environment"]
    noise_changes = [c for c in changes if c[1] == "background_noise"]
    print(f"Planned changes: {len(changes)} cells across {len(set(c[0] for c in changes))} videos")
    print(f"  environment:       {len(env_changes)}")
    print(f"  background_noise:  {len(noise_changes)}")
    print()
    if env_changes:
        env_dist = pd.Series([c[3] for c in env_changes]).value_counts()
        print("environment distribution:")
        for k, v in env_dist.items():
            print(f"  {k:10s} {v}")
        print()
    if noise_changes:
        noise_dist = pd.Series([c[3] for c in noise_changes]).value_counts()
        print("background_noise distribution:")
        for k, v in noise_dist.items():
            print(f"  {k:10s} {v}")
        print()

    if not args.apply:
        print("DRY-RUN: pass --apply to write changes.")
        # Show the first 15 changes for visibility.
        print("\nFirst 15 changes (vid, col, before -> after):")
        for vid, col, before, after in changes[:15]:
            before_disp = before or "(empty)"
            print(f"  {vid:10s} {col:18s} {before_disp:10s} -> {after}")
        return 0

    # Apply.
    bak = backup(CSV_PATH)
    print(f"Backup: {bak.name}")

    for vid, col, _, new_val in changes:
        master.loc[master["video_id"] == vid, col] = new_val

    master.to_csv(CSV_PATH, index=False, encoding="utf-8")
    print(f"Wrote {CSV_PATH.name}")
    print()
    print("Final coverage:")
    print(f"  environment tagged:      {int(master['environment'].notna().sum())} / {len(master)}")
    print(f"  background_noise tagged: {int(master['background_noise'].notna().sum())} / {len(master)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
