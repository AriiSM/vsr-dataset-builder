"""
Update `suggested_environment` for specific video_ids in proposals.csv.

Usage:
    python backend/tools/set_env_proposals.py md_007:studio md_008:studio md_009:indoor ...

Valid env values: studio, indoor, outdoor, mixed, unknown.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent.parent
PROPOSALS_CSV = ROOT / "data" / "catalog" / "auto_tag_workdir" / "proposals.csv"
VALID_ENV = {"studio", "indoor", "outdoor", "mixed", "unknown"}


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: set_env_proposals.py vid:env [vid:env ...]", file=sys.stderr)
        return 1
    if not PROPOSALS_CSV.exists():
        print(f"ERROR: {PROPOSALS_CSV} not found.", file=sys.stderr)
        return 1

    pairs: dict[str, str] = {}
    for arg in sys.argv[1:]:
        if ":" not in arg:
            print(f"Skipping malformed arg: {arg!r}", file=sys.stderr)
            continue
        vid, env = arg.split(":", 1)
        env = env.strip().lower()
        if env not in VALID_ENV:
            print(f"Skipping invalid env {env!r} for {vid}", file=sys.stderr)
            continue
        pairs[vid.strip()] = env

    if not pairs:
        print("Nothing to update.", file=sys.stderr)
        return 1

    df = pd.read_csv(PROPOSALS_CSV)
    if "suggested_environment" not in df.columns:
        df["suggested_environment"] = ""
    df["suggested_environment"] = df["suggested_environment"].fillna("").astype(str)

    updated = 0
    not_found: list[str] = []
    for vid, env in pairs.items():
        mask = df["video_id"] == vid
        if not mask.any():
            not_found.append(vid)
            continue
        df.loc[mask, "suggested_environment"] = env
        updated += 1

    df.to_csv(PROPOSALS_CSV, index=False, encoding="utf-8")
    print(f"Updated {updated} rows.")
    if not_found:
        print(f"Not found in proposals.csv: {not_found}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
