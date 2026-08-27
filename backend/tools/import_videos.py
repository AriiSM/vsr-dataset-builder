"""
Import the curator-edited videos_master.csv into the catalog database.

One-time migration to storage v2 + ongoing seeding: run it again whenever
you add new rows to the CSV — existing DB rows keep their pipeline-written
fields (status, stats); only the curator columns are refreshed.

Usage (from the repo root):
    python backend/tools/import_videos.py                       # default paths
    python backend/tools/import_videos.py --csv path/to.csv --catalog data/catalog
"""

import argparse
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _BACKEND_DIR.parent
sys.path[:0] = [str(_BACKEND_DIR), str(_BACKEND_DIR / "shared")]

import pandas as pd  # noqa: E402

from vsr_shared.catalog_db import CatalogDatabase  # noqa: E402

# Curator-owned columns — refreshed from the CSV on every import.
_CURATOR_COLUMNS = [
    "youtube_url", "source", "source_channel", "license", "region", "title",
    "duration_seconds", "speaker_id", "num_speakers", "gender", "age_group",
    "environment", "background_noise",
]
# Pipeline-owned columns — imported only when the DB row is brand new.
_PIPELINE_COLUMNS = [
    "status", "processed_date", "total_segments", "total_duration_extracted",
    "avg_asd_score", "avg_syncnet_conf", "error_message",
]


def import_videos(csv_path: Path, catalog_dir: Path) -> int:
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return 0

    df = pd.read_csv(csv_path)
    db = CatalogDatabase(catalog_dir / "dataset.db")

    imported = 0
    for _, row in df.iterrows():
        video_id = str(row.get("video_id") or "").strip()
        if not video_id:
            continue
        existing = db.videos.get(video_id)
        db.videos.ensure_exists(video_id)

        fields = {}
        for col in _CURATOR_COLUMNS:
            if col in df.columns and not pd.isna(row[col]):
                fields[col] = row[col]
        if existing is None:
            for col in _PIPELINE_COLUMNS:
                if col in df.columns and not pd.isna(row[col]):
                    fields[col] = row[col]
        db.videos.update_fields(video_id, fields)
        imported += 1

    db.close()
    return imported


def main():
    parser = argparse.ArgumentParser(
        description="Import videos_master.csv into data/catalog/dataset.db")
    parser.add_argument("--csv", type=Path,
                        default=_PROJECT_ROOT / "data" / "catalog" / "videos_master.csv")
    parser.add_argument("--catalog", type=Path,
                        default=_PROJECT_ROOT / "data" / "catalog")
    args = parser.parse_args()

    csv_path = args.csv
    if not csv_path.exists():
        legacy = _PROJECT_ROOT / "data" / "metadata" / "videos_master.csv"
        if legacy.exists():
            csv_path = legacy
    count = import_videos(csv_path, args.catalog)
    print(f"Imported/refreshed {count} video row(s) from {csv_path}")


if __name__ == "__main__":
    main()
