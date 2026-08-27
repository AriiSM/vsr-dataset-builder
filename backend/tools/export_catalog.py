"""
Export the catalog database to CSVs (the storage-v1 shapes), on demand.

Storage v2 made dataset.db the source of truth; whenever you want CSVs —
for Excel, for sharing, or to regenerate a broken mirror — run this.

Usage (from the repo root):
    python backend/tools/export_catalog.py                    # all three CSVs
    python backend/tools/export_catalog.py --only segments
    python backend/tools/export_catalog.py --out data/catalog/exports
"""

import argparse
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _BACKEND_DIR.parent
sys.path[:0] = [str(_BACKEND_DIR), str(_BACKEND_DIR / "shared")]

import pandas as pd  # noqa: E402

from vsr_shared.catalog_db import CatalogDatabase  # noqa: E402
from vsr_shared.excel_schema import (  # noqa: E402
    SEGMENTS_INDEX_SCHEMA,
    SPEAKERS_REGISTRY_SCHEMA,
    VIDEOS_MASTER_SCHEMA,
)


def _project_paths(project_root: Path, video_id: str, segment_id: str):
    base = project_root / "data" / "processed" / video_id
    return (base / "face_crop" / f"{segment_id}.mp4",
            base / "text" / f"{segment_id}.txt")


def export_videos(db: CatalogDatabase, out_path: Path) -> int:
    rows = db.videos.all()
    columns = list(VIDEOS_MASTER_SCHEMA.keys())
    df = pd.DataFrame([{c: r.get(c, "") for c in columns} for r in rows],
                      columns=columns)
    df.to_csv(out_path, index=False)
    return len(df)


def export_segments(db: CatalogDatabase, out_path: Path,
                    project_root: Path) -> int:
    columns = list(SEGMENTS_INDEX_SCHEMA.keys())
    rows = []
    for r in db.segments.all():
        video_path, annotation_path = _project_paths(
            project_root, r["video_id"], r["segment_id"])
        rows.append({
            **{c: r.get(c, "") for c in columns},
            "av_speaker_mismatch": (
                "" if r.get("av_speaker_mismatch") is None
                else bool(r["av_speaker_mismatch"])
            ),
            "video_path": str(video_path),
            "annotation_path": str(annotation_path),
        })
    pd.DataFrame(rows, columns=columns).to_csv(out_path, index=False)
    return len(rows)


def export_speakers(db: CatalogDatabase, out_path: Path) -> int:
    columns = list(SPEAKERS_REGISTRY_SCHEMA.keys())
    rows = [{c: r.get(c, "") for c in columns}
            for r in db.speakers.all_with_stats()]
    pd.DataFrame(rows, columns=columns).to_csv(out_path, index=False)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Export dataset.db to CSVs (storage-v1 shapes)")
    parser.add_argument("--catalog", type=Path,
                        default=_PROJECT_ROOT / "data" / "catalog")
    parser.add_argument("--out", type=Path, default=None,
                        help="output dir (default: the catalog dir itself, "
                             "so the mirrors used by the frontend refresh)")
    parser.add_argument("--only", choices=["videos", "segments", "speakers"],
                        help="export just one CSV")
    args = parser.parse_args()

    out_dir = args.out or args.catalog
    out_dir.mkdir(parents=True, exist_ok=True)
    db = CatalogDatabase(args.catalog / "dataset.db")

    if args.only in (None, "videos"):
        n = export_videos(db, out_dir / "videos_master.csv")
        print(f"videos_master.csv       {n} rows")
    if args.only in (None, "segments"):
        n = export_segments(db, out_dir / "segments_index.csv", _PROJECT_ROOT)
        print(f"segments_index.csv      {n} rows")
    if args.only in (None, "speakers"):
        n = export_speakers(db, out_dir / "speakers_registry.csv")
        print(f"speakers_registry.csv   {n} rows")
    db.close()


if __name__ == "__main__":
    main()
