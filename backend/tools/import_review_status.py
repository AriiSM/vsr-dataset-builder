"""
One-time migration: review_status.json → segments review columns.

The Flask frontend stored curation verdicts in a JSON sidecar; storage v2
keeps them on the segment rows (review_status / reviewed_at /
transcript_edited / trimmed). Run once after upgrading; idempotent — a
segment whose DB row already carries a verdict is left alone.

Usage (from the repo root):
    python backend/tools/import_review_status.py
    python backend/tools/import_review_status.py --json path/to/review_status.json
"""

import argparse
import json
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _BACKEND_DIR.parent
sys.path[:0] = [str(_BACKEND_DIR), str(_BACKEND_DIR / "shared")]

from vsr_shared.catalog_db import CatalogDatabase  # noqa: E402


def import_review(json_path: Path, catalog_dir: Path) -> int:
    if not json_path.exists():
        print(f"Nothing to import — {json_path} not found.")
        return 0
    try:
        review = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"Cannot read {json_path}: {e}")
        return 0

    db = CatalogDatabase(catalog_dir / "dataset.db")
    imported = skipped = missing = 0
    with db.connection:
        for segment_id, info in review.items():
            if not isinstance(info, dict):
                continue
            row = db.connection.execute(
                "SELECT review_status FROM segments WHERE segment_id = ?",
                (segment_id,)).fetchone()
            if row is None:
                missing += 1
                continue
            if (row["review_status"] or "") != "":
                skipped += 1          # DB verdict wins — idempotent
                continue
            status = info.get("status", "")
            if status not in ("approved", "rejected"):
                status = ""
            db.connection.execute(
                "UPDATE segments SET review_status = ?,"
                " transcript_edited = ?, trimmed = ? WHERE segment_id = ?",
                (status,
                 1 if info.get("transcript_edited") else 0,
                 1 if info.get("trimmed") else 0,
                 segment_id))
            imported += 1
    db.close()
    print(f"Imported {imported} verdict(s); {skipped} already in DB;"
          f" {missing} segment(s) not in catalog.")
    return imported


def main():
    parser = argparse.ArgumentParser(
        description="Import review_status.json verdicts into dataset.db")
    parser.add_argument("--json", type=Path,
                        default=_PROJECT_ROOT / "data" / "catalog" / "review_status.json")
    parser.add_argument("--catalog", type=Path,
                        default=_PROJECT_ROOT / "data" / "catalog")
    args = parser.parse_args()

    json_path = args.json
    if not json_path.exists():
        legacy = _PROJECT_ROOT / "data" / "metadata" / "review_status.json"
        if legacy.exists():
            json_path = legacy
    import_review(json_path, args.catalog)


if __name__ == "__main__":
    main()
