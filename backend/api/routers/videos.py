"""
Videos route — the master registry with LIVE per-video stats.

Flask recomputed total_segments / duration / avg_asd from segments_index
on every request (the master CSV went stale after rejects); here the same
freshness comes from the video_stats VIEW — always correct by definition.
Rejected segments are excluded, preserving the old semantics.
"""

from fastapi import APIRouter, Depends

from api.db import get_db
from vsr_shared.catalog_db import CatalogDatabase

router = APIRouter(prefix="/api", tags=["videos"])


@router.get("/videos")
def videos(db: CatalogDatabase = Depends(get_db)):
    rows = db.videos.all()
    if not rows:
        return {"videos": []}

    # Fresh per-video stats over the surviving (non-rejected) segments.
    fresh = {
        r["video_id"]: r for r in db.connection.execute(
            "SELECT video_id, COUNT(*) AS n, ROUND(SUM(duration), 2) AS dur,"
            " ROUND(AVG(asd_score), 3) AS asd"
            " FROM segments WHERE COALESCE(review_status, '') != 'rejected'"
            " GROUP BY video_id")
    }

    records = []
    for row in rows:
        stats = fresh.get(row["video_id"])
        row = dict(row)
        row["total_segments"] = stats["n"] if stats else 0
        row["total_duration_extracted"] = stats["dur"] if stats else 0
        row["avg_asd_score"] = stats["asd"] if stats else 0
        # Flask stringified every cell (CSV heritage) — the React table
        # expects strings; empty stays empty.
        records.append({
            k: ("" if v is None or v == "" else str(v))
            for k, v in row.items()
        })
    return {"videos": records}
