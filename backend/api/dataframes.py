"""
DataFrame builders — DB tables in the CSV-era shapes the aggregations
(ported from the Flask app) expect.

The heavy stats logic was written against pandas frames of
segments_index.csv / videos_master.csv; rather than rewriting hundreds of
lines of proven aggregation, we feed it identical frames built straight
from dataset.db. Region arrives via SQL join (segments only carry
video_id), exactly like the old CSV-merge did.
"""

import pandas as pd

from vsr_shared.catalog_db import CatalogDatabase


def segments_frame(db: CatalogDatabase) -> pd.DataFrame:
    """All segments + per-video region, CSV-shaped (syncnet_conf etc.)."""
    rows = db.connection.execute(
        "SELECT s.*, COALESCE(NULLIF(v.region, ''), 'UNKNOWN') AS region"
        " FROM segments s LEFT JOIN videos v USING (video_id)"
        " ORDER BY s.video_id, s.segment_id"
    ).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def videos_frame(db: CatalogDatabase) -> pd.DataFrame:
    return pd.DataFrame(db.videos.all())


def speakers_frame(db: CatalogDatabase) -> pd.DataFrame:
    rows = db.speakers.all_with_stats()
    for row in rows:
        row.pop("centroid", None)   # BLOB — never serialised to the UI
    return pd.DataFrame(rows)
