"""
DataFrame builders — DB tables in the CSV-era shapes the aggregations
(ported from the Flask app) expect.

The heavy stats logic was written against pandas frames of
segments_index.csv / videos_master.csv; rather than rewriting hundreds of
lines of proven aggregation, we feed it identical frames built straight
from the catalog (MongoDB). Region is joined from videos (segments only carry
video_id), exactly like the old CSV-merge did.
"""

import pandas as pd

from vsr_shared.catalog_db import CatalogDatabase


def segments_frame(db: CatalogDatabase) -> pd.DataFrame:
    """Stats-relevant columns only — skips blobs/paths and raw text variants."""
    fields = ["segment_id", "video_id", "review_status", "duration", "num_words",
              "num_chars", "text", "asd_score", "syncnet_conf", "whisper_conf",
              "whisper_conf_min", "wer", "quality_tier", "speaker_id",
              "transcript_edited", "trimmed", "mouth_landmark_fail_rate",
              "transcript_consensus", "wer_dialectat_mediu_vs_large"]
    rows = db.segments.query(fields=fields, with_region=True)
    return pd.DataFrame([{k: r.get(k) for k in fields + ["region"]} for r in rows])


def videos_frame(db: CatalogDatabase) -> pd.DataFrame:
    return pd.DataFrame(db.videos.all())


def speakers_frame(db: CatalogDatabase) -> pd.DataFrame:
    rows = db.speakers.all_with_stats()
    for row in rows:
        row.pop("centroid", None)   # BLOB — never serialised to the UI
    return pd.DataFrame(rows)
