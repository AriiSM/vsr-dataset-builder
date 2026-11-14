"""
Speakers registry — speakers with curator-edited metadata (name / gender /
age / accent) and live aggregated stats.

Storage v2: rows live in the `speakers` table of data/catalog/dataset.db;
the aggregates (num_videos, num_segments, total_duration_s, avg_asd,
avg_wer) are the `speaker_stats` VIEW — always correct, computed from
segments, never recomputed by hand. speakers_registry.csv is an on-demand
export (backend/tools/export_catalog.py).

The public function API is unchanged from the CSV era; `metadata_dir` is
the catalog dir (data/catalog) holding dataset.db.

Conventions
-----------
- One speaker per row, keyed by `speaker_id`.
- Default id for single-speaker videos: `{video_id}_spk0`.
- Curator/auto-editable fields: speaker_name, gender, gender_confidence,
  age_estimate, age_std, age_group, accent_region, identity_match.
"""

from pathlib import Path
from typing import Optional

from loguru import logger

from vsr_shared.catalog_db import CatalogDatabase

# One connection per catalog dir, reused across calls in the same process.
_DB_CACHE: dict = {}


def _db(metadata_dir: Path) -> CatalogDatabase:
    key = str(Path(metadata_dir).resolve())
    if key not in _DB_CACHE:
        _DB_CACHE[key] = CatalogDatabase(Path(metadata_dir) / "dataset.db")
    return _DB_CACHE[key]


def get_speaker(metadata_dir: Path, speaker_id: str) -> Optional[dict]:
    return _db(metadata_dir).speakers.get(speaker_id)


def upsert_speaker(metadata_dir: Path, speaker_id: str, fields: dict) -> None:
    """Create or update a speaker row (editable fields only — aggregate
    values are a VIEW and silently ignored here, as before)."""
    _db(metadata_dir).speakers.upsert(speaker_id, fields)


def ensure_speaker_exists(metadata_dir: Path, speaker_id: str) -> None:
    """Idempotent: create a blank row for `speaker_id` if not registered."""
    _db(metadata_dir).speakers.ensure_exists(speaker_id)


def load_speakers_with_stats(metadata_dir: Path) -> list:
    """Every speaker joined with the live aggregates view (export shape)."""
    return _db(metadata_dir).speakers.all_with_stats()


# --------------------------------------------------------------- compat
# DataFrame-shaped API kept for the Flask frontend + legacy tools until the
# FastAPI step reads the DB directly. Backed by the same speakers table.

_EXPORT_COLUMNS = [
    "speaker_id", "speaker_name", "gender", "age_group", "age_estimate",
    "age_std", "gender_confidence", "identity_match", "accent_region",
    "num_videos", "num_segments", "total_duration_s", "avg_asd", "avg_wer",
]


def load_speakers(metadata_dir: Path):
    """Registry as a DataFrame (speakers ⋈ speaker_stats), CSV-era shape."""
    import pandas as pd

    rows = load_speakers_with_stats(metadata_dir)
    df = pd.DataFrame(rows, columns=_EXPORT_COLUMNS + ["centroid"]) \
        if rows else pd.DataFrame(columns=_EXPORT_COLUMNS)
    return df[_EXPORT_COLUMNS] if not df.empty else df


def save_speakers(metadata_dir: Path, df) -> None:
    """Persist a DataFrame's EDITABLE fields back into the speakers table.

    Aggregate columns are ignored (they're a view). Rows are upserted by
    speaker_id — deletions must go through the DB directly.
    """
    import pandas as pd

    for _, row in df.iterrows():
        speaker_id = str(row.get("speaker_id") or "").strip()
        if not speaker_id:
            continue
        fields = {
            k: (None if pd.isna(v) else v)
            for k, v in row.to_dict().items()
            if k != "speaker_id"
        }
        upsert_speaker(metadata_dir, speaker_id, fields)


def recompute_aggregates(metadata_dir: Path) -> int:
    """Storage v2: aggregates ARE the speaker_stats view — nothing to
    recompute. Kept for call-site compatibility; returns the speaker count
    so callers that log it keep working."""
    count = len(_db(metadata_dir).speakers.all_with_stats())
    logger.debug(f"Speakers registry: {count} speaker(s), stats via view")
    return count
