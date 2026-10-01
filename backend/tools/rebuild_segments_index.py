"""
Recompute quality_tier for every segment in the catalog, without reprocessing
any video.  Run this after changing the quality_tiers block in config.yaml.

Usage (from the repo root, inside the worker container or local venv):
    python backend/tools/rebuild_segments_index.py
    python backend/tools/rebuild_segments_index.py --config config/config.yaml
    python backend/tools/rebuild_segments_index.py --db data/catalog/dataset.db
"""

import argparse
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_BACKEND), str(_BACKEND / "shared")]

import yaml  # noqa: E402

from vsr_shared.catalog_db import CatalogDatabase  # noqa: E402
from services.quality_indexer.quality_tiers import compute_quality_tier  # noqa: E402


def _load_tier_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return dict(cfg.get("quality_tiers", {}))


def rebuild(db: CatalogDatabase, tier_config: dict) -> dict:
    rows = db.connection.execute(
        "SELECT segment_id,"
        " whisper_conf, whisper_conf_min,"
        " face_visibility_ratio, mouth_landmark_fail_rate,"
        " asd_method, syncnet_method, mouth_roi_method,"
        " boundary_start_type, boundary_end_type,"
        " duration"
        " FROM segments"
    ).fetchall()

    counts = {"A": 0, "B": 0, "C": 0}
    updates = []
    for r in rows:
        metrics = dict(r)
        tier = compute_quality_tier(metrics, tier_config)
        updates.append((tier, r["segment_id"]))
        counts[tier] += 1

    with db.connection:
        db.connection.executemany(
            "UPDATE segments SET quality_tier = ? WHERE segment_id = ?",
            updates,
        )

    return counts


def main():
    parser = argparse.ArgumentParser(description="Recompute quality tiers from config thresholds.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--db", default="data/catalog/dataset.db")
    args = parser.parse_args()

    config_path = Path(args.config)
    db_path = Path(args.db)

    if not config_path.exists():
        sys.exit(f"Config not found: {config_path}")
    if not db_path.exists():
        sys.exit(f"Database not found: {db_path}")

    tier_config = _load_tier_config(config_path)
    print(f"Thresholds loaded from {config_path}")
    print(f"  Tier A: {tier_config.get('tier_a')}")
    print(f"  Tier B: {tier_config.get('tier_b')}")

    db = CatalogDatabase(str(db_path))
    try:
        counts = rebuild(db, tier_config)
    finally:
        db.close()

    total = sum(counts.values())
    if total == 0:
        print("No segments found.")
        return

    print(f"\nDone — {total} segments updated:")
    for tier in ("A", "B", "C"):
        n = counts[tier]
        pct = 100.0 * n / total
        print(f"  Tier {tier}: {n:>6}  ({pct:.1f}%)")


if __name__ == "__main__":
    main()
