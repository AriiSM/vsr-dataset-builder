"""
Centroid store — cross-video speaker identity persistence.

Storage v2: centroids live in the `speakers.centroid` BLOB column of
data/catalog/dataset.db (this file used to write index.json + .npy files —
exactly the migration the class was isolated for; nothing else changed).

A cluster re-matched in a later video updates its stored centroid with a
running average, so the identity stays stable as more videos contribute
evidence. The connection opens lazily so importing the module costs nothing.
"""

from pathlib import Path
from typing import Dict

import numpy as np
from loguru import logger

from services.quality_indexer.identity_records import SpeakerProfile
from vsr_shared.catalog_db import CatalogDatabase


class CentroidStore:
    """Load + persist per-speaker centroids in the catalog database."""

    def __init__(self, catalog_db_path: Path):
        self.catalog_db_path = Path(catalog_db_path)
        self._db = None

    def _speakers(self):
        if self._db is None:
            self._db = CatalogDatabase(self.catalog_db_path)
        return self._db.speakers

    def load_known(self) -> Dict[str, np.ndarray]:
        """All persisted centroids: speaker_id → normalized vector."""
        try:
            return self._speakers().centroids()
        except Exception as e:
            logger.warning(f"Cannot read speaker centroids: {e}")
            return {}

    def persist(self, profiles: Dict[str, SpeakerProfile]) -> None:
        """Store/update centroids so future videos can re-identify."""
        speakers = self._speakers()
        for speaker_id, profile in profiles.items():
            try:
                if profile.identity_match == "auto":
                    stored = speakers.get(speaker_id)
                    if stored and stored.get("centroid"):
                        # Running average with the stored centroid keeps the
                        # identity stable as more videos contribute evidence.
                        previous = np.frombuffer(stored["centroid"], dtype=np.float32)
                        updated = (previous + profile.embedding) / 2.0
                        updated = updated / (np.linalg.norm(updated) or 1.0)
                        speakers.set_centroid(speaker_id, updated)
                        continue
                speakers.set_centroid(speaker_id, profile.embedding)
            except Exception as e:
                logger.warning(f"Centroid persist failed for {speaker_id}: {e}")
