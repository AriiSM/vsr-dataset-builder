"""
Speaker identifier — the quality_indexer service's identity orchestrator.

Resolves WHO is speaking in every exported segment DURING the main pipeline
run — no post-hoc tool needed:

1. Per segment (inline, during export): FaceEmbedder turns the face-crop
   frames sampled by the exporter into ArcFace + gender/age evidence.
2. Per video (end): cluster all segment embeddings (DBSCAN, cosine) — a
   speaker who reappears throughout the video (anchor ↔ field reports ↔
   anchor) gets ONE id, so distinct-speaker statistics come out right.
3. Demographics per CLUSTER (all sampled frames of that person): majority
   gender + median numeric age → far more stable than a few random frames.
4. Optional cross-video re-identification through CentroidStore (the
   speakers.centroid column of dataset.db): a new
   cluster matching a stored centroid (cosine similarity ≥ threshold)
   adopts the existing GLOBAL speaker id. Essential for the MD subset,
   where the same news anchors appear in dozens of videos and would
   otherwise be counted as new "distinct" speakers every time — and would
   leak across train/test splits.

Heavy deps (insightface via FaceEmbedder, sklearn) import lazily so the
module stays importable everywhere.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from loguru import logger

from services.quality_indexer.centroid_store import CentroidStore
from services.quality_indexer.face_embedder import FaceEmbedder
from services.quality_indexer.identity_records import (
    SegmentIdentityRecord,
    SpeakerProfile,
    aggregate_demographics,
)


class SpeakerIdentifier:
    """ArcFace-based per-video speaker clustering with optional cross-video
    re-identification, composed from FaceEmbedder + CentroidStore."""

    def __init__(
        self,
        models_dir: Path,
        catalog_db_path: Path,
        cluster_eps: float = 0.40,
        cross_video_enabled: bool = True,
        cross_video_similarity: float = 0.60,
    ):
        self.cluster_eps = cluster_eps
        self.cross_video_enabled = cross_video_enabled
        self.cross_video_similarity = cross_video_similarity

        self.embedder = FaceEmbedder(models_dir)
        self.centroids = CentroidStore(catalog_db_path)

    # ------------------------------------------------------------ embedding

    def embed_frames(self, frames) -> Optional[SegmentIdentityRecord]:
        """Per-segment evidence from the exporter's sampled frames (RAM)."""
        return self.embedder.embed_frames(frames)

    # ------------------------------------------------------------ clustering

    def assign_speakers(
        self,
        video_id: str,
        records: Dict[str, SegmentIdentityRecord],
    ) -> Tuple[Dict[str, str], Dict[str, SpeakerProfile]]:
        """Cluster segment identities for one video and (optionally) match
        clusters against speakers already known from other videos.

        Args:
            video_id: e.g. "md_001".
            records: segment_id → SegmentIdentityRecord.

        Returns:
            (segment_id → speaker_id, speaker_id → SpeakerProfile)
        """
        if not records:
            return {}, {}

        from sklearn.cluster import DBSCAN

        segment_ids = list(records.keys())
        matrix = np.array([records[s].embedding for s in segment_ids])

        # min_samples=1 → every segment joins a cluster (no DBSCAN noise);
        # a one-off interviewee simply becomes a singleton speaker.
        labels = DBSCAN(
            eps=self.cluster_eps, min_samples=1, metric="cosine",
        ).fit(matrix).labels_

        known_centroids = self.centroids.load_known() if self.cross_video_enabled else {}

        segment_to_speaker: Dict[str, str] = {}
        profiles: Dict[str, SpeakerProfile] = {}
        next_local_index = 0

        for label in sorted(set(labels)):
            member_indices = [i for i, l in enumerate(labels) if l == label]
            member_segments = [segment_ids[i] for i in member_indices]
            centroid = matrix[member_indices].mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) or 1.0)

            speaker_id, identity_match = self._resolve_global_identity(
                video_id, next_local_index, centroid, known_centroids,
            )
            next_local_index += 1

            genders, ages = [], []
            for segment in member_segments:
                genders.extend(records[segment].genders)
                ages.extend(records[segment].ages)
            demographics = aggregate_demographics(genders, ages)

            profiles[speaker_id] = SpeakerProfile(
                speaker_id=speaker_id,
                num_segments=len(member_segments),
                identity_match=identity_match,
                embedding=centroid,
                **demographics,
            )
            for segment in member_segments:
                segment_to_speaker[segment] = speaker_id

        if self.cross_video_enabled:
            self.centroids.persist(profiles)

        matched = sum(1 for p in profiles.values() if p.identity_match == "auto")
        logger.info(
            f"Speaker identity [{video_id}]: {len(records)} segments → "
            f"{len(profiles)} speakers ({matched} matched cross-video)"
        )
        return segment_to_speaker, profiles

    # ------------------------------------------------- cross-video identity

    def _resolve_global_identity(
        self,
        video_id: str,
        local_index: int,
        centroid: np.ndarray,
        known_centroids: Dict[str, np.ndarray],
    ) -> Tuple[str, str]:
        """Match a local cluster against stored global speakers.

        Returns (speaker_id, identity_match) where identity_match is "auto"
        for a cross-video match (review GUI can audit those) or "new".
        """
        if known_centroids:
            similarities = {
                speaker_id: float(np.dot(centroid, vector))
                for speaker_id, vector in known_centroids.items()
            }
            best_id, best_similarity = max(similarities.items(), key=lambda kv: kv[1])
            if best_similarity >= self.cross_video_similarity:
                logger.debug(
                    f"Cross-video identity: cluster → {best_id} "
                    f"(similarity {best_similarity:.3f})"
                )
                return best_id, "auto"
        return f"{video_id}_spk{local_index}", "new"
