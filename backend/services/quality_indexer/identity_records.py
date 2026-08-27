"""
Identity records — the data types of speaker identity.

SegmentIdentityRecord is the per-segment evidence (embedding + demographic
votes) that travels through the checkpoint (resume-safe) to the end-of-video
clustering. SpeakerProfile is the aggregated result for one clustered person.

Pure data + pure math (numpy only) — no models, no disk, fully unit-testable.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


def bucket_age(age_years: float) -> str:
    """Numeric age → speakers_registry age_group enum."""
    if age_years < 31:
        return "18-30"
    if age_years < 51:
        return "31-50"
    return "51+"


@dataclass
class SegmentIdentityRecord:
    """Face identity evidence collected from one exported segment."""
    embedding: List[float]              # mean ArcFace embedding (512-d, normalized)
    genders: List[int] = field(default_factory=list)   # 0=female, 1=male per sampled frame
    ages: List[float] = field(default_factory=list)    # age estimate per sampled frame

    def to_json_dict(self) -> dict:
        """Compact form for the per-video checkpoint (resume-safe)."""
        return {
            "embedding": [round(v, 4) for v in self.embedding],
            "genders": self.genders,
            "ages": [round(a, 1) for a in self.ages],
        }

    @classmethod
    def from_json_dict(cls, data: dict) -> Optional["SegmentIdentityRecord"]:
        if not data or not data.get("embedding"):
            return None
        return cls(
            embedding=list(data["embedding"]),
            genders=list(data.get("genders", [])),
            ages=list(data.get("ages", [])),
        )


@dataclass
class SpeakerProfile:
    """Aggregated identity + demographics for one clustered speaker."""
    speaker_id: str
    gender: str                 # "M" | "F" | ""
    gender_confidence: float    # fraction of frames voting for the majority
    age_estimate: float         # median age over all sampled frames
    age_std: float              # spread — large means unreliable, review it
    age_group: str
    num_segments: int
    identity_match: str         # "new" | "auto" (matched a stored centroid)
    embedding: np.ndarray


def build_identity_record(
    embeddings: List[np.ndarray],
    genders: List[int],
    ages: List[float],
) -> Optional[SegmentIdentityRecord]:
    """Aggregate per-frame evidence into one segment record.

    The mean embedding is re-normalized to unit length so cosine math stays
    valid downstream. Returns None when no frame produced an embedding.
    """
    if not embeddings:
        return None

    mean_embedding = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(mean_embedding)
    if norm > 0:
        mean_embedding = mean_embedding / norm

    return SegmentIdentityRecord(
        embedding=[float(v) for v in mean_embedding],
        genders=list(genders),
        ages=list(ages),
    )


def aggregate_demographics(genders: List[int], ages: List[float]) -> dict:
    """Majority gender + median age over ALL sampled frames of one cluster.

    Far more stable than judging from a few frames of a single segment.
    """
    gender = ""
    gender_confidence = 0.0
    if genders:
        male_votes = sum(genders)
        gender = "M" if male_votes > len(genders) / 2 else "F"
        gender_confidence = max(male_votes, len(genders) - male_votes) / len(genders)

    age_estimate = float(np.median(ages)) if ages else float("nan")
    age_std = float(np.std(ages)) if len(ages) > 1 else 0.0

    return {
        "gender": gender,
        "gender_confidence": round(gender_confidence, 3),
        "age_estimate": round(age_estimate, 1) if not math.isnan(age_estimate) else float("nan"),
        "age_std": round(age_std, 1),
        "age_group": bucket_age(age_estimate) if not math.isnan(age_estimate) else "",
    }
