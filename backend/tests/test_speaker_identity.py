"""Unit tests for speaker identity — pure parts always run; clustering runs
only when scikit-learn is installed.

Run from the repo root:
    python backend/tests/test_speaker_identity.py
"""

import sys
import tempfile
import types
import unittest
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_BACKEND_DIR), str(_BACKEND_DIR / "shared")]

if "loguru" not in sys.modules:
    _loguru = types.ModuleType("loguru")

    class _SilentLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    _loguru.logger = _SilentLogger()
    sys.modules["loguru"] = _loguru

import numpy as np  # noqa: E402

from services.quality_indexer.identity_records import (  # noqa: E402
    SegmentIdentityRecord,
    bucket_age,
    build_identity_record,
)
from services.quality_indexer.speaker_identifier import SpeakerIdentifier  # noqa: E402

try:
    import sklearn  # noqa: F401
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class AgeBucketTests(unittest.TestCase):
    def test_buckets(self):
        self.assertEqual(bucket_age(22), "18-30")
        self.assertEqual(bucket_age(30.9), "18-30")
        self.assertEqual(bucket_age(31), "31-50")
        self.assertEqual(bucket_age(50.9), "31-50")
        self.assertEqual(bucket_age(51), "51+")


class RecordRoundtripTests(unittest.TestCase):
    def test_json_roundtrip(self):
        record = SegmentIdentityRecord(
            embedding=[0.1] * 8, genders=[1, 1, 0], ages=[40.2, 43.7],
        )
        restored = SegmentIdentityRecord.from_json_dict(record.to_json_dict())
        self.assertEqual(len(restored.embedding), 8)
        self.assertEqual(restored.genders, [1, 1, 0])
        self.assertEqual(restored.ages, [40.2, 43.7])

    def test_empty_returns_none(self):
        self.assertIsNone(SegmentIdentityRecord.from_json_dict({}))
        self.assertIsNone(SegmentIdentityRecord.from_json_dict({"embedding": []}))


class BuildRecordTests(unittest.TestCase):
    def test_mean_embedding_is_renormalized(self):
        # Two unit vectors along different axes — their mean has norm < 1
        # and must come back normalized for cosine math downstream.
        e1 = np.array([1.0, 0.0], dtype=np.float32)
        e2 = np.array([0.0, 1.0], dtype=np.float32)
        record = build_identity_record([e1, e2], genders=[1, 1], ages=[40.0, 42.0])
        self.assertAlmostEqual(
            float(np.linalg.norm(record.embedding)), 1.0, places=6,
        )
        self.assertEqual(record.genders, [1, 1])

    def test_no_embeddings_returns_none(self):
        self.assertIsNone(build_identity_record([], genders=[1], ages=[40.0]))


def _unit(vector):
    v = np.array(vector, dtype=float)
    return (v / np.linalg.norm(v)).tolist()


@unittest.skipUnless(HAS_SKLEARN, "scikit-learn not installed")
class ClusteringTests(unittest.TestCase):
    def _identifier(self, tmp, cross_video=False):
        return SpeakerIdentifier(
            models_dir=Path(tmp) / "models",
            catalog_db_path=Path(tmp) / "dataset.db",
            cluster_eps=0.40,
            cross_video_enabled=cross_video,
            cross_video_similarity=0.60,
        )

    def test_recurring_speaker_gets_one_id(self):
        # Anchor (direction ~e1) speaks in segments 0, 2, 4; guest (~e2) in 1, 3.
        anchor = _unit([1.0, 0.05, 0.0])
        guest = _unit([0.0, 1.0, 0.05])
        records = {
            f"seg_{i}": SegmentIdentityRecord(
                embedding=anchor if i % 2 == 0 else guest,
                genders=[1] if i % 2 == 0 else [0],
                ages=[45.0] if i % 2 == 0 else [30.0],
            )
            for i in range(5)
        }
        with tempfile.TemporaryDirectory() as tmp:
            mapping, profiles = self._identifier(tmp).assign_speakers("md_001", records)

        self.assertEqual(len(profiles), 2)
        anchor_ids = {mapping[f"seg_{i}"] for i in (0, 2, 4)}
        guest_ids = {mapping[f"seg_{i}"] for i in (1, 3)}
        self.assertEqual(len(anchor_ids), 1)
        self.assertEqual(len(guest_ids), 1)
        self.assertNotEqual(anchor_ids, guest_ids)

        anchor_profile = profiles[anchor_ids.pop()]
        self.assertEqual(anchor_profile.gender, "M")
        self.assertEqual(anchor_profile.age_group, "31-50")
        self.assertEqual(anchor_profile.num_segments, 3)

    def test_cross_video_reidentification(self):
        anchor = _unit([1.0, 0.02, 0.01])
        records_v1 = {"a": SegmentIdentityRecord(embedding=anchor, genders=[1], ages=[50.0])}
        # Slightly different evidence for the same person in the next video
        anchor_later = _unit([0.98, 0.05, 0.0])
        records_v2 = {"b": SegmentIdentityRecord(embedding=anchor_later, genders=[1], ages=[51.0])}

        with tempfile.TemporaryDirectory() as tmp:
            identifier = self._identifier(tmp, cross_video=True)
            _, profiles_v1 = identifier.assign_speakers("md_001", records_v1)
            mapping_v2, profiles_v2 = identifier.assign_speakers("md_002", records_v2)

        original_id = next(iter(profiles_v1))
        self.assertEqual(mapping_v2["b"], original_id)  # same GLOBAL speaker
        self.assertEqual(profiles_v2[original_id].identity_match, "auto")


if __name__ == "__main__":
    unittest.main(verbosity=2)
