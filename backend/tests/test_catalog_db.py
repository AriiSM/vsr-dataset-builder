"""Unit tests for the catalog database (real SQLite — stdlib, no stubs).

Run from the repo root:
    python backend/tests/test_catalog_db.py
"""

import sys
import tempfile
import unittest
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_BACKEND_DIR), str(_BACKEND_DIR / "shared")]

import numpy as np  # noqa: E402

from vsr_shared.catalog_db import CatalogDatabase, SCHEMA_VERSION  # noqa: E402


def _segment_row(segment_id="md_001_clip_001_00000", video_id="md_001", **extra):
    row = {
        "segment_id": segment_id,
        "video_id": video_id,
        "clip_id": "md_001_clip_001",
        "start_time": 12.5,
        "end_time": 15.1,
        "duration": 2.6,
        "text": "ACESTE ZILE",
        "num_words": 2,
        "asd_score": 8.4,
        "quality_tier": "A",
        "audio_speaker_label": "SPEAKER_00",
    }
    row.update(extra)
    return row


class CatalogDatabaseTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = CatalogDatabase(Path(self._tmp.name) / "dataset.db")
        self.db.videos.ensure_exists("md_001")

    def tearDown(self):
        self.db.close()
        self._tmp.cleanup()

    def test_schema_init_is_idempotent(self):
        # Re-opening the same file must not fail or duplicate anything.
        self.db.close()
        db2 = CatalogDatabase(Path(self._tmp.name) / "dataset.db")
        version = db2.connection.execute("PRAGMA user_version").fetchone()[0]
        self.assertEqual(version, SCHEMA_VERSION)
        db2.close()
        self.db = CatalogDatabase(Path(self._tmp.name) / "dataset.db")

    def test_segment_and_words_roundtrip(self):
        words = [
            {"word": "ACESTE", "start": 0.0, "end": 0.4, "score": 0.9, "asd_score": 8.1},
            {"word": "ZILE", "start": 0.5, "end": 0.8, "score": 0.95, "asd_score": 8.7},
        ]
        self.db.segments.upsert(_segment_row(), words=words)

        stored = self.db.segments.for_video("md_001")
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["text"], "ACESTE ZILE")
        self.assertEqual(stored[0]["quality_tier"], "A")

        stored_words = self.db.segments.words_for("md_001_clip_001_00000")
        self.assertEqual([w["word"] for w in stored_words], ["ACESTE", "ZILE"])
        self.assertAlmostEqual(stored_words[1]["asd_score"], 8.7)

    def test_upsert_replaces_words_not_duplicates(self):
        words = [{"word": "UNU", "start": 0, "end": 1, "score": 1, "asd_score": 1}]
        self.db.segments.upsert(_segment_row(), words=words)
        self.db.segments.upsert(_segment_row(), words=words)  # re-run (resume)
        self.assertEqual(len(self.db.segments.words_for("md_001_clip_001_00000")), 1)

    def test_unknown_columns_are_ignored(self):
        self.db.segments.upsert(_segment_row(identity_frames="junk", bogus=1))
        self.assertEqual(len(self.db.segments.for_video("md_001")), 1)

    def test_delete_segment_cascades_to_words(self):
        self.db.segments.upsert(
            _segment_row(),
            words=[{"word": "X", "start": 0, "end": 1, "score": 1, "asd_score": 0}],
        )
        self.db.segments.delete_for_video("md_001")
        self.assertEqual(self.db.segments.words_for("md_001_clip_001_00000"), [])

    def test_video_status_and_fields(self):
        self.db.videos.set_status("md_002", "processing")
        self.db.videos.update_fields("md_002", {
            "total_segments": 12, "region": "MD", "not_a_column": "ignored",
        })
        video = self.db.videos.get("md_002")
        self.assertEqual(video["status"], "processing")
        self.assertEqual(video["total_segments"], 12)
        self.assertEqual(self.db.videos.region("md_002"), "MD")

    def test_speaker_upsert_and_stats_view(self):
        self.db.segments.upsert(_segment_row(speaker_id="md_001_spk0"))
        self.db.segments.upsert(_segment_row(
            segment_id="md_001_clip_002_00001", speaker_id="md_001_spk0",
            duration=3.0,
        ))
        self.db.speakers.upsert("md_001_spk0", {"gender": "M", "age_group": "31-50"})

        rows = self.db.speakers.all_with_stats()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["gender"], "M")
        self.assertEqual(rows[0]["num_segments"], 2)
        self.assertAlmostEqual(rows[0]["total_duration_s"], 5.6)

    def test_speaker_centroid_roundtrip(self):
        vector = np.random.default_rng(0).random(512).astype(np.float32)
        self.db.speakers.set_centroid("md_001_spk0", vector)
        stored = self.db.speakers.centroids()
        self.assertIn("md_001_spk0", stored)
        np.testing.assert_array_almost_equal(stored["md_001_spk0"], vector)

    def test_dropped_clip_recorded_with_reason(self):
        self.db.dropped.record(
            "md_001", "md_001_clip_007", "low_asd_score=1.3",
            start_time=88.0, end_time=91.0, asd_score=1.3,
        )
        drops = self.db.dropped.for_video("md_001")
        self.assertEqual(len(drops), 1)
        self.assertEqual(drops[0]["reason"], "low_asd_score=1.3")

    def test_job_claim_is_atomic_and_single_winner(self):
        job_id = self.db.jobs.create("batch", {"status": ["pending"]},
                                     git_sha="abc123")
        claimed = self.db.jobs.claim_next("worker-1")
        self.assertEqual(claimed["id"], job_id)
        self.assertEqual(claimed["status"], "running")
        # The queue is now empty — a second claim finds nothing.
        self.assertIsNone(self.db.jobs.claim_next("worker-2"))

    def test_job_lifecycle_and_cancel(self):
        job_id = self.db.jobs.create("single", {"video_id": "md_001"})
        self.db.jobs.claim_next("w")
        self.db.jobs.heartbeat(job_id, progress={"video": "md_001", "pct": 40})
        self.assertTrue(self.db.jobs.request_cancel(job_id))
        self.db.jobs.finish(job_id, "cancelled")
        job = self.db.jobs.get(job_id)
        self.assertEqual(job["status"], "cancelled")
        self.assertIn("md_001", job["progress_json"])
        # Finished jobs cannot be cancel-requested again.
        self.assertFalse(self.db.jobs.request_cancel(job_id))

    def test_stale_running_job_is_recovered_as_interrupted(self):
        job_id = self.db.jobs.create("batch", {})
        self.db.jobs.claim_next("w1")
        # Fresh heartbeat → NOT stale.
        self.assertEqual(self.db.jobs.recover_stale(60), [])
        # Fake an old heartbeat (worker died an hour ago).
        with self.db.connection:
            self.db.connection.execute(
                "UPDATE jobs SET heartbeat_at = '2020-01-01 00:00:00' WHERE id = ?",
                (job_id,))
        self.assertEqual(self.db.jobs.recover_stale(60), [job_id])
        job = self.db.jobs.get(job_id)
        self.assertEqual(job["status"], "interrupted")
        self.assertIn("heartbeat", job["error"])

    def test_stamp_execution_and_cancel_flag(self):
        job_id = self.db.jobs.create("single", {"video_id": "md_001"})
        self.db.jobs.stamp_execution(job_id, "abc123", "cfg456", "logs/job_1.log")
        job = self.db.jobs.get(job_id)
        self.assertEqual(job["git_sha"], "abc123")
        self.assertEqual(job["config_hash"], "cfg456")
        self.assertFalse(self.db.jobs.is_cancel_requested(job_id))
        self.db.jobs.request_cancel(job_id)
        self.assertTrue(self.db.jobs.is_cancel_requested(job_id))

    def test_jobs_listing_newest_first(self):
        first = self.db.jobs.create("single", {})
        second = self.db.jobs.create("batch", {})
        listed = self.db.jobs.all()
        self.assertEqual([j["id"] for j in listed], [second, first])

    def test_dataset_overview_view(self):
        self.db.segments.upsert(_segment_row(quality_tier="A"))
        self.db.segments.upsert(_segment_row(
            segment_id="md_001_clip_002_00001", quality_tier="B"))
        overview = dict(self.db.connection.execute(
            "SELECT * FROM dataset_overview").fetchone())
        self.assertEqual(overview["num_segments"], 2)
        self.assertEqual(overview["tier_a"], 1)
        self.assertEqual(overview["tier_b"], 1)

    def test_online_backup(self):
        self.db.segments.upsert(_segment_row())
        backup_path = Path(self._tmp.name) / "backups" / "dataset.db"
        self.db.backup_to(backup_path)
        restored = CatalogDatabase(backup_path)
        self.assertEqual(len(restored.segments.for_video("md_001")), 1)
        restored.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
