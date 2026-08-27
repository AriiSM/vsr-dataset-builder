"""Functional tests for the read-only API routes (FastAPI TestClient on a
synthetic dataset.db). No models, no network, no media transcoding.

Run from the repo root:
    python backend/tests/test_api_readonly.py
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_BACKEND_DIR), str(_BACKEND_DIR / "shared")]

_TMP = tempfile.TemporaryDirectory()
_ROOT = Path(_TMP.name)
os.environ["VSR_CATALOG_DIR"] = str(_ROOT / "catalog")
os.environ["VSR_PROCESSED_DIR"] = str(_ROOT / "processed")
os.environ["VSR_CACHE_DIR"] = str(_ROOT / "cache")
os.environ["VSR_FRONTEND_DIST"] = str(_ROOT / "no-dist")

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402
from vsr_shared.catalog_db import CatalogDatabase  # noqa: E402


def _seed():
    db = CatalogDatabase(_ROOT / "catalog" / "dataset.db")
    db.videos.ensure_exists("md_001")
    db.videos.update_fields("md_001", {
        "youtube_url": "http://y/1", "region": "MD", "status": "completed",
        "title": "Știri de seară", "duration_seconds": 600.0,
        "total_segments": 3, "total_duration_extracted": 7.4,
    })
    common = {"video_id": "md_001", "speaker_id": "md_001_spk0"}
    db.segments.upsert({
        "segment_id": "md_001_clip_001_00000", **common,
        "start_time": 10.0, "end_time": 12.5, "duration": 2.5,
        "text": "BUNĂ SEARA", "num_words": 2, "num_chars": 10,
        "asd_score": 8.0, "syncnet_conf": 3.1, "whisper_conf": 0.95,
        "whisper_conf_min": 0.8, "quality_tier": "A", "wer": 0.0,
        "review_status": "approved",
    }, words=[
        {"word": "BUNĂ", "start": 0.0, "end": 0.5, "score": None, "asd_score": 8.2},
        {"word": "SEARA", "start": 0.6, "end": 1.1, "score": None, "asd_score": 7.8},
    ])
    db.segments.upsert({
        "segment_id": "md_001_clip_002_00001", **common,
        "start_time": 20.0, "end_time": 22.9, "duration": 2.9,
        "text": "VREMEA DE MÂINE", "num_words": 3, "num_chars": 15,
        "asd_score": 6.0, "syncnet_conf": 2.0, "whisper_conf": 0.75,
        "whisper_conf_min": 0.55, "quality_tier": "B", "wer": 0.1,
    })
    # Rejected — must vanish from lists/aggregates but count as rejected.
    db.segments.upsert({
        "segment_id": "md_001_clip_003_00002", **common,
        "duration": 2.0, "text": "GUNOI", "num_words": 1, "num_chars": 5,
        "review_status": "rejected",
    })
    db.speakers.upsert("md_001_spk0", {"gender": "M", "age_group": "31-50"})
    db.close()


_seed()
client = TestClient(app)


class HealthTests(unittest.TestCase):
    def test_health(self):
        body = client.get("/api/health").json()
        self.assertTrue(body["ok"])
        self.assertTrue(body["catalog_db_exists"])


class SegmentsTests(unittest.TestCase):
    def test_list_excludes_rejected_and_joins_region(self):
        body = client.get("/api/segments").json()
        self.assertEqual(body["total"], 2)
        ids = {s["segment_id"] for s in body["segments"]}
        self.assertNotIn("md_001_clip_003_00002", ids)
        self.assertEqual(body["segments"][0]["region"], "MD")

    def test_detail_shape(self):
        body = client.get("/api/segment/md_001_clip_001_00000").json()
        self.assertEqual(body["text"], "BUNĂ SEARA")
        self.assertEqual(body["review"], {"status": "approved"})
        self.assertFalse(body["has_video"])
        self.assertIn("video_path", body)

    def test_detail_404(self):
        self.assertEqual(client.get("/api/segment/nope").status_code, 404)

    def test_review_status_map(self):
        body = client.get("/api/review_status").json()
        self.assertEqual(body["md_001_clip_001_00000"]["status"], "approved")
        self.assertEqual(body["md_001_clip_003_00002"]["status"], "rejected")

    def test_set_speaker(self):
        response = client.post(
            "/api/segment/md_001_clip_002_00001/speaker",
            json={"speaker_id": "md_001_spk1"})
        self.assertTrue(response.json()["ok"])
        self.assertEqual(response.json()["old_speaker_id"], "md_001_spk0")
        # New speaker got a registry stub; move the segment back for
        # the other tests (module-level seed is shared).
        speakers = client.get("/api/speakers").json()["speakers"]
        self.assertIn("md_001_spk1", {s["speaker_id"] for s in speakers})
        client.post("/api/segment/md_001_clip_002_00001/speaker",
                    json={"speaker_id": "md_001_spk0"})

    def test_media_404_when_file_missing(self):
        response = client.get("/api/media/md_001/md_001_clip_001_00000")
        self.assertEqual(response.status_code, 404)


class VideosTests(unittest.TestCase):
    def test_videos_live_stats_exclude_rejected(self):
        body = client.get("/api/videos").json()
        self.assertEqual(len(body["videos"]), 1)
        video = body["videos"][0]
        self.assertEqual(video["total_segments"], "2")     # not 3 — rejected out
        self.assertEqual(video["region"], "MD")


class StatsTests(unittest.TestCase):
    def test_stats_shapes_and_reject_handling(self):
        body = client.get("/api/stats").json()
        self.assertEqual(body["videos"]["total"], 1)
        seg = body["segments"]
        self.assertEqual(seg["total"], 2)
        self.assertEqual(seg["rejected_count"], 1)
        self.assertEqual(seg["approved"]["count"], 1)
        self.assertAlmostEqual(seg["total_duration_s"], 5.4, places=1)
        self.assertEqual(seg["tiers"]["A"]["count"], 1)
        self.assertEqual(seg["training_ready"]["count"], 1)
        # Conf: seg1 (0.8/0.95)→3, seg2 (0.55/0.75)→2
        self.assertEqual(seg["by_conf"]["3"], 1)
        self.assertEqual(seg["by_conf"]["2"], 1)

    def test_distributions_health(self):
        body = client.get("/api/stats/distributions").json()
        self.assertEqual(body["health"]["n_segments"], 2)
        self.assertEqual(len(body["duration_buckets"]), 14)

    def test_vocabulary_uses_words_table_with_fallback(self):
        body = client.get("/api/vocabulary").json()
        by_word = {w["word"]: w for w in body["words"]}
        # From the words table (real timings): 0.5s each.
        self.assertAlmostEqual(by_word["BUNĂ"]["duration"], 0.5, places=2)
        # Fallback (segment 2 has no words rows): 2.9s / 3 words each.
        self.assertAlmostEqual(by_word["VREMEA"]["duration"], 0.97, places=2)
        # The rejected segment's word must NOT appear.
        self.assertNotIn("GUNOI", by_word)


class SpeakersTests(unittest.TestCase):
    def test_speakers_with_live_aggregates(self):
        body = client.get("/api/speakers").json()
        spk = next(s for s in body["speakers"]
                   if s["speaker_id"] == "md_001_spk0")
        self.assertEqual(spk["gender"], "M")
        self.assertEqual(spk["num_segments"], "3")   # view counts all rows

    def test_update_speaker(self):
        response = client.post("/api/speaker/md_001_spk0",
                               json={"speaker_name": "Crainicul", "bogus": "x"})
        self.assertTrue(response.json()["ok"])
        self.assertEqual(response.json()["fields"], {"speaker_name": "Crainicul"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
