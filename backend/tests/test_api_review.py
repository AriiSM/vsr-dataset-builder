"""Functional tests for the review routes — real files on disk (annotation,
fake media), real DB; ffmpeg-dependent video trim runs only where present.

Run from the repo root:
    python backend/tests/test_api_review.py
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

client = TestClient(app)

_ANNO = """Text:  BUNĂ SEARA DRAGI TELESPECTATORI
Original:  BUNA SEARA DRAGI TELESPECTATORI
Conf:  2
WORD START END ASDSCORE
BUNĂ 0.00 0.40 8.1
SEARA 0.50 0.90 8.5
DRAGI 1.00 1.40 7.9
TELESPECTATORI 1.50 2.40 8.8"""


def _db():
    return CatalogDatabase(_ROOT / "catalog" / "dataset.db")


def _seed(segment_id: str, video_id: str = "md_001"):
    db = _db()
    db.videos.ensure_exists(video_id)
    db.videos.set_status(video_id, "completed")
    db.segments.upsert({
        "segment_id": segment_id, "video_id": video_id,
        "duration": 2.4, "text": "BUNĂ SEARA DRAGI TELESPECTATORI",
        "num_words": 4, "quality_tier": "A",
    }, words=[
        {"word": w, "start": i * 0.5, "end": i * 0.5 + 0.4,
         "score": None, "asd_score": 8.0}
        for i, w in enumerate(["BUNĂ", "SEARA", "DRAGI", "TELESPECTATORI"])
    ])
    db.close()

    base = _ROOT / "processed" / video_id
    for sub, name in [("face_crop", f"{segment_id}.mp4"),
                      ("mouth_crop", f"{segment_id}.mp4"),
                      ("audio", f"{segment_id}.wav")]:
        (base / sub).mkdir(parents=True, exist_ok=True)
        (base / sub / name).write_bytes(b"fake")
    (base / "text").mkdir(parents=True, exist_ok=True)
    (base / "text" / f"{segment_id}.txt").write_text(_ANNO, encoding="utf-8")


class ApproveRejectTests(unittest.TestCase):
    def test_approve_sets_status_and_conf3(self):
        _seed("md_001_clip_010_00010")
        body = client.post("/api/segment/md_001_clip_010_00010/review",
                           json={"action": "approve"}).json()
        self.assertEqual(body["status"], "approved")
        db = _db()
        row = db.connection.execute(
            "SELECT review_status, reviewed_at FROM segments"
            " WHERE segment_id = 'md_001_clip_010_00010'").fetchone()
        db.close()
        self.assertEqual(row["review_status"], "approved")
        self.assertTrue(row["reviewed_at"])
        anno = (_ROOT / "processed" / "md_001" / "text"
                / "md_001_clip_010_00010.txt").read_text()
        self.assertIn("Conf:  3", anno)

    def test_reject_deletes_files_keeps_row(self):
        _seed("md_001_clip_011_00011")
        base = _ROOT / "processed" / "md_001"
        body = client.post("/api/segment/md_001_clip_011_00011/review",
                           json={"action": "reject"}).json()
        self.assertEqual(body["status"], "rejected")
        for sub, ext in [("face_crop", ".mp4"), ("mouth_crop", ".mp4"),
                         ("audio", ".wav"), ("text", ".txt")]:
            self.assertFalse(
                (base / sub / f"md_001_clip_011_00011{ext}").exists(),
                f"{sub} not deleted")
        db = _db()
        row = db.connection.execute(
            "SELECT review_status FROM segments"
            " WHERE segment_id = 'md_001_clip_011_00011'").fetchone()
        db.close()
        self.assertEqual(row["review_status"], "rejected")  # row survives

    def test_video_validated_when_all_survivors_approved(self):
        _seed("md_002_clip_001_00000", video_id="md_002")
        _seed("md_002_clip_002_00001", video_id="md_002")
        client.post("/api/segment/md_002_clip_001_00000/review",
                    json={"action": "reject"})
        client.post("/api/segment/md_002_clip_002_00001/review",
                    json={"action": "approve"})
        db = _db()
        status = db.videos.get("md_002")["status"]
        db.close()
        self.assertEqual(status, "validated")

    def test_revert_demotes_video(self):
        _seed("md_006_clip_001_00000", video_id="md_006")
        client.post("/api/segment/md_006_clip_001_00000/review",
                    json={"action": "approve"})
        db = _db()
        self.assertEqual(db.videos.get("md_006")["status"], "validated")
        db.close()
        client.post("/api/segment/md_006_clip_001_00000/review",
                    json={"action": "revert"})
        db = _db()
        status = db.videos.get("md_006")["status"]
        db.close()
        self.assertEqual(status, "completed")


class SaveTests(unittest.TestCase):
    def test_save_updates_text_wer_and_conf(self):
        _seed("md_003_clip_001_00000", video_id="md_003")
        body = client.post(
            "/api/segment/md_003_clip_001_00000/review",
            json={"action": "save", "text": "BUNA SEARA DRAGI TELESPECTATORI"},
        ).json()
        self.assertTrue(body["ok"])
        db = _db()
        row = db.connection.execute(
            "SELECT text, wer, transcript_edited, original_text FROM segments"
            " WHERE segment_id = 'md_003_clip_001_00000'").fetchone()
        db.close()
        self.assertEqual(row["text"], "BUNA SEARA DRAGI TELESPECTATORI")
        self.assertEqual(row["transcript_edited"], 1)
        self.assertEqual(row["wer"], 0.0)   # matches the Original reference
        anno = (_ROOT / "processed" / "md_003" / "text"
                / "md_003_clip_001_00000.txt").read_text()
        self.assertIn("Text:  BUNA SEARA DRAGI TELESPECTATORI", anno)
        self.assertIn("Original:  BUNA SEARA", anno)   # preserved

    def test_save_words_rewrites_words_table(self):
        _seed("md_003_clip_002_00001", video_id="md_003")
        words = [
            {"word": "SALUT", "start": 0.0, "end": 0.6, "score": 9.0},
            {"word": "LUME", "start": 0.7, "end": 1.2, "score": 8.0},
        ]
        client.post("/api/segment/md_003_clip_002_00001/review",
                    json={"action": "save_words", "words": words,
                          "text": "SALUT LUME"})
        db = _db()
        stored = db.segments.words_for("md_003_clip_002_00001")
        row = db.connection.execute(
            "SELECT num_words, text FROM segments"
            " WHERE segment_id = 'md_003_clip_002_00001'").fetchone()
        db.close()
        self.assertEqual([w["word"] for w in stored], ["SALUT", "LUME"])
        self.assertEqual(row["num_words"], 2)
        self.assertEqual(row["text"], "SALUT LUME")


class TrimTests(unittest.TestCase):
    def test_trim_shifts_and_drops_words(self):
        # Media files are fakes → ffmpeg fails gracefully (logged);
        # the annotation/word/duration math is what we verify.
        _seed("md_004_clip_001_00000", video_id="md_004")
        body = client.post("/api/segment/md_004_clip_001_00000/trim",
                           json={"start": 0.5, "end": 1.5}).json()
        self.assertEqual(body["duration"], 1.0)
        db = _db()
        stored = db.segments.words_for("md_004_clip_001_00000")
        row = db.connection.execute(
            "SELECT duration, num_words, trimmed, text FROM segments"
            " WHERE segment_id = 'md_004_clip_001_00000'").fetchone()
        db.close()
        # Words at 0.5–0.9 and 1.0–1.4 survive (shifted); 0.0–0.4 and
        # 1.5–2.4 fall outside.
        self.assertEqual(row["num_words"], 2)
        self.assertEqual(row["trimmed"], 1)
        self.assertEqual(row["text"], "SEARA DRAGI")
        self.assertAlmostEqual(stored[0]["start_time"], 0.0, places=2)
        self.assertAlmostEqual(stored[1]["end_time"], 0.9, places=2)

    def test_trim_invalid_range(self):
        self.assertEqual(
            client.post("/api/segment/md_004_clip_001_00000/trim",
                        json={"start": 2, "end": 1}).status_code, 400)


class ImportReviewJsonTests(unittest.TestCase):
    def test_one_time_import_is_idempotent(self):
        _seed("md_005_clip_001_00000", video_id="md_005")
        review_json = _ROOT / "catalog" / "review_status.json"
        review_json.write_text(
            '{"md_005_clip_001_00000": {"status": "approved", "trimmed": true},'
            ' "ghost_segment": {"status": "rejected"}}')
        sys.path.insert(0, str(_BACKEND_DIR / "tools"))
        from import_review_status import import_review
        first = import_review(review_json, _ROOT / "catalog")
        self.assertEqual(first, 1)          # ghost not in catalog
        second = import_review(review_json, _ROOT / "catalog")
        self.assertEqual(second, 0)         # idempotent — DB verdict wins
        db = _db()
        row = db.connection.execute(
            "SELECT review_status, trimmed FROM segments"
            " WHERE segment_id = 'md_005_clip_001_00000'").fetchone()
        db.close()
        self.assertEqual(row["review_status"], "approved")
        self.assertEqual(row["trimmed"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
