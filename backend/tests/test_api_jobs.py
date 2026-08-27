"""Functional tests for the jobs API — REST surface + Flask-compat layer.

Run from the repo root:
    python backend/tests/test_api_jobs.py
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


def _db():
    return CatalogDatabase(_ROOT / "catalog" / "dataset.db")


def _drain_queue():
    """Finish every non-terminal job so the next test starts clean."""
    db = _db()
    for job in db.jobs.all(limit=100):
        if job["status"] in ("pending", "running", "cancel_requested"):
            db.jobs.finish(job["id"], "done")
    db.close()


class JobsRestTests(unittest.TestCase):
    def setUp(self):
        _drain_queue()

    def test_create_and_get(self):
        response = client.post("/api/jobs", json={
            "type": "batch", "params": {"status": ["pending"], "limit": 2}})
        self.assertEqual(response.status_code, 201)
        job_id = response.json()["id"]

        job = client.get(f"/api/jobs/{job_id}").json()
        self.assertEqual(job["status"], "pending")
        self.assertEqual(job["params"]["limit"], 2)

    def test_bad_type_rejected(self):
        self.assertEqual(
            client.post("/api/jobs", json={"type": "teleport"}).status_code, 400)

    def test_single_gpu_gate(self):
        first = client.post("/api/jobs", json={"type": "sync", "params": {}})
        self.assertEqual(first.status_code, 201)
        second = client.post("/api/jobs", json={"type": "sync", "params": {}})
        self.assertEqual(second.status_code, 409)

    def test_cancel_pending_job(self):
        job_id = client.post(
            "/api/jobs", json={"type": "sync", "params": {}}).json()["id"]
        self.assertTrue(client.post(f"/api/jobs/{job_id}/cancel").json()["ok"])
        self.assertEqual(
            client.get(f"/api/jobs/{job_id}").json()["status"],
            "cancel_requested")

    def test_log_tail_incremental(self):
        job_id = client.post(
            "/api/jobs", json={"type": "sync", "params": {}}).json()["id"]
        log_path = _ROOT / "logs" / f"job_{job_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("linia unu\n")
        db = _db()
        db.jobs.stamp_execution(job_id, "sha", "cfg", str(log_path))
        db.close()

        first = client.get(f"/api/jobs/{job_id}/log").json()
        self.assertIn("linia unu", first["log"])
        with open(log_path, "a") as f:
            f.write("linia doi\n")
        second = client.get(
            f"/api/jobs/{job_id}/log", params={"offset": first["offset"]}).json()
        self.assertEqual(second["log"], "linia doi\n")


class FlaskCompatTests(unittest.TestCase):
    def setUp(self):
        _drain_queue()

    def test_start_batch_failed_maps_to_batch_job(self):
        response = client.post("/api/start", json={
            "mode": "batch-failed", "limit": 3, "video_ids": "md_001, md_005"})
        body = response.json()
        self.assertEqual(body["status"], "started")
        job = client.get(f"/api/jobs/{body['job_id']}").json()
        self.assertEqual(job["type"], "batch")
        self.assertEqual(job["params"]["status"], ["failed"])
        self.assertEqual(job["params"]["video_ids"], ["md_001", "md_005"])

    def test_start_single_requires_id_and_url(self):
        self.assertEqual(
            client.post("/api/start", json={"mode": "single"}).status_code, 400)
        response = client.post("/api/start", json={
            "mode": "single", "video_id": "md_009", "url": "http://y"})
        job = client.get(f"/api/jobs/{response.json()['job_id']}").json()
        self.assertEqual(job["type"], "single")
        self.assertEqual(job["params"]["video_id"], "md_009")

    def test_start_409_while_active(self):
        client.post("/api/start", json={"mode": "batch-pending"})
        self.assertEqual(
            client.post("/api/start", json={"mode": "batch-pending"}).status_code,
            409)

    def test_status_reflects_latest_job(self):
        response = client.post("/api/start", json={"mode": "batch-pending"})
        status = client.get("/api/status").json()
        self.assertTrue(status["running"])
        self.assertEqual(status["mode"], "batch-pending")
        self.assertEqual(status["job_id"], response.json()["job_id"])
        self.assertIsInstance(status["log"], list)

    def test_stop_cancels_active(self):
        client.post("/api/start", json={"mode": "batch-pending"})
        self.assertTrue(client.post("/api/stop").json()["ok"])
        self.assertFalse(client.get("/api/status").json()["running"]
                         if client.get("/api/status").json()["job_status"]
                         not in ("cancel_requested",) else False)

    def test_stop_409_when_idle(self):
        self.assertEqual(client.post("/api/stop").status_code, 409)

    def test_bulk_import_accepts_url_blob(self):
        response = client.post("/api/bulk_import", json={
            "urls": "http://a\n# comentariu\nhttp://b", "prefix": "md"})
        body = response.json()
        self.assertEqual(body["urls"], 2)
        job = client.get(f"/api/jobs/{body['job_id']}").json()
        self.assertEqual(job["type"], "bulk_import")
        self.assertEqual(job["params"]["urls"], ["http://a", "http://b"])

    def test_bulk_import_empty_rejected(self):
        self.assertEqual(
            client.post("/api/bulk_import", json={"urls": "# doar comentarii"})
            .status_code, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
