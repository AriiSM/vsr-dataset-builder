"""Functional tests for the queue worker — real SQLite queue + log files,
fake pipeline (no models, no network).

Run from the repo root:
    python backend/tests/test_worker_queue.py
"""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_BACKEND_DIR), str(_BACKEND_DIR / "shared")]


# Stub the heavy dependency chain pulled in via orchestrator.pipeline.
class _StubModule(types.ModuleType):
    __path__ = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {"__init__": lambda self, *a, **k: None})


for _name in ["cv2", "torch", "torch.nn", "torch.nn.functional", "torchaudio",
              "whisperx", "mediapipe", "yt_dlp", "scipy", "scipy.ndimage",
              "scipy.signal", "python_speech_features", "sklearn",
              "sklearn.cluster", "insightface", "insightface.app", "pandas"]:
    if _name not in sys.modules:
        sys.modules[_name] = _StubModule(_name)
        _parent, _, _child = _name.rpartition(".")
        if _parent and _parent in sys.modules:
            setattr(sys.modules[_parent], _child, sys.modules[_name])

# loguru is real when installed; otherwise a silent stand-in. The worker
# writes job logs through it, so prefer the real one for the log-file test.
try:
    import loguru  # noqa: F401
except ImportError:
    _loguru = types.ModuleType("loguru")

    class _FileLogger:
        def __init__(self):
            self._sinks = {}
            self._next = 1

        def add(self, sink, **kwargs):
            sink_id = self._next
            self._next += 1
            self._sinks[sink_id] = Path(str(sink))
            return sink_id

        def remove(self, sink_id=None):
            self._sinks.pop(sink_id, None)

        def _write(self, message):
            for path in self._sinks.values():
                with open(path, "a") as f:
                    f.write(message + "\n")

        def info(self, message, *a, **k):
            self._write(str(message))

        def __getattr__(self, _name):
            return lambda *a, **k: self._write(" ".join(str(x) for x in a))

    _loguru.logger = _FileLogger()
    sys.modules["loguru"] = _loguru

from orchestrator.processing_results import PipelineCancelled  # noqa: E402
from vsr_shared.catalog_db import CatalogDatabase  # noqa: E402
from worker.job_executor import JobExecutor  # noqa: E402
from worker.main import Worker  # noqa: E402


class _FakePipeline:
    """Looks like VSRPipeline to JobExecutor: hooks + process_* methods."""

    def __init__(self, behavior="ok"):
        self.behavior = behavior
        self.on_progress = None
        self.should_cancel = None
        self.calls = []
        self.config = types.SimpleNamespace(metadata_dir=Path("."))

    def _clip_loop(self, video_id):
        for clip_num in range(1, 4):
            if self.should_cancel is not None and self.should_cancel():
                raise PipelineCancelled(video_id)
            if self.on_progress is not None:
                self.on_progress({"video_id": video_id, "stage": "clips",
                                  "clip_num": clip_num, "total_clips": 3})

    def process_video(self, video_id, youtube_url, **kwargs):
        self.calls.append(("single", video_id))
        if self.behavior == "crash":
            raise ValueError("boom")
        self._clip_loop(video_id)

    def process_batch(self, csv_path, **kwargs):
        self.calls.append(("batch", kwargs))
        self._clip_loop("md_batch")


class _FakeExecutor(JobExecutor):
    def __init__(self, pipeline):
        super().__init__(Path("config/config.yaml"))
        self._fake = pipeline

    def _get_pipeline(self):
        return self._fake


class WorkerQueueTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        (root / "catalog").mkdir()
        self.config_path = root / "config.yaml"
        self.config_path.write_text(
            f"paths:\n  base_dir: '{root}'\n  catalog_dir: '{root / 'catalog'}'\n")
        self.worker = Worker(self.config_path, worker_name="test-worker")
        self.db = CatalogDatabase(root / "catalog" / "dataset.db")

    def tearDown(self):
        self.worker.db.close()
        self.db.close()
        self._tmp.cleanup()

    def _install_fake(self, behavior="ok"):
        pipeline = _FakePipeline(behavior)
        self.worker.executor = _FakeExecutor(pipeline)
        return pipeline

    def test_empty_queue_runs_nothing(self):
        self._install_fake()
        self.assertFalse(self.worker.run_one())

    def test_single_job_runs_to_done_with_log_and_stamp(self):
        pipeline = self._install_fake()
        job_id = self.db.jobs.create(
            "single", {"video_id": "md_001", "youtube_url": "http://y"})
        self.assertTrue(self.worker.run_one())

        job = self.db.jobs.get(job_id)
        self.assertEqual(job["status"], "done")
        self.assertEqual(pipeline.calls, [("single", "md_001")])
        self.assertIn("job_", job["log_path"])
        self.assertTrue(job["config_hash"])
        log_file = self.worker.logs_dir / f"job_{job_id}.log"
        self.assertTrue(log_file.exists())
        self.assertIn("started by test-worker", log_file.read_text())
        # Final progress snapshot reached the row.
        progress = json.loads(job["progress_json"])
        self.assertEqual(progress.get("clip_num"), 3)

    def test_cancel_requested_marks_job_cancelled(self):
        pipeline = self._install_fake()
        job_id = self.db.jobs.create("single", {"video_id": "md_002"})
        # Request cancel BEFORE the run — the first should_cancel() hits it.
        claimed = self.db.jobs.claim_next("someone-else")  # simulate race? no:
        # undo: we want the WORKER to claim it, so put it back to pending.
        with self.db.connection:
            self.db.connection.execute(
                "UPDATE jobs SET status='pending', claimed_by='' WHERE id=?",
                (job_id,))
        self.db.jobs.request_cancel(job_id)
        # cancel_requested jobs are not claimable — but a running job that
        # turns cancel_requested mid-flight must stop. Simulate mid-flight:
        with self.db.connection:
            self.db.connection.execute(
                "UPDATE jobs SET status='pending' WHERE id=?", (job_id,))
        original_should_cancel = {}

        class _CancelAfterFirstClip(_FakePipeline):
            def _clip_loop(self, video_id):
                for clip_num in range(1, 4):
                    if clip_num == 2:
                        # someone presses Stop in the UI mid-run
                        db2 = CatalogDatabase(Path(worker_db_path))
                        db2.jobs.request_cancel(job_id)
                        db2.close()
                    if self.should_cancel is not None and self.should_cancel():
                        raise PipelineCancelled(video_id)
                    if self.on_progress is not None:
                        self.on_progress({"clip_num": clip_num})

        worker_db_path = self.worker.db_path
        pipeline = _CancelAfterFirstClip()
        self.worker.executor = _FakeExecutor(pipeline)

        self.assertTrue(self.worker.run_one())
        job = self.db.jobs.get(job_id)
        self.assertEqual(job["status"], "cancelled")

    def test_crash_marks_job_failed_with_error(self):
        self._install_fake(behavior="crash")
        job_id = self.db.jobs.create("single", {"video_id": "md_003"})
        self.assertTrue(self.worker.run_one())
        job = self.db.jobs.get(job_id)
        self.assertEqual(job["status"], "failed")
        self.assertIn("ValueError", job["error"])

    def test_cancel_before_claim_is_swept_to_cancelled(self):
        self._install_fake()
        job_id = self.db.jobs.create("single", {"video_id": "md_004"})
        self.db.jobs.request_cancel(job_id)
        # Nothing claimable — but the sweep must finalise the cancel.
        self.assertFalse(self.worker.run_one())
        self.assertEqual(self.db.jobs.get(job_id)["status"], "cancelled")

    def test_unknown_job_type_fails_loudly(self):
        self._install_fake()
        job_id = self.db.jobs.create("teleport", {})
        self.assertTrue(self.worker.run_one())
        self.assertEqual(self.db.jobs.get(job_id)["status"], "failed")
        self.assertIn("unknown job type", self.db.jobs.get(job_id)["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
