"""
Progress reporter — bridges the pipeline's on_progress callback to the
jobs table.

The pipeline calls on_progress(payload) from ITS thread at stage changes
and after every clip; a separate heartbeat thread (own SQLite connection —
sqlite3 connections are not shareable across threads) writes the latest
snapshot + a fresh heartbeat_at every few seconds. Writes are therefore
rate-limited by design: a 200-clip video does not produce 200 UPDATEs.
"""

import threading
from pathlib import Path

from loguru import logger

from vsr_shared.catalog_db import CatalogDatabase


class ProgressReporter:
    """Collects pipeline progress and heartbeats it into the jobs row."""

    def __init__(self, db_path: Path, job_id: int, interval_seconds: float = 3.0):
        self.db_path = Path(db_path)
        self.job_id = job_id
        self.interval_seconds = interval_seconds

        self._latest: dict = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread = None

    # Called from the PIPELINE thread (VSRPipeline.on_progress).
    def on_progress(self, payload: dict) -> None:
        with self._lock:
            self._latest.update(payload)

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._latest)

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name=f"heartbeat-job-{self.job_id}", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds + 2)

    def _run(self) -> None:
        # Own connection — created inside this thread, used only here.
        db = CatalogDatabase(self.db_path)
        try:
            while not self._stop.wait(self.interval_seconds):
                try:
                    db.jobs.heartbeat(self.job_id, progress=self.snapshot())
                except Exception as e:
                    logger.debug(f"heartbeat write failed: {e}")
            # Final snapshot so the finished job shows its last state.
            try:
                db.jobs.heartbeat(self.job_id, progress=self.snapshot())
            except Exception:
                pass
        finally:
            db.close()
