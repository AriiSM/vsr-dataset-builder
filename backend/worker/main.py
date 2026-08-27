"""
Worker — the queue consumer that runs the pipeline.

The only long-lived GPU process. Talks EXCLUSIVELY to the jobs table of
data/catalog/dataset.db (no HTTP): the API inserts jobs, this loop claims
them atomically, executes them in-process through the existing VSRPipeline,
and writes status + progress + heartbeat back. Kill it any time — on the
next start it marks stale 'running' jobs as 'interrupted' (resumable) and
carries on.

Per job:
    claim (atomic) → stamp git_sha/config_hash/log_path → attach a loguru
    sink to data/logs/job_{id}.log → heartbeat thread (3 s) → execute →
    final status (done | cancelled | failed).

Run natively (pilot) or as the vsr-worker container entrypoint:
    python backend/worker/main.py --config config/config.yaml
"""

import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _BACKEND_DIR.parent
sys.path[:0] = [str(_BACKEND_DIR), str(_BACKEND_DIR / "shared")]

import yaml  # noqa: E402
from loguru import logger  # noqa: E402

# Cache redirection BEFORE any ML import — every model lands under models/.
from vsr_shared.model_env import apply_model_env  # noqa: E402
apply_model_env(_PROJECT_ROOT / "models")

from vsr_shared.catalog_db import CatalogDatabase  # noqa: E402
from worker.job_executor import JobExecutor, config_hash  # noqa: E402
from worker.progress_reporter import ProgressReporter  # noqa: E402

_STALE_HEARTBEAT_SECONDS = 60


def _git_sha() -> str:
    """Current commit (baked as env GIT_SHA in Docker; git locally)."""
    env_sha = os.environ.get("GIT_SHA", "")
    if env_sha:
        return env_sha
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=_PROJECT_ROOT, timeout=5,
        ).stdout.strip()
    except Exception:
        return ""


def _catalog_db_path(config_path: Path) -> Path:
    cfg = yaml.safe_load(config_path.read_text()) or {}
    paths = cfg.get("paths", {})
    catalog_dir = Path(paths.get(
        "catalog_dir", Path(paths.get("base_dir", "./data")) / "catalog"))
    if not catalog_dir.is_absolute():
        catalog_dir = _PROJECT_ROOT / catalog_dir
    return catalog_dir / "dataset.db"


class Worker:
    """Poll → claim → execute → report, forever (or --once for tests)."""

    def __init__(self, config_path: Path, poll_seconds: float = 1.0,
                 worker_name: str = ""):
        self.config_path = config_path
        self.poll_seconds = poll_seconds
        self.worker_name = worker_name or f"{platform.node()}-{os.getpid()}"

        self.db_path = _catalog_db_path(config_path)
        self.db = CatalogDatabase(self.db_path)
        self.executor = JobExecutor(config_path)
        self.logs_dir = self.db_path.parent.parent / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def startup_recovery(self) -> None:
        stale = self.db.jobs.recover_stale(_STALE_HEARTBEAT_SECONDS)
        if stale:
            logger.warning(
                f"Recovered {len(stale)} stale job(s) as interrupted: {stale}")

    def run_forever(self) -> None:
        logger.info(
            f"Worker '{self.worker_name}' online — queue: {self.db_path}")
        self.startup_recovery()
        while True:
            if not self.run_one():
                time.sleep(self.poll_seconds)

    def run_one(self) -> bool:
        """Claim and execute at most one job. Returns True if one ran."""
        self.db.jobs.cancel_unclaimed()
        job = self.db.jobs.claim_next(self.worker_name)
        if job is None:
            return False

        job_id = job["id"]
        log_path = self.logs_dir / f"job_{job_id}.log"
        self.db.jobs.stamp_execution(
            job_id, _git_sha(), config_hash(self.config_path),
            str(log_path.relative_to(_PROJECT_ROOT))
            if log_path.is_relative_to(_PROJECT_ROOT) else str(log_path),
        )

        sink_id = logger.add(log_path, level="INFO", enqueue=True,
                             backtrace=False, diagnose=False)
        reporter = ProgressReporter(self.db_path, job_id)
        reporter.start()
        logger.info(f"Job {job_id} ({job['type']}) started by {self.worker_name}")

        try:
            import json
            params = json.loads(job.get("params_json") or "{}")
            outcome = self.executor.execute(
                job, params,
                on_progress=reporter.on_progress,
                should_cancel=lambda: self.db.jobs.is_cancel_requested(job_id),
            )
        finally:
            reporter.stop()
            logger.remove(sink_id)

        self.db.jobs.finish(job_id, outcome["status"], error=outcome["error"])
        logger.info(f"Job {job_id} finished: {outcome['status']}")
        return True


def main():
    parser = argparse.ArgumentParser(description="VSR pipeline queue worker")
    parser.add_argument("--config", type=Path,
                        default=_PROJECT_ROOT / "config" / "config.yaml")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--worker-name", default="")
    parser.add_argument("--once", action="store_true",
                        help="run at most one job, then exit (smoke tests)")
    args = parser.parse_args()

    worker = Worker(args.config, args.poll_seconds, args.worker_name)
    if args.once:
        worker.startup_recovery()
        ran = worker.run_one()
        print("ran one job" if ran else "queue empty")
    else:
        worker.run_forever()


if __name__ == "__main__":
    main()
