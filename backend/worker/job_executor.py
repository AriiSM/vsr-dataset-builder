"""
Job executor — maps a claimed job onto the EXISTING VSRPipeline methods.

Job types and their params_json:
    single       {"video_id", "youtube_url", "verify_cc"?}
    batch        {"status"?: [..], "video_ids"?: [..], "limit"?}
    resume       {"video_id", "youtube_url"?}
    resume_batch {"video_ids"?: [..], "limit"?}
    sync         {"video_ids"?: [..]}

Config is re-read at EVERY job (change thresholds without restarting the
worker); the pipeline instance — and therefore every loaded model — is
reused across jobs as long as config.yaml is unchanged (hash-keyed cache),
so models stay warm between consecutive jobs.
"""

import hashlib
from pathlib import Path
from typing import Optional

from loguru import logger

from orchestrator.pipeline import VSRPipeline
from orchestrator.pipeline_config import PipelineConfig
from orchestrator.processing_results import PipelineCancelled


def config_hash(config_path: Path) -> str:
    """sha256 of the config file — provenance + pipeline-cache key."""
    try:
        return hashlib.sha256(Path(config_path).read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


class JobExecutor:
    """Executes one job at a time on a warm, config-hash-cached pipeline."""

    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self._pipeline: Optional[VSRPipeline] = None
        self._pipeline_config_hash: str = ""

    def _get_pipeline(self) -> VSRPipeline:
        current = config_hash(self.config_path)
        if self._pipeline is None or current != self._pipeline_config_hash:
            if self._pipeline is not None:
                logger.info("config.yaml changed — rebuilding pipeline (models reload)")
            self._pipeline = VSRPipeline(PipelineConfig.from_yaml(self.config_path))
            self._pipeline_config_hash = current
        return self._pipeline

    def execute(self, job: dict, params: dict,
                on_progress, should_cancel) -> dict:
        """Run one claimed job. Returns {"status": done|cancelled|failed,
        "error": str} — the worker writes it back to the jobs row."""
        pipeline = self._get_pipeline()
        pipeline.on_progress = on_progress
        pipeline.should_cancel = should_cancel
        self._apply_cookie_overrides(pipeline, params)
        try:
            handler = {
                "single": self._run_single,
                "batch": self._run_batch,
                "resume": self._run_resume,
                "resume_batch": self._run_resume_batch,
                "sync": self._run_sync,
                "bulk_import": self._run_bulk_import,
            }.get(job["type"])
            if handler is None:
                return {"status": "failed",
                        "error": f"unknown job type: {job['type']}"}
            handler(pipeline, params)
            return {"status": "done", "error": ""}
        except PipelineCancelled as e:
            logger.info(f"Job {job['id']} cancelled cleanly ({e.video_id} resumable)")
            return {"status": "cancelled", "error": ""}
        except Exception as e:
            logger.exception(f"Job {job['id']} failed")
            return {"status": "failed", "error": f"{type(e).__name__}: {e}"}
        finally:
            pipeline.on_progress = None
            pipeline.should_cancel = None

    # ------------------------------------------------------------- handlers

    @staticmethod
    def _apply_cookie_overrides(pipeline: VSRPipeline, params: dict):
        """UI-provided YouTube cookie settings for this job only."""
        cookies = (params.get("cookies") or "").strip()
        browser = (params.get("cookies_from_browser") or "").strip()
        if cookies or browser:
            if cookies:
                pipeline.config.cookies_file = cookies
            if browser:
                pipeline.config.cookies_from_browser = browser
            pipeline.services.invalidate_downloader()

    @staticmethod
    def _master_csv(pipeline: VSRPipeline) -> None:
        """Storage v2 final: selection reads dataset.db — no CSV path."""
        return None

    def _run_single(self, pipeline: VSRPipeline, params: dict):
        pipeline.process_video(
            video_id=params["video_id"],
            youtube_url=params.get("youtube_url", ""),
            verify_cc=bool(params.get("verify_cc", True)),
        )

    def _run_batch(self, pipeline: VSRPipeline, params: dict):
        pipeline.process_batch(
            self._master_csv(pipeline),
            status_filter=params.get("status") or ["pending"],
            limit=params.get("limit"),
            video_ids=params.get("video_ids"),
        )

    def _run_resume(self, pipeline: VSRPipeline, params: dict):
        video_id = params["video_id"]
        url = params.get("youtube_url", "")
        if not url:
            video = pipeline.catalog.db.videos.get(video_id) or {}
            url = video.get("youtube_url", "")
        pipeline.resume_video(video_id, url)

    def _run_resume_batch(self, pipeline: VSRPipeline, params: dict):
        pipeline.process_batch_resume(
            self._master_csv(pipeline),
            limit=params.get("limit"),
            video_ids=params.get("video_ids"),
        )

    def _run_sync(self, pipeline: VSRPipeline, params: dict):
        pipeline.sync_excel_from_disk(
            self._master_csv(pipeline),
            video_ids=params.get("video_ids"),
        )

    def _run_bulk_import(self, pipeline: VSRPipeline, params: dict):
        """Seed videos: download a URL list + register rows (CLI bulk-import).

        Runs the existing CLI in a subprocess (it owns the seeding logic);
        output streams into the job log line by line.
        """
        import subprocess
        import sys as _sys

        urls = [u for u in (params.get("urls") or []) if u.strip()]
        if not urls:
            raise ValueError("bulk_import: no URLs provided")

        cli = Path(__file__).resolve().parent.parent / "orchestrator" / "cli.py"
        cmd = [
            _sys.executable, str(cli), "bulk-import",
            "--prefix", (params.get("prefix") or "vid").strip() or "vid",
            "--region", (params.get("region") or "UNKNOWN").strip() or "UNKNOWN",
            "--source", (params.get("source") or "YouTube_CC").strip() or "YouTube_CC",
            "--urls", *urls,
        ]
        if params.get("no_cc_check"):
            cmd.append("--no-cc-check")
        if params.get("pre_downloaded"):
            cmd.append("--pre-downloaded")
        if (params.get("cookies") or "").strip():
            cmd += ["--cookies", params["cookies"].strip()]
        if (params.get("cookies_from_browser") or "").strip():
            cmd += ["--cookies-from-browser", params["cookies_from_browser"].strip()]

        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            cwd=str(cli.parent.parent.parent),
        )
        try:
            for line in process.stdout:
                logger.info(line.rstrip())
                if pipeline.should_cancel is not None and pipeline.should_cancel():
                    process.terminate()
                    from orchestrator.processing_results import PipelineCancelled
                    raise PipelineCancelled("bulk_import")
            process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
        if process.returncode not in (0, None):
            raise RuntimeError(f"bulk-import exited with {process.returncode}")
