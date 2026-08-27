"""
Jobs routes — the API side of the API↔worker queue.

New REST surface:
    POST   /api/jobs                {type, params}    → {id}
    GET    /api/jobs                                  → history (newest first)
    GET    /api/jobs/{id}                             → row + parsed progress
    GET    /api/jobs/{id}/log?offset=                 → incremental log tail
    POST   /api/jobs/{id}/cancel                      → cooperative stop

Compatibility surface (exact Flask contract — the React Process tab runs
unchanged): POST /api/start · GET /api/status · POST /api/stop ·
POST /api/bulk_import. These adapt the old bodies onto the queue.

The API never executes anything: it INSERTs rows; the worker (its own
process, possibly its own container) claims and runs them. A job survives
a restart of either side.
"""

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query

from api.db import get_db
from api.settings import PROJECT_ROOT, get_settings
from vsr_shared.catalog_db import CatalogDatabase

router = APIRouter(prefix="/api", tags=["jobs"])

_JOB_TYPES = ("single", "batch", "resume", "resume_batch", "sync", "bulk_import")
_ACTIVE_STATUSES = ("pending", "running", "cancel_requested")


def _parse(job: dict) -> dict:
    job = dict(job)
    for key in ("params_json", "progress_json"):
        try:
            job[key.replace("_json", "")] = json.loads(job.pop(key) or "{}")
        except (json.JSONDecodeError, TypeError):
            job[key.replace("_json", "")] = {}
    return job


def _active_job(db: CatalogDatabase) -> dict:
    for job in db.jobs.all(limit=20):
        if job["status"] in _ACTIVE_STATUSES:
            return job
    return None


def _read_log(job: dict, offset: int = 0, max_bytes: int = 512 * 1024):
    """Incremental read of the job's log file from a byte offset."""
    raw_path = job.get("log_path") or ""
    if not raw_path:
        return "", offset
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return "", offset
    size = path.stat().st_size
    if offset >= size:
        return "", size
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(offset)
        chunk = f.read(max_bytes)
    return chunk, offset + len(chunk.encode("utf-8", errors="replace"))


# ------------------------------------------------------------- REST surface

@router.post("/jobs", status_code=201)
def create_job(body: dict, db: CatalogDatabase = Depends(get_db)):
    job_type = body.get("type", "")
    if job_type not in _JOB_TYPES:
        raise HTTPException(400, f"type must be one of {_JOB_TYPES}")
    active = _active_job(db)
    if active is not None:
        raise HTTPException(
            409, f"job {active['id']} is already {active['status']} — "
                 "one job at a time (single GPU)")
    job_id = db.jobs.create(job_type, body.get("params") or {},
                            git_sha=get_settings().git_sha)
    return {"id": job_id, "status": "pending"}


@router.get("/jobs")
def list_jobs(limit: int = Query(50, ge=1, le=500),
              db: CatalogDatabase = Depends(get_db)):
    return {"jobs": [_parse(j) for j in db.jobs.all(limit=limit)]}


@router.get("/jobs/{job_id}")
def get_job(job_id: int, db: CatalogDatabase = Depends(get_db)):
    job = db.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return _parse(job)


@router.get("/jobs/{job_id}/log")
def job_log(job_id: int, offset: int = Query(0, ge=0),
            db: CatalogDatabase = Depends(get_db)):
    job = db.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    chunk, new_offset = _read_log(job, offset)
    return {"log": chunk, "offset": new_offset, "status": job["status"]}


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: int, db: CatalogDatabase = Depends(get_db)):
    job = db.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if not db.jobs.request_cancel(job_id):
        raise HTTPException(409, f"job is {job['status']} — nothing to cancel")
    return {"ok": True, "id": job_id}


# ---------------------------------------------- Flask-compatible surface

def _normalise_ids(raw) -> list:
    if isinstance(raw, str):
        raw = raw.replace(",", " ").split()
    return [v.strip() for v in (raw or []) if v and v.strip()]


@router.post("/start")
def compat_start(body: dict, db: CatalogDatabase = Depends(get_db)):
    """Old contract: {mode, limit, video_id, url, video_ids, cookies...}."""
    active = _active_job(db)
    if active is not None:
        raise HTTPException(409, "Pipeline is already running")

    mode = body.get("mode", "batch-pending")
    params = {
        k: v for k, v in {
            "limit": body.get("limit"),
            "video_ids": _normalise_ids(body.get("video_ids")) or None,
            "cookies": (body.get("cookies") or "").strip() or None,
            "cookies_from_browser":
                (body.get("cookies_from_browser") or "").strip() or None,
        }.items() if v
    }

    if mode == "single":
        video_id = (body.get("video_id") or "").strip()
        url = (body.get("url") or "").strip()
        if not video_id or not url:
            raise HTTPException(400, "video_id and url are required")
        job_type = "single"
        params.update({"video_id": video_id, "youtube_url": url})
    elif mode == "resume":
        job_type = "resume_batch"
    else:  # batch-pending | batch-failed
        job_type = "batch"
        params["status"] = ["failed" if mode == "batch-failed" else "pending"]

    job_id = db.jobs.create(job_type, params, git_sha=get_settings().git_sha)
    return {"status": "started", "mode": mode, "job_id": job_id}


@router.post("/bulk_import")
def compat_bulk_import(body: dict, db: CatalogDatabase = Depends(get_db)):
    if _active_job(db) is not None:
        raise HTTPException(409, "A pipeline or import is already running")
    urls_raw = body.get("urls") or []
    if isinstance(urls_raw, str):
        urls_raw = urls_raw.replace(",", "\n").splitlines()
    urls = [u.strip() for u in urls_raw
            if u and u.strip() and not u.strip().startswith("#")]
    if not urls:
        raise HTTPException(400, "No URLs provided")
    params = {
        "urls": urls,
        "prefix": body.get("prefix") or "vid",
        "region": body.get("region") or "UNKNOWN",
        "source": body.get("source") or "YouTube_CC",
        "no_cc_check": bool(body.get("no_cc_check", False)),
        "cookies": (body.get("cookies") or "").strip(),
        "cookies_from_browser": (body.get("cookies_from_browser") or "").strip(),
    }
    job_id = db.jobs.create("bulk_import", params,
                            git_sha=get_settings().git_sha)
    return {"status": "started", "mode": "bulk-import",
            "urls": len(urls), "job_id": job_id}


@router.get("/status")
def compat_status(db: CatalogDatabase = Depends(get_db)):
    """Old contract: {running, mode, started_at, log: [lines]}."""
    jobs = db.jobs.all(limit=1)
    if not jobs:
        return {"running": False, "mode": None, "started_at": None, "log": []}
    job = jobs[0]
    running = job["status"] in _ACTIVE_STATUSES

    chunk, _ = _read_log(job, offset=0)
    log_lines = chunk.splitlines()[-300:]
    if not running and job["status"] in ("failed", "interrupted") and job["error"]:
        log_lines.append(f"[{job['status']}] {job['error']}")

    params = _parse(job)["params"]
    mode = job["type"]
    if mode == "batch":
        statuses = params.get("status") or ["pending"]
        mode = f"batch-{statuses[0]}"
    return {
        "running": running,
        "mode": mode,
        "started_at": job["started_at"] or job["created_at"],
        "log": log_lines,
        "job_id": job["id"],
        "job_status": job["status"],
        "progress": _parse(job)["progress"],
    }


@router.post("/stop")
def compat_stop(db: CatalogDatabase = Depends(get_db)):
    active = _active_job(db)
    if active is None:
        raise HTTPException(409, "No pipeline running")
    db.jobs.request_cancel(active["id"])
    return {"ok": True, "job_id": active["id"]}
