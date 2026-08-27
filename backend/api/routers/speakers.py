"""
Speakers routes — registry with live aggregates, curator edits, thumbnails.

Aggregates come from the speaker_stats VIEW (always fresh — the Flask app
had to recompute them from CSV on every request). Thumbnails keep the
disk-cache + mid-frame-ffmpeg strategy.
"""

import subprocess
import time

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from api.db import get_db
from api.routers.segments import _find_ffmpeg, _media_path
from api.settings import get_settings
from vsr_shared.catalog_db import CatalogDatabase

router = APIRouter(prefix="/api", tags=["speakers"])

_EDITABLE = ("speaker_name", "gender", "age_group", "accent_region")


@router.get("/speakers")
def speakers(db: CatalogDatabase = Depends(get_db)):
    records = []
    for row in db.speakers.all_with_stats():
        row.pop("centroid", None)
        records.append({
            k: ("" if v is None or v == "" else str(v)) for k, v in row.items()
        })
    return {"speakers": records}


@router.post("/speaker/{speaker_id}")
def update_speaker(speaker_id: str, body: dict,
                   db: CatalogDatabase = Depends(get_db)):
    if not speaker_id:
        raise HTTPException(400, "speaker_id is required")
    editable = {k: body[k] for k in _EDITABLE if k in body}
    db.speakers.upsert(speaker_id, editable)
    return {"ok": True, "speaker_id": speaker_id, "fields": editable}


@router.get("/speaker/{speaker_id}/thumbnail")
def speaker_thumbnail(speaker_id: str, seg: int = Query(0, ge=0),
                      db: CatalogDatabase = Depends(get_db)):
    """Mid-frame JPEG of one of the speaker's face_crop videos (cached)."""
    if "/" in speaker_id or "\\" in speaker_id or ".." in speaker_id:
        raise HTTPException(400, "invalid speaker_id")

    settings = get_settings()
    thumb_dir = settings.cache_dir / "speaker_thumbs"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    cached = thumb_dir / f"{speaker_id}_seg{seg}.jpg"
    if cached.exists() and (time.time() - cached.stat().st_mtime) < 30 * 24 * 3600:
        return FileResponse(cached, media_type="image/jpeg")

    rows = db.connection.execute(
        "SELECT segment_id, video_id FROM segments WHERE speaker_id = ?"
        " AND COALESCE(review_status, '') != 'rejected'"
        " ORDER BY segment_id", (speaker_id,)).fetchall()
    candidates = [
        path for r in rows
        if (path := _media_path(settings, r["video_id"], r["segment_id"])).exists()
    ]
    if not candidates:
        raise HTTPException(404, "no face_crop video on disk for this speaker")

    n = len(candidates)
    if seg <= 0 or n == 1:
        chosen = candidates[0]
    elif seg >= n:
        chosen = candidates[-1]
    else:
        step = max(1, n // 4)
        chosen = candidates[min(n - 1, seg * step)]

    ffmpeg = _find_ffmpeg()
    ffprobe = ffmpeg.replace("ffmpeg", "ffprobe")
    try:
        duration = float(subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(chosen)],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip() or 1.0)
    except Exception:
        duration = 1.0
    try:
        subprocess.run(
            [ffmpeg, "-y", "-ss", f"{max(0.0, duration / 2):.3f}",
             "-i", str(chosen), "-vframes", "1", "-q:v", "2",
             str(cached), "-loglevel", "error"],
            check=True, timeout=15,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        raise HTTPException(500, "ffmpeg failed")
    if not cached.exists() or cached.stat().st_size == 0:
        raise HTTPException(500, "thumbnail extraction produced empty file")
    return FileResponse(cached, media_type="image/jpeg")
