"""
Segments routes — Explorer list, per-segment detail, review-status map,
media streaming, manual speaker reassignment.

Shapes mirror the Flask endpoints; rejected segments are excluded from the
list (parity with the CSV-removal era) but still addressable by id.
"""

import math
import subprocess
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from api.db import get_db
from api.settings import get_settings
from vsr_shared.catalog_db import CatalogDatabase

router = APIRouter(prefix="/api", tags=["segments"])

# Columns the Explorer grid consumes (payload stays small at ~12k rows).
_EXPLORER_COLUMNS = (
    "segment_id", "video_id", "duration", "text", "num_words",
    "asd_score", "syncnet_conf", "whisper_conf",
    "speaker_id", "original_text", "wer",
    "quality_tier", "mouth_roi_method", "mouth_landmark_fail_rate",
    "needs_review", "wer_medium_vs_large",
)


def _scrub(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str) and value == "nan":
        return None
    return value


def _media_path(settings, video_id: str, segment_id: str,
                crop: str = "face") -> Path:
    subdir = "mouth_crop" if crop == "mouth" else "face_crop"
    return settings.processed_dir / video_id / subdir / f"{segment_id}.mp4"


def _annotation_file(settings, video_id: str, segment_id: str) -> Path:
    return settings.processed_dir / video_id / "text" / f"{segment_id}.txt"


def parse_annotation(path: Path) -> dict:
    """LRS2 annotation → {text, original, conf, words} (Flask shape)."""
    result: dict = {"text": "", "original": None, "conf": 2, "words": []}
    in_words = False
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        if line.startswith("Original:"):
            result["original"] = line[len("Original:"):].strip()
        elif line.startswith("Text:"):
            result["text"] = line[len("Text:"):].strip()
        elif line.startswith("Conf:"):
            try:
                result["conf"] = int(line[len("Conf:"):].strip())
            except ValueError:
                pass
        elif line.startswith("WORD START END"):
            in_words = True
        elif in_words and line.strip():
            parts = line.split()
            if len(parts) >= 4:
                result["words"].append({
                    "word": parts[0], "start": float(parts[1]),
                    "end": float(parts[2]), "score": float(parts[3]),
                })
    return result


@router.get("/segments")
def segments_list(db: CatalogDatabase = Depends(get_db)):
    columns = ", ".join(f"s.{c}" for c in _EXPLORER_COLUMNS)
    rows = db.connection.execute(
        f"SELECT {columns},"
        " COALESCE(NULLIF(v.region, ''), 'UNKNOWN') AS region"
        " FROM segments s LEFT JOIN videos v USING (video_id)"
        " WHERE COALESCE(s.review_status, '') != 'rejected'"
        " ORDER BY s.video_id, s.segment_id"
    ).fetchall()
    records = [{k: _scrub(v) for k, v in dict(r).items()} for r in rows]
    return {"segments": records, "total": len(records)}


@router.get("/review_status")
def review_status(db: CatalogDatabase = Depends(get_db)):
    """{segment_id: {status, transcript_edited?, trimmed?}} from the DB."""
    result = {}
    for row in db.connection.execute(
            "SELECT segment_id, review_status, transcript_edited, trimmed"
            " FROM segments WHERE COALESCE(review_status, '') != ''"
            " OR transcript_edited = 1 OR trimmed = 1"):
        entry: dict = {"status": row["review_status"] or "pending"}
        if row["transcript_edited"]:
            entry["transcript_edited"] = True
        if row["trimmed"]:
            entry["trimmed"] = True
        result[row["segment_id"]] = entry
    return result


@router.get("/segment/{segment_id}")
def segment_detail(segment_id: str, db: CatalogDatabase = Depends(get_db)):
    settings = get_settings()
    row = db.connection.execute(
        "SELECT * FROM segments WHERE segment_id = ?", (segment_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Segment not found")

    seg = {k: ("" if v is None else str(v)) for k, v in dict(row).items()}
    video_id = seg.get("video_id", "")
    video_path = _media_path(settings, video_id, segment_id)
    anno_path = _annotation_file(settings, video_id, segment_id)

    seg["video_path"] = str(video_path)
    seg["annotation_path"] = str(anno_path)
    seg["has_video"] = video_path.exists()
    seg["has_annotation"] = anno_path.exists()
    if anno_path.exists():
        seg["annotation_raw"] = anno_path.read_text(encoding="utf-8")
        seg["annotation"] = parse_annotation(anno_path)

    seg["review"] = {"status": row["review_status"] or "pending"}
    if row["transcript_edited"]:
        seg["review"]["transcript_edited"] = True
    if row["trimmed"]:
        seg["review"]["trimmed"] = True
    return seg


@router.post("/segment/{segment_id}/speaker")
def set_segment_speaker(segment_id: str, body: dict,
                        db: CatalogDatabase = Depends(get_db)):
    """Manual re-assignment when clustering picked the wrong person."""
    new_sid = (body.get("speaker_id") or "").strip()
    if not new_sid:
        raise HTTPException(400, "speaker_id is required")
    row = db.connection.execute(
        "SELECT speaker_id FROM segments WHERE segment_id = ?",
        (segment_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "segment not found")

    old_sid = row["speaker_id"] or ""
    with db.connection:
        db.connection.execute(
            "UPDATE segments SET speaker_id = ? WHERE segment_id = ?",
            (new_sid, segment_id))
    db.speakers.ensure_exists(new_sid)
    return {"ok": True, "segment_id": segment_id,
            "old_speaker_id": old_sid, "speaker_id": new_sid}


# ------------------------------------------------------------------- media

def _find_ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


@router.get("/media/{video_id}/{segment_id}")
def media(video_id: str, segment_id: str,
          type: str = Query("face", pattern="^(face|mouth)$")):
    """Browser-playable H.264 with seek support (Range via FileResponse).

    Transcoded once to a disk cache keyed by source mtime — same strategy
    the Flask app used; FileResponse adds Accept-Ranges/Content-Length so
    <video> seeks work.
    """
    settings = get_settings()
    source = _media_path(settings, video_id, segment_id, type)
    if not source.exists():
        raise HTTPException(404, "File not found")

    cache_dir = settings.cache_dir / "h264"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{source.parent.name}__{video_id}__{source.stem}.mp4"

    if (not cached.exists()
            or cached.stat().st_mtime < source.stat().st_mtime
            or cached.stat().st_size == 0):
        try:
            subprocess.run(
                [_find_ffmpeg(), "-y", "-i", str(source),
                 "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                 "-loglevel", "error", str(cached)],
                check=True, timeout=60,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            raise HTTPException(500, "transcode failed")
    return FileResponse(cached, media_type="video/mp4")
