"""
Review routes — curation actions on one segment, DB-backed.

Storage v2 semantics (deliberate change, approved): REJECT deletes the
media files (disk space) but KEEPS the DB row with review_status='rejected'
— statistics stay honest, history complete, and every list/aggregate
already filters rejected out. review_status.json is gone.

Actions (POST /api/segment/{id}/review, body {"action": ...}):
    approve     sign-off → review_status=approved, annotation Conf → 3
    reject      files deleted, row kept as rejected, checkpoint patched
    save        new transcript text (+ WER vs Original, Conf → 3)
    save_words  full word-table rewrite (annotation + words table)
    revert      back to pending (flags cleared)
POST /api/segment/{id}/trim {"start", "end"} re-cuts the media + timings.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path

import logging

from fastapi import APIRouter, Depends, HTTPException

from api.db import get_db
from api.routers.segments import (
    _annotation_file,
    _find_ffmpeg,
    _media_path,
    parse_annotation,
)
from api.settings import PROJECT_ROOT, get_settings
from vsr_shared.catalog_db import CatalogDatabase
from vsr_shared.wer_utils import compute_wer

logger = logging.getLogger("vsr.api.review")

router = APIRouter(prefix="/api", tags=["review"])

_HUMAN_REVIEWED_CONF = 3


# ------------------------------------------------------------------ helpers

def _segment_row(db: CatalogDatabase, segment_id: str) -> dict:
    row = db.connection.execute(
        "SELECT * FROM segments WHERE segment_id = ?", (segment_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Segment not found")
    return dict(row)


def _update_segment(db: CatalogDatabase, segment_id: str, **fields):
    columns = {r[1] for r in db.connection.execute("PRAGMA table_info(segments)")}
    safe = {k: v for k, v in fields.items() if k in columns}
    if not safe:
        return
    assignments = ", ".join(f"{k} = ?" for k in safe)
    with db.connection:
        db.connection.execute(
            f"UPDATE segments SET {assignments} WHERE segment_id = ?",
            (*safe.values(), segment_id))


def _replace_words(db: CatalogDatabase, segment_id: str, words: list):
    with db.connection:
        db.connection.execute(
            "DELETE FROM words WHERE segment_id = ?", (segment_id,))
        db.connection.executemany(
            "INSERT INTO words (segment_id, word_index, word, start_time,"
            " end_time, confidence, asd_score) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(segment_id, i, w["word"], w["start"], w["end"], None,
              float(w.get("asd_score") or w.get("score") or 0))
             for i, w in enumerate(words)])


def _recompute_wer(db: CatalogDatabase, segment_id: str, original, current: str):
    try:
        wer_value, ref_words = compute_wer(original, current)
    except Exception as e:
        logger.warning(f"WER computation failed for {segment_id}: {e}")
        return
    if wer_value is None:
        return
    fields = {"wer": wer_value, "wer_word_count_ref": ref_words}
    if original is not None:
        fields["original_text"] = original
    _update_segment(db, segment_id, **fields)


def _set_conf_in_annotation(anno_path: Path, new_conf: int):
    if not anno_path.exists():
        return
    lines = anno_path.read_text(encoding="utf-8").splitlines()
    anno_path.write_text(
        "\n".join(f"Conf:  {new_conf}" if l.startswith("Conf:") else l
                  for l in lines),
        encoding="utf-8")


def _write_annotation(anno_path: Path, text: str, original, conf: int,
                      words: list):
    lines = [f"Text:  {text}"]
    if original is not None:
        lines.append(f"Original:  {original}")
    lines.append(f"Conf:  {conf}")
    lines.append("WORD START END ASDSCORE")
    for w in words:
        asd = w.get("asd_score") or w.get("score") or 0
        lines.append(
            f"{w['word']} {float(w['start']):.2f} {float(w['end']):.2f}"
            f" {float(asd):.1f}")
    anno_path.write_text("\n".join(lines), encoding="utf-8")


def _recompute_video_validation(db: CatalogDatabase, video_id: str):
    """validated ⟺ every surviving segment is approved (survivors ≠ ∅)."""
    rows = db.connection.execute(
        "SELECT review_status FROM segments WHERE video_id = ?"
        " AND COALESCE(review_status, '') != 'rejected'", (video_id,)).fetchall()
    video = db.videos.get(video_id)
    if video is None or video.get("status") not in ("completed", "validated"):
        return
    new_status = ("validated"
                  if rows and all(r["review_status"] == "approved" for r in rows)
                  else "completed")
    if new_status != video["status"]:
        db.videos.set_status(video_id, new_status)


def _mark_checkpoint_rejected(video_id: str, segment_id: str):
    """Patch the per-video checkpoint so resume skips the rejected clip."""
    cp_path = PROJECT_ROOT / "data" / "clips" / video_id / ".checkpoint.json"
    if not cp_path.exists():
        return
    try:
        checkpoint = json.loads(cp_path.read_text(encoding="utf-8"))
        for info in checkpoint.get("processed_clips", {}).values():
            if info.get("segment_id") == segment_id:
                info["result"] = "rejected"
                info["rejected_at"] = datetime.now().isoformat()
                break
        else:
            return
        tmp = cp_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
        tmp.replace(cp_path)
    except Exception as e:
        logger.warning(f"Checkpoint update for rejected {segment_id} failed: {e}")


def _trim_wav(source: Path, target: Path,
              start_time: float, end_time: float) -> bool:
    """[start, end] of a PCM wav → target, by pure sample math (stdlib)."""
    import wave
    try:
        with wave.open(str(source), "rb") as src:
            rate = src.getframerate()
            total = src.getnframes()
            start_frame = max(0, int(start_time * rate))
            end_frame = min(total, int(end_time * rate))
            if end_frame <= start_frame:
                return False
            src.setpos(start_frame)
            payload = src.readframes(end_frame - start_frame)
            with wave.open(str(target), "wb") as dst:
                dst.setnchannels(src.getnchannels())
                dst.setsampwidth(src.getsampwidth())
                dst.setframerate(rate)
                dst.writeframes(payload)
        return True
    except Exception as e:
        logger.warning(f"Audio trim failed for {target.name}: {e}")
        return False


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ------------------------------------------------------------------ actions

@router.post("/segment/{segment_id}/review")
def review_segment(segment_id: str, body: dict,
                   db: CatalogDatabase = Depends(get_db)):
    action = body.get("action")
    settings = get_settings()
    seg = _segment_row(db, segment_id)
    video_id = seg["video_id"]
    anno_path = _annotation_file(settings, video_id, segment_id)

    if action == "approve":
        _update_segment(db, segment_id,
                        review_status="approved", reviewed_at=_now())
        _set_conf_in_annotation(anno_path, _HUMAN_REVIEWED_CONF)
        _recompute_video_validation(db, video_id)
        return {"ok": True, "status": "approved"}

    if action == "reject":
        for path in [
            _media_path(settings, video_id, segment_id, "face"),
            _media_path(settings, video_id, segment_id, "mouth"),
            settings.processed_dir / video_id / "audio" / f"{segment_id}.wav",
            anno_path,
        ]:
            try:
                path.unlink(missing_ok=True)
            except OSError as e:
                logger.warning(f"Reject cleanup error for {path.name}: {e}")
        _update_segment(db, segment_id,
                        review_status="rejected", reviewed_at=_now())
        _mark_checkpoint_rejected(video_id, segment_id)
        _recompute_video_validation(db, video_id)
        return {"ok": True, "status": "rejected"}

    if action == "save":
        new_text = (body.get("text") or "").strip()
        if not new_text:
            raise HTTPException(400, "text is required")
        _update_segment(db, segment_id, text=new_text, transcript_edited=1)
        original = None
        if anno_path.exists():
            lines = anno_path.read_text(encoding="utf-8").splitlines()
            anno_path.write_text(
                "\n".join(f"Text:  {new_text}" if l.startswith("Text:") else l
                          for l in lines),
                encoding="utf-8")
            original = parse_annotation(anno_path).get("original")
            _set_conf_in_annotation(anno_path, _HUMAN_REVIEWED_CONF)
        _recompute_wer(db, segment_id, original, new_text)
        return {"ok": True}

    if action == "save_words":
        words = body.get("words", [])
        new_text = (body.get("text") or "").strip()
        if not anno_path.exists():
            raise HTTPException(404, "Annotation not found")
        parsed = parse_annotation(anno_path)
        if not new_text:
            new_text = parsed["text"]
        original = parsed.get("original")
        _write_annotation(anno_path, new_text, original,
                          _HUMAN_REVIEWED_CONF, words)
        _replace_words(db, segment_id, words)
        _update_segment(db, segment_id, text=new_text,
                        num_words=len(words), transcript_edited=1)
        _recompute_wer(db, segment_id, original, new_text)
        return {"ok": True}

    if action == "revert":
        _update_segment(db, segment_id, review_status="", reviewed_at="",
                        transcript_edited=0)
        _recompute_video_validation(db, video_id)
        return {"ok": True, "status": "pending"}

    raise HTTPException(400, "Unknown action")


# -------------------------------------------------------------------- trim

@router.post("/segment/{segment_id}/trim")
def trim_segment(segment_id: str, body: dict,
                 db: CatalogDatabase = Depends(get_db)):
    """Re-cut the segment to [start, end] (seconds, clip-relative)."""
    t_in = float(body.get("start", 0))
    t_out = float(body.get("end", 0))
    if t_out <= t_in or t_in < 0:
        raise HTTPException(400, "Invalid trim range")

    settings = get_settings()
    seg = _segment_row(db, segment_id)
    video_id = seg["video_id"]
    duration = round(t_out - t_in, 3)
    ffmpeg = _find_ffmpeg()

    for crop in ("face", "mouth"):
        src = _media_path(settings, video_id, segment_id, crop)
        if not src.exists():
            continue
        tmp = src.with_suffix(".trimming.mp4")
        try:
            subprocess.run(
                [ffmpeg, "-y", "-ss", str(t_in), "-t", str(duration),
                 "-i", str(src),
                 "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                 "-c:a", "aac", "-ar", "16000", "-ac", "1",
                 str(tmp), "-loglevel", "error"],
                check=True, capture_output=True, timeout=120)
            tmp.replace(src)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            tmp.unlink(missing_ok=True)
            logger.warning(f"Trim failed for {crop}/{segment_id}: {e}")

    # Segment audio artifact — sample-exact trim (stdlib wave, same math
    # as the pipeline's audio_slicer; inlined so the API stays torch-free
    # and loguru-free).
    wav = settings.processed_dir / video_id / "audio" / f"{segment_id}.wav"
    if wav.exists():
        tmp_wav = wav.with_suffix(".trimming.wav")
        if _trim_wav(wav, tmp_wav, t_in, t_out):
            tmp_wav.replace(wav)
        else:
            tmp_wav.unlink(missing_ok=True)

    anno_path = _annotation_file(settings, video_id, segment_id)
    if anno_path.exists():
        parsed = parse_annotation(anno_path)
        kept = []
        for w in parsed["words"]:
            ws = round(w["start"] - t_in, 3)
            we = round(w["end"] - t_in, 3)
            if we <= 0 or ws >= duration:
                continue
            kept.append({**w, "start": max(0.0, ws), "end": min(duration, we)})
        original = parsed.get("original")
        new_text = " ".join(w["word"] for w in kept) if kept else parsed["text"]
        _write_annotation(anno_path, new_text, original,
                          parsed.get("conf", 2), kept)
        _replace_words(db, segment_id, kept)
        _update_segment(db, segment_id, duration=duration,
                        num_words=len(kept), text=new_text, trimmed=1)
        _recompute_wer(db, segment_id, original, new_text)
    else:
        _update_segment(db, segment_id, duration=duration, trimmed=1)

    return {"ok": True, "duration": duration}
