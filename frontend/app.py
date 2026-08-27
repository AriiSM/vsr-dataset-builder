"""
DEPRECATED — Flask frontend (storage v1 era).

Superseded by the FastAPI backend + queue worker:
    uvicorn api.main:app --app-dir backend --port 8000
    python backend/worker/main.py
The FastAPI app serves the SAME React build and the SAME /api/*
contract (compat layer), backed by dataset.db + the jobs queue.
This file is kept ONLY as a fallback until UI parity is confirmed
on the processing machine — then it gets deleted.

(Original description follows.)

Flask frontend for the Romanian VSR Dataset Pipeline.

Run from the project root:
    python frontend/app.py

Three tabs:
  - Process: select mode, run pipeline, watch live logs
  - Stats: dataset overview (videos_master.csv + segments_index.csv)
  - Explorer: browse exported segments with video preview + annotation
"""

import functools
import gzip
import json
import logging
import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from flask import Flask, Response, jsonify, request, send_file, send_from_directory, stream_with_context

# Make `src.*` imports work when this app is launched from anywhere.
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Paths ──────────────────────────────────────────────────────────────────
_FRONTEND_DIR = Path(__file__).parent
_PROJECT_ROOT = _FRONTEND_DIR.parent
_CONFIG_PATH  = _PROJECT_ROOT / "config" / "config.yaml"
# Storage v2: metadata moved to data/catalog (dataset.db + CSV mirrors)
_METADATA_DIR = _PROJECT_ROOT / "data" / "catalog"
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
_ANNOTATIONS_DIR = _PROJECT_ROOT / "data" / "annotations"   # legacy (storage v1)


def _annotation_path(video_id: str, segment_id: str):
    """Storage v2 puts annotations at processed/{vid}/text/; fall back to the
    legacy data/annotations/{vid}/ layout for segments exported before 6.5."""
    v2 = _PROCESSED_DIR / video_id / "text" / f"{segment_id}.txt"
    if v2.exists():
        return v2
    legacy = _ANNOTATIONS_DIR / video_id / f"{segment_id}.txt"
    return legacy if legacy.exists() else v2


_SCRIPTS_DIR  = _PROJECT_ROOT / "backend" / "orchestrator"

# Use the project's own venv python if present, otherwise fall back to the
# Python that is currently running (useful when launched inside the venv).
_VENV_PYTHON = _PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
PYTHON = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable


def _get_ffmpeg() -> str:
    """Return the path to an ffmpeg binary (prefers imageio-ffmpeg bundle)."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


FFMPEG = _get_ffmpeg()

# ── Flask app ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("werkzeug").setLevel(logging.WARNING)

app = Flask(
    __name__,
    static_folder=None,  # the React UI (dist/) is served by the routes below
)

# Production build of the React frontend (created with `npm run build`).
_REACT_DIST = _FRONTEND_DIR / "dist"

# Serialises every metadata write (review_status.json + segments_index.csv)
# within this process, so two browser windows doing review actions at the
# same time cannot interleave their read-modify-write cycles. RLock because
# the locked handlers call locked helpers (update_segment etc.).
_METADATA_WRITE_LOCK = threading.RLock()

# Caches for the two expensive disk scans, keyed on file mtimes so any
# actual data change invalidates them:
#  - Conf-level distribution (reads up to 3k annotation .txt files)
#  - vocabulary (reads EVERY annotation file for per-word durations)
_conf_dist_cache: dict = {"key": None, "dist": None}
_vocab_cache: dict = {"key": None, "payload": None}


def _with_metadata_lock(fn):
    """Route decorator: run the whole handler under the metadata write lock."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _METADATA_WRITE_LOCK:
            return fn(*args, **kwargs)
    return wrapper


@app.after_request
def _gzip_json_responses(response):
    """Gzip large JSON responses (e.g. /api/segments ships tens of MB of rows).

    Skips streamed/passthrough responses (video files) and small payloads
    where compression overhead isn't worth it.
    """
    try:
        if (
            response.status_code != 200
            or response.direct_passthrough
            or response.is_streamed
            or not (response.content_type or "").startswith("application/json")
            or "gzip" not in request.headers.get("Accept-Encoding", "").lower()
            or response.headers.get("Content-Encoding")
        ):
            return response
        data = response.get_data()
        if len(data) < 4096:
            return response
        response.set_data(gzip.compress(data, 5))
        response.headers["Content-Encoding"] = "gzip"
        response.headers["Content-Length"] = str(len(response.get_data()))
        response.headers.add("Vary", "Accept-Encoding")
    except Exception:
        logger.exception("gzip compression failed; sending uncompressed")
    return response

# Flask's JSON provider sorts dict keys by default. If our response dicts
# accidentally mix str and non-str keys (e.g. pandas Series.value_counts()
# returns numpy keys when the column is numeric), the encoder raises
# `TypeError: '<' not supported between instances of 'str' and 'int'` on
# the whole response. Disable sorting — clients don't depend on key order
# and unsortable keys silently break the API otherwise.
try:
    app.json.sort_keys = False
except AttributeError:
    app.config["JSON_SORT_KEYS"] = False   # older Flask

# ── Pipeline state ──────────────────────────────────────────────────────────
_pipeline_state: dict = {
    "running": False,
    "mode": None,
    "started_at": None,
    "log": [],
    "process": None,  # subprocess.Popen reference for stop
}

# ── Per-session log files ───────────────────────────────────────────────────
# The in-memory log is capped at 500 lines (it only feeds the live UI), so
# every session is ALSO written in full to its own file on disk:
#
#   logs/sessions/session_2026-08-20_17-30-05_running.log        (while running)
#   logs/sessions/session_2026-08-20_17-30-05_duration_4h12m.log (after it ends)
#
# The name encodes the start timestamp and, once finished, the total runtime.
# Only filesystem-safe characters are used (no colons/spaces — Windows rejects
# colons in filenames), and the ISO date prefix keeps the folder sorted
# chronologically. The CLI's own daily DEBUG log (logs/pipeline_YYYYMMDD.log)
# is unaffected — this is the per-session INFO view, one file per Start press.
_SESSION_LOGS_DIR = _PROJECT_ROOT / "logs" / "sessions"
_session_log = {"file": None}  # open file handle while a session is running


def _session_log_write(line: str) -> None:
    """Append one line to the current session's log file (best-effort)."""
    handle = _session_log["file"]
    if handle is None:
        return
    try:
        handle.write(line + "\n")
    except Exception:
        pass  # a disk hiccup must never kill the pipeline reader thread


def _format_duration_for_filename(seconds: float) -> str:
    """Runtime formatted for a filename: '4h12m' / '37m05s' / '52s'."""
    secs = int(max(0, seconds))
    hours, rem = divmod(secs, 3600)
    minutes, sec = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def _append_log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    _pipeline_state["log"].append(line)
    _session_log_write(line)
    if len(_pipeline_state["log"]) > 500:
        _pipeline_state["log"] = _pipeline_state["log"][-500:]


def _run_pipeline(cmd: list):
    """Spawn the pipeline subprocess and stream its output into the log.

    The read-loop is wrapped so transient errors (e.g. a line that can't be
    decoded) never kill the thread while the child process is still alive —
    otherwise the frontend would see `running=False` even though the pipeline
    keeps writing to its own log file on disk.
    """
    process = None
    session_start = datetime.now()
    session_stamp = session_start.strftime("%Y-%m-%d_%H-%M-%S")
    session_path = _SESSION_LOGS_DIR / f"session_{session_stamp}_running.log"
    try:
        _SESSION_LOGS_DIR.mkdir(parents=True, exist_ok=True)
        # Line-buffered so the file stays current even if the process dies.
        _session_log["file"] = open(session_path, "a", encoding="utf-8", buffering=1)
    except Exception as e:
        logger.warning(f"Could not open session log {session_path}: {e}")
        _session_log["file"] = None

    try:
        _pipeline_state["running"] = True
        _pipeline_state["started_at"] = session_start.isoformat()
        _append_log(f"CMD: {' '.join(str(c) for c in cmd[2:])}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=str(_PROJECT_ROOT),
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
        )
        _pipeline_state["process"] = process

        # Resilient read loop: keep going until the child is truly gone,
        # even if a single readline call fails.
        while True:
            try:
                raw_line = process.stdout.readline()
            except Exception as read_err:
                _append_log(f"[reader] {read_err!r} — continuing")
                if process.poll() is not None:
                    break
                continue
            if raw_line == "":
                if process.poll() is not None:
                    break
                # Stdout closed but process still alive: just wait it out.
                try:
                    process.wait(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    continue
            line = raw_line.rstrip()
            if line:
                _pipeline_state["log"].append(line)
                _session_log_write(line)
                if len(_pipeline_state["log"]) > 500:
                    _pipeline_state["log"] = _pipeline_state["log"][-500:]

        process.wait()
        code = process.returncode
        _append_log(f"Process exited with code {code}")
        _append_log("Pipeline finished" if code == 0 else "Pipeline finished with errors")

    except Exception as e:
        _append_log(f"Runner error: {e}")
        logger.exception("Pipeline runner failed")
        # If the subprocess is still alive, try to terminate it so the state is
        # consistent — otherwise the UI says "done" while the OS process runs on.
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
    finally:
        # Close the session log and stamp the total runtime into its name:
        # session_<start>_running.log → session_<start>_duration_<runtime>.log
        duration_label = _format_duration_for_filename(
            (datetime.now() - session_start).total_seconds()
        )
        _session_log_write(f"[session] total runtime: {duration_label}")
        handle = _session_log["file"]
        _session_log["file"] = None
        if handle is not None:
            try:
                handle.close()
                final_path = _SESSION_LOGS_DIR / (
                    f"session_{session_stamp}_duration_{duration_label}.log"
                )
                session_path.replace(final_path)
                _append_log(f"Session log saved: {final_path.name}")
            except Exception as e:
                logger.warning(f"Could not finalise session log: {e}")

        _pipeline_state["running"] = False
        _pipeline_state["mode"] = None
        _pipeline_state["process"] = None


# ── Routes — Pages ──────────────────────────────────────────────────────────
# The UI is the React app in frontend/src, compiled into frontend/dist with
# `npm run build`. Flask only serves the resulting files; during development
# use `npm run dev` (Vite, port 5173, with a proxy to /api).
@app.route("/")
def index():
    index_file = _REACT_DIST / "index.html"
    if not index_file.exists():
        return (
            "<h3>React build not found.</h3>"
            "<p>Run <code>cd frontend && npm install && npm run build</code>, "
            "then reload this page.<br>"
            "For development, run <code>npm run dev</code> and open "
            "<a href='http://localhost:5173'>http://localhost:5173</a>.</p>",
            503,
        )
    return send_file(index_file)


@app.route("/assets/<path:filename>")
def react_assets(filename):
    """The JS/CSS files generated by Vite in dist/assets."""
    return send_from_directory(_REACT_DIST / "assets", filename)


# ── Routes — API ────────────────────────────────────────────────────────────
@app.route("/api/start", methods=["POST"])
def api_start():
    if _pipeline_state["running"]:
        return jsonify({"error": "Pipeline is already running"}), 409

    data = request.json or {}
    mode            = data.get("mode", "batch-pending")
    limit           = data.get("limit")
    video_id        = data.get("video_id", "").strip()
    url             = data.get("url", "").strip()
    cookies         = data.get("cookies", "").strip()
    cookies_browser = data.get("cookies_from_browser", "").strip()
    video_ids_raw   = data.get("video_ids", [])
    if isinstance(video_ids_raw, str):
        video_ids_raw = video_ids_raw.replace(",", " ").split()
    video_ids = [v.strip() for v in video_ids_raw if v and v.strip()]
    excel_path      = str(_METADATA_DIR / "videos_master.csv")

    # Auto-init data directory structure if it doesn't exist yet
    if not (_METADATA_DIR / "videos_master.csv").exists():
        try:
            init_cmd = [PYTHON, str(_SCRIPTS_DIR / "cli.py"), "init",
                        str(_PROJECT_ROOT / "data")]
            subprocess.run(init_cmd, cwd=str(_PROJECT_ROOT), capture_output=True, timeout=10)
            logger.info("Auto-initialized data directory (first run)")
        except Exception as e:
            logger.warning(f"Auto-init failed: {e}")

    if mode == "single":
        if not video_id or not url:
            return jsonify({"error": "video_id and url are required"}), 400
        cmd = [PYTHON, str(_SCRIPTS_DIR / "cli.py"), "single", video_id, url]

    elif mode == "resume":
        cmd = [PYTHON, str(_SCRIPTS_DIR / "cli.py"), "resume-batch", excel_path]
        if limit:
            cmd += ["--limit", str(limit)]
        if video_ids:
            cmd += ["--video-id", *video_ids]

    else:
        # batch-pending or batch-failed
        status = "failed" if mode == "batch-failed" else "pending"
        cmd = [
            PYTHON, str(_SCRIPTS_DIR / "cli.py"),
            "batch", excel_path, "--status", status,
        ]
        if limit:
            cmd += ["--limit", str(limit)]
        if video_ids:
            cmd += ["--video-id", *video_ids]

    if cookies:
        cmd += ["--cookies", cookies]
    if cookies_browser:
        cmd += ["--cookies-from-browser", cookies_browser]

    _pipeline_state["log"] = []
    _pipeline_state["mode"] = mode

    threading.Thread(target=_run_pipeline, args=(cmd,), daemon=True).start()
    return jsonify({"status": "started", "mode": mode})


@app.route("/api/bulk_import", methods=["POST"])
def api_bulk_import():
    """Download a list of YouTube URLs sequentially and seed videos_master.csv.

    Expects JSON: {
      "urls": ["https://youtube.com/...", ...],
      "prefix": "md",
      "region": "RO",
      "source": "YouTube_CC",
      "no_cc_check": bool,
      "cookies": "path/to/cookies.txt",
      "cookies_from_browser": "chrome"
    }
    """
    if _pipeline_state["running"]:
        return jsonify({"error": "A pipeline or import is already running"}), 409

    data = request.json or {}
    urls_raw = data.get("urls") or []
    if isinstance(urls_raw, str):
        urls_raw = urls_raw.replace(",", "\n").splitlines()
    urls = [u.strip() for u in urls_raw if u and u.strip() and not u.strip().startswith("#")]
    if not urls:
        return jsonify({"error": "No URLs provided"}), 400

    prefix  = (data.get("prefix") or "vid").strip() or "vid"
    region  = (data.get("region") or "UNKNOWN").strip() or "UNKNOWN"
    source  = (data.get("source") or "YouTube_CC").strip() or "YouTube_CC"
    cookies = (data.get("cookies") or "").strip()
    cookies_browser = (data.get("cookies_from_browser") or "").strip()
    no_cc_check = bool(data.get("no_cc_check", False))

    excel_path = str(_METADATA_DIR / "videos_master.csv")

    # Auto-init the CSV if it's missing (same policy as /api/start)
    if not (_METADATA_DIR / "videos_master.csv").exists():
        try:
            init_cmd = [PYTHON, str(_SCRIPTS_DIR / "cli.py"), "init",
                        str(_PROJECT_ROOT / "data")]
            subprocess.run(init_cmd, cwd=str(_PROJECT_ROOT), capture_output=True, timeout=10)
        except Exception as e:
            logger.warning(f"Auto-init failed: {e}")

    cmd = [
        PYTHON, str(_SCRIPTS_DIR / "cli.py"),
        "bulk-import", excel_path,
        "--prefix", prefix,
        "--region", region,
        "--source", source,
        "--urls", *urls,
    ]
    if no_cc_check:
        cmd.append("--no-cc-check")
    if cookies:
        cmd += ["--cookies", cookies]
    if cookies_browser:
        cmd += ["--cookies-from-browser", cookies_browser]

    _pipeline_state["log"] = []
    _pipeline_state["mode"] = "bulk-import"

    threading.Thread(target=_run_pipeline, args=(cmd,), daemon=True).start()
    return jsonify({"status": "started", "mode": "bulk-import", "urls": len(urls)})


@app.route("/api/status")
def api_status():
    proc = _pipeline_state.get("process")
    flag_running = bool(_pipeline_state["running"])
    proc_alive = False
    try:
        proc_alive = proc is not None and proc.poll() is None
    except Exception:
        proc_alive = False
    # Authoritative: if the OS process is still alive, we're running —
    # regardless of whether the reader thread flipped the flag prematurely.
    running = flag_running or proc_alive
    return jsonify({
        "running": running,
        "mode": _pipeline_state["mode"],
        "started_at": _pipeline_state["started_at"],
        "log": _pipeline_state["log"][-300:],
        "proc_alive": proc_alive,
    })


@app.route("/api/stop", methods=["POST"])
def api_stop():
    proc = _pipeline_state.get("process")
    if not _pipeline_state["running"] or proc is None:
        return jsonify({"error": "No pipeline running"}), 409
    try:
        proc.terminate()
        _append_log("Pipeline stopped by user")
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"ok": True})


def _stats_videos(pd, excel_path):
    """Aggregate the videos_master.csv block for /api/stats."""
    # ── Videos from master CSV ──
    if excel_path.exists():
        try:
            df = pd.read_csv(excel_path)
            # Stringify keys defensively (see _vc() comment below).
            by_status = (
                {str(k): int(v) for k, v in df["status"].value_counts().to_dict().items()}
                if "status" in df.columns else {}
            )

            ext_series = pd.to_numeric(
                df.get("total_duration_extracted", pd.Series(dtype=float)),
                errors="coerce",
            )
            src_series = pd.to_numeric(
                df.get("duration_seconds", pd.Series(dtype=float)),
                errors="coerce",
            )

            total_dur = float(ext_series.fillna(0).sum())
            total_src = float(src_series.fillna(0).sum())

            # Per-video duration distributions (whole-video, not segment-level).
            def _dist(s):
                s = s.dropna()
                if s.empty:
                    return None
                return {
                    "min":  round(float(s.min()), 2),
                    "mean": round(float(s.mean()), 2),
                    "max":  round(float(s.max()), 2),
                }
            src_dist = _dist(src_series)
            ext_dist = _dist(ext_series[ext_series > 0])

            # Breakdowns by categorical fields — useful to see corpus balance.
            # We force-coerce both keys and values to plain Python str/int so
            # the JSON encoder never sees pandas/numpy types or mixed-type
            # keys (root cause of the sort-keys TypeError seen in prod).
            def _vc(col):
                if col not in df.columns:
                    return {}
                s = df[col].dropna().astype(str).str.strip()
                s = s[s != ""]
                raw = s.value_counts().head(20).to_dict()
                return {str(k): int(v) for k, v in raw.items()}

            by_region   = _vc("region")
            by_source   = _vc("source")
            by_license  = _vc("license")
            top_channels = _vc("source_channel")

            # Speech mined per region. Sum extracted-duration per region so the
            # curator sees not just "how many videos per region" but "how much
            # audio". This is the metric that matters for a VSR dataset.
            speech_by_region: dict = {}
            if "region" in df.columns and "total_duration_extracted" in df.columns:
                gd = df.groupby(df["region"].fillna("UNKNOWN").astype(str))
                for reg, sub in gd:
                    s = pd.to_numeric(sub["total_duration_extracted"], errors="coerce").fillna(0).sum()
                    if s > 0:
                        speech_by_region[reg] = round(float(s), 2)

            # Top 10 most-extracted videos + top 10 best mining ratios.
            top_extracted: list = []
            top_mined: list = []
            if "video_id" in df.columns:
                cols = ["video_id", "title", "region"]
                cols = [c for c in cols if c in df.columns]
                aux = df[cols].copy()
                aux["_extracted"] = ext_series
                aux["_source"] = src_series
                t1 = aux.dropna(subset=["_extracted"]).sort_values("_extracted", ascending=False).head(10)
                for _, r in t1.iterrows():
                    top_extracted.append({
                        "video_id":      str(r.get("video_id", "")),
                        "title":         str(r.get("title", "") or ""),
                        "region":        str(r.get("region", "") or ""),
                        "extracted_s":   round(float(r["_extracted"]), 1),
                        "source_s":      None if pd.isna(r["_source"]) else round(float(r["_source"]), 1),
                    })
                aux2 = aux[(aux["_source"] > 0) & aux["_extracted"].notna()].copy()
                aux2["_ratio"] = aux2["_extracted"] / aux2["_source"]
                t2 = aux2.sort_values("_ratio", ascending=False).head(10)
                for _, r in t2.iterrows():
                    top_mined.append({
                        "video_id":      str(r.get("video_id", "")),
                        "title":         str(r.get("title", "") or ""),
                        "region":        str(r.get("region", "") or ""),
                        "extracted_s":   round(float(r["_extracted"]), 1),
                        "source_s":      round(float(r["_source"]), 1),
                        "ratio_pct":     round(float(r["_ratio"]) * 100, 1),
                    })

            return {
                "total": len(df),
                "by_status": by_status,
                "by_region": by_region,
                "by_source": by_source,
                "by_license": by_license,
                "top_channels": top_channels,
                "speech_by_region_s": speech_by_region,
                "total_duration_h": round(total_dur / 3600, 2),
                "total_source_duration_s": round(total_src, 2),
                "total_source_duration_h": round(total_src / 3600, 2),
                "source_duration": src_dist,
                "extracted_duration": ext_dist,
                "top_extracted": top_extracted,
                "top_mined": top_mined,
            }
        except Exception as e:
            return {"total": 0, "by_status": {}, "error": str(e)}
    else:
        return {"total": 0, "by_status": {}}



def _stats_segments(pd, csv_path, excel_path):
    """Aggregate the segments_index.csv block for /api/stats."""
    # ── Segments from CSV ──
    if csv_path.exists():
        try:
            seg_df = pd.read_csv(csv_path)
            n = len(seg_df)
            dur_col   = next((c for c in ("duration",) if c in seg_df.columns), None)
            asd_col   = next((c for c in ("asd_score",) if c in seg_df.columns), None)
            sync_col  = next((c for c in ("syncnet_confidence", "syncnet_conf") if c in seg_df.columns), None)
            wh_col    = next((c for c in ("whisper_confidence", "whisper_conf") if c in seg_df.columns), None)
            wrd_col   = next((c for c in ("num_words",) if c in seg_df.columns), None)
            chr_col   = next((c for c in ("num_chars",) if c in seg_df.columns), None)
            text_col  = "text" if "text" in seg_df.columns else None

            total_dur_s = float(seg_df[dur_col].fillna(0).sum()) if dur_col else 0.0
            total_words = int(seg_df[wrd_col].fillna(0).sum()) if wrd_col else 0
            total_chars = int(seg_df[chr_col].fillna(0).sum()) if chr_col else 0

            unique_words = 0
            if text_col is not None:
                vocab: set = set()
                for txt in seg_df[text_col].dropna().astype(str):
                    for w in txt.split():
                        w_clean = w.strip(".,;:!?\"'()[]").upper()
                        if w_clean:
                            vocab.add(w_clean)
                unique_words = len(vocab)

            # ── Approved subset (manual review) ──
            approved = {"count": 0, "total_duration_s": 0.0, "total_words": 0, "unique_words": 0}
            rejected_count = 0
            edited_count = 0
            try:
                review = _load_review()
                approved_ids = {sid for sid, info in review.items()
                                if isinstance(info, dict) and info.get("status") == "approved"}
                rejected_count = sum(1 for info in review.values()
                                     if isinstance(info, dict) and info.get("status") == "rejected")
                edited_count = sum(
                    1 for info in review.values()
                    if isinstance(info, dict)
                    and (info.get("transcript_edited") is True or info.get("trimmed") is True)
                )
                if approved_ids and "segment_id" in seg_df.columns:
                    mask = seg_df["segment_id"].isin(approved_ids)
                    ap_df = seg_df[mask]
                    approved["count"] = int(len(ap_df))
                    if dur_col:
                        approved["total_duration_s"] = round(float(ap_df[dur_col].fillna(0).sum()), 2)
                    if wrd_col:
                        approved["total_words"] = int(ap_df[wrd_col].fillna(0).sum())
                    if text_col is not None:
                        ap_vocab: set = set()
                        for txt in ap_df[text_col].dropna().astype(str):
                            for w in txt.split():
                                wc = w.strip(".,;:!?\"'()[]").upper()
                                if wc:
                                    ap_vocab.add(wc)
                        approved["unique_words"] = len(ap_vocab)
            except Exception:
                pass

            wer_col = "wer" if "wer" in seg_df.columns else None
            wer_series = pd.to_numeric(seg_df[wer_col], errors="coerce") if wer_col else None
            avg_wer = round(float(wer_series.mean()), 4) if wer_series is not None and wer_series.notna().any() else None
            with_wer_count = int(wer_series.notna().sum()) if wer_series is not None else 0

            # Quantile distribution for segment durations — useful for VSR
            # since training expects clips in a narrow length band. The
            # curator can spot if too many clips are at the limits (1s / 15s).
            duration_stats = None
            duration_bands: list = []
            if dur_col:
                d = pd.to_numeric(seg_df[dur_col], errors="coerce").dropna()
                if not d.empty:
                    duration_stats = {
                        "min":     round(float(d.min()), 2),
                        "p25":     round(float(d.quantile(0.25)), 2),
                        "median":  round(float(d.median()), 2),
                        "mean":    round(float(d.mean()), 2),
                        "p75":     round(float(d.quantile(0.75)), 2),
                        "p95":     round(float(d.quantile(0.95)), 2),
                        "max":     round(float(d.max()), 2),
                    }
                    # Training-band breakdown: how many clips fall in each
                    # length bucket. 3–6s is the LRS2 sweet-spot; tails
                    # (<1s, >15s) flag VAD misconfiguration.
                    bands_def = [
                        ("<1s",    None, 1.0),
                        ("1–3s",   1.0,  3.0),
                        ("3–6s",   3.0,  6.0),
                        ("6–10s",  6.0,  10.0),
                        ("10–15s", 10.0, 15.0),
                        (">15s",   15.0, None),
                    ]
                    total_n = int(len(d))
                    for label, lo, hi in bands_def:
                        if lo is None:
                            mask = d < hi
                        elif hi is None:
                            mask = d >= lo
                        else:
                            mask = (d >= lo) & (d < hi)
                        cnt = int(mask.sum())
                        dur_s = float(d[mask].sum())
                        duration_bands.append({
                            "label": label,
                            "count": cnt,
                            "pct": round(100.0 * cnt / total_n, 1) if total_n else 0.0,
                            "duration_s": round(dur_s, 1),
                        })
            words_stats = None
            if wrd_col:
                w = pd.to_numeric(seg_df[wrd_col], errors="coerce").dropna()
                if not w.empty:
                    words_stats = {
                        "min":    int(w.min()),
                        "median": int(w.median()),
                        "mean":   round(float(w.mean()), 1),
                        "max":    int(w.max()),
                    }

            # Speaking pace — mean words per minute across the dataset.
            wpm = round((total_words / total_dur_s) * 60.0, 1) if total_dur_s > 0 else 0

            # Segment counts/duration broken down by region (needs JOIN with
            # videos_master since segments only carry video_id, not region).
            seg_by_region: dict = {}
            try:
                if excel_path.exists() and "video_id" in seg_df.columns:
                    master_df = pd.read_csv(excel_path)
                    if "region" in master_df.columns:
                        regions = master_df.set_index(master_df["video_id"].astype(str))["region"]
                        seg_df["_region"] = seg_df["video_id"].astype(str).map(regions).fillna("UNKNOWN")
                        grp = seg_df.groupby("_region")
                        for reg, sub in grp:
                            ds = pd.to_numeric(sub.get(dur_col, pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
                            seg_by_region[str(reg)] = {
                                "count": int(len(sub)),
                                "duration_s": round(float(ds), 1),
                            }
                        seg_df.drop(columns=["_region"], inplace=True, errors="ignore")
            except Exception:
                pass

            # Conf-level distribution (1/2/3) — needs reading the per-segment
            # annotation files since Conf is in the .txt, not the CSV.
            # We sample up to ~3k files for responsiveness; on a 12k-segment
            # corpus this gives a fast, representative answer.
            # Keys are all strings so Flask's JSON encoder (sort_keys=True in
            # debug mode) doesn't try to compare int vs str.
            # Cached on (segments CSV mtime, review file mtime): approving a
            # segment rewrites its annotation's Conf line AND the review file,
            # so the review mtime invalidates the cache when Conf changes.
            conf_dist = {"1": 0, "2": 0, "3": 0, "unknown": 0}
            try:
                cache_key = (
                    csv_path.stat().st_mtime,
                    _REVIEW_FILE.stat().st_mtime if _REVIEW_FILE.exists() else None,
                )
                if _conf_dist_cache["key"] == cache_key and _conf_dist_cache["dist"]:
                    conf_dist = _conf_dist_cache["dist"]
                else:
                    anno_paths: list = []
                    if "annotation_path" in seg_df.columns:
                        anno_paths = seg_df["annotation_path"].dropna().astype(str).tolist()
                    # Cap to keep the request snappy.
                    for p in anno_paths[:3000]:
                        pth = (_PROJECT_ROOT / p) if not Path(p).is_absolute() else Path(p)
                        if not pth.exists():
                            continue
                        seen = False
                        for line in pth.read_text(encoding="utf-8").splitlines():
                            if line.startswith("Conf:"):
                                try:
                                    lvl = int(line[len("Conf:"):].strip())
                                    if lvl in (1, 2, 3):
                                        conf_dist[str(lvl)] += 1
                                    else:
                                        conf_dist["unknown"] += 1
                                except ValueError:
                                    conf_dist["unknown"] += 1
                                seen = True
                                break
                        if not seen:
                            conf_dist["unknown"] += 1
                    _conf_dist_cache["key"] = cache_key
                    _conf_dist_cache["dist"] = conf_dist
            except Exception:
                pass

            # Rare-words count — words that occur ≤ 1 or ≤ 2 times across
            # the whole corpus. Useful coverage signal for VSR (rare words
            # are the hardest to recognise without enough samples).
            rare1 = rare2 = 0
            if text_col is not None:
                word_freq: dict = {}
                for txt in seg_df[text_col].dropna().astype(str):
                    for w in txt.split():
                        wc = w.strip(".,;:!?\"'()[]").upper()
                        if wc:
                            word_freq[wc] = word_freq.get(wc, 0) + 1
                for c in word_freq.values():
                    if c == 1:
                        rare1 += 1
                    elif c == 2:
                        rare2 += 1

            # ── v3 quality tiers (present only after a v3 run) ──
            # Per-tier counts/durations + "training-ready" = segments that are
            # BOTH approved by a curator AND tier A/B — the honest size of the
            # trainable dataset.
            tiers = None
            training_ready = None
            if "quality_tier" in seg_df.columns:
                try:
                    tier_series = seg_df["quality_tier"].astype(str).str.strip().str.upper()
                    tiers = {}
                    for tier_name in ("A", "B", "C"):
                        mask = tier_series == tier_name
                        dur_s = (
                            float(seg_df.loc[mask, dur_col].fillna(0).sum())
                            if dur_col else 0.0
                        )
                        tiers[tier_name] = {
                            "count": int(mask.sum()),
                            "duration_s": round(dur_s, 2),
                        }
                    review_map = _load_review()
                    approved_seg_ids = {
                        sid for sid, info in review_map.items()
                        if isinstance(info, dict) and info.get("status") == "approved"
                    }
                    ready_mask = (
                        seg_df["segment_id"].astype(str).isin(approved_seg_ids)
                        & tier_series.isin(["A", "B"])
                    )
                    ready_dur = (
                        float(seg_df.loc[ready_mask, dur_col].fillna(0).sum())
                        if dur_col else 0.0
                    )
                    training_ready = {
                        "count": int(ready_mask.sum()),
                        "duration_s": round(ready_dur, 2),
                    }
                except Exception:
                    tiers = None
                    training_ready = None

            return {
                "total": n,
                "total_duration_s": round(total_dur_s, 2),
                "total_duration_h": round(total_dur_s / 3600, 3),
                "avg_duration_s":   round(float(seg_df[dur_col].mean()), 2) if dur_col else 0,
                "duration_stats":   duration_stats,
                "duration_bands":   duration_bands,
                "words_stats":      words_stats,
                "words_per_minute": wpm,
                "total_words":      total_words,
                "unique_words":     unique_words,
                "rare_words_1":     rare1,    # words occurring exactly once
                "rare_words_2":     rare2,    # words occurring exactly twice
                "total_chars":      total_chars,
                "avg_words":        round(float(seg_df[wrd_col].mean()), 1) if wrd_col else 0,
                "avg_asd":          round(float(seg_df[asd_col].mean()), 3) if asd_col else 0,
                "avg_syncnet":      round(float(seg_df[sync_col].mean()), 3) if sync_col else 0,
                "avg_whisper_conf": round(float(seg_df[wh_col].mean()), 3) if wh_col else 0,
                "avg_wer":          avg_wer,
                "with_wer_count":   with_wer_count,
                "edited_count":     edited_count,
                "approved":         approved,
                "rejected_count":   rejected_count,
                "by_region":        seg_by_region,
                "by_conf":          conf_dist,
                "tiers":            tiers,
                "training_ready":   training_ready,
            }
        except Exception as e:
            return {"total": 0, "error": str(e)}
    else:
        return {"total": 0}



@app.route("/api/stats")
def api_stats():
    try:
        import pandas as pd
    except ImportError:
        return jsonify({"error": "pandas not installed"}), 500

    stats: dict = {}
    excel_path = _METADATA_DIR / "videos_master.csv"
    csv_path   = _METADATA_DIR / "segments_index.csv"

    stats["videos"] = _stats_videos(pd, excel_path)

    stats["segments"] = _stats_segments(pd, csv_path, excel_path)

    # Also expose raw video-level duration in seconds for nicer formatting
    if "videos" in stats and "total_duration_h" in stats["videos"]:
        stats["videos"]["total_duration_s"] = round(stats["videos"]["total_duration_h"] * 3600, 2)

    return jsonify(stats)


@app.route("/api/videos")
def api_videos():
    """Per-video master rows, with stats RECOMPUTED from segments_index.csv.

    videos_master.total_duration_extracted / total_segments / avg_asd_score
    are written once when a video finishes processing, but never updated
    after rejects / orphan cleanup / re-runs. Without this recompute, the
    Video Registry table sums to a different total than the dashboard KPI
    (which reads segments_index directly).
    """
    try:
        import pandas as pd
    except ImportError:
        return jsonify({"videos": []})

    excel_path = _METADATA_DIR / "videos_master.csv"
    if not excel_path.exists():
        return jsonify({"videos": []})

    try:
        df = pd.read_csv(excel_path).fillna("")

        # Recompute per-video stats from segments_index — single CSV pass,
        # negligible cost vs returning stale numbers users have to debug.
        seg_csv = _METADATA_DIR / "segments_index.csv"
        if seg_csv.exists():
            seg_df = pd.read_csv(seg_csv)
            if not seg_df.empty and "video_id" in seg_df.columns:
                grp = seg_df.groupby(seg_df["video_id"].astype(str))
                fresh = pd.DataFrame({
                    "_video_id":          grp.size().index,
                    "_total_segments":    grp.size().values,
                    "_total_duration_s":  (
                        grp["duration"].sum().values if "duration" in seg_df.columns
                        else [0] * len(grp)
                    ),
                    "_avg_asd_score":     (
                        grp["asd_score"].mean().values if "asd_score" in seg_df.columns
                        else [0] * len(grp)
                    ),
                })
                # Map back onto the master rows. We keep the master columns
                # (status, region, license, ...) and overwrite ONLY the
                # per-video stats with the freshly-computed values.
                df = df.merge(fresh, how="left",
                              left_on=df["video_id"].astype(str),
                              right_on="_video_id")
                # Truth = segments_index. A video with no surviving segments
                # gets 0 (it was either never processed yet, or every segment
                # was rejected — either way, 0 is the honest answer here).
                # We deliberately do NOT fall back to the (stale) master value.
                df["total_segments"]           = df["_total_segments"].fillna(0).astype(int)
                df["total_duration_extracted"] = df["_total_duration_s"].fillna(0).round(2)
                df["avg_asd_score"]            = df["_avg_asd_score"].fillna(0).round(3)
                df = df.drop(columns=[c for c in df.columns
                                      if c.startswith("_") or c == "key_0"], errors="ignore")

        records = []
        for r in df.to_dict("records"):
            records.append({str(k): str(v) if v != "" else "" for k, v in r.items()})
        return jsonify({"videos": records})
    except Exception as e:
        logger.exception("api_videos failed")
        return jsonify({"videos": [], "error": str(e)})


# Columns the Explorer grid actually consumes — keeping the payload small
# matters because we ship ~12k rows on every load. Anything outside this list
# (video_path, face_bbox, head_pose_avg, …) is dropped before serialising.
_EXPLORER_COLUMNS = (
    "segment_id", "video_id", "duration", "text", "num_words",
    "asd_score", "syncnet_conf", "whisper_conf",
    "speaker_id", "original_text", "wer",
    # v3 pipeline columns (additive; simply absent until v3 has run).
    # text_largev3 is intentionally NOT here — it would double the payload;
    # the per-segment detail endpoint returns every column anyway.
    "quality_tier", "mouth_roi_method", "mouth_landmark_fail_rate",
    "boundary_type", "needs_review", "wer_medium_vs_large",
)

# Tiny in-process cache: re-parse the CSVs only when their mtimes change.
# The key covers BOTH segments_index and videos_master, since the response
# joins the per-video region onto every segment row.
_segments_cache: dict = {"mtime": None, "payload": None}


@app.route("/api/segments")
def api_segments():
    try:
        import pandas as pd
    except ImportError:
        return jsonify({"segments": []})

    csv_path = _METADATA_DIR / "segments_index.csv"
    if not csv_path.exists():
        return jsonify({"segments": []})

    try:
        import math
        master_path = _METADATA_DIR / "videos_master.csv"
        mtime = (
            csv_path.stat().st_mtime,
            master_path.stat().st_mtime if master_path.exists() else None,
        )
        if _segments_cache["mtime"] == mtime and _segments_cache["payload"] is not None:
            return jsonify(_segments_cache["payload"])

        df = pd.read_csv(csv_path)
        keep = [c for c in _EXPLORER_COLUMNS if c in df.columns]
        df = df[keep].copy()

        # Join the per-video region from videos_master (segments only carry
        # video_id) so the UI can filter by region (MD-only dataset target).
        if master_path.exists() and "video_id" in df.columns:
            try:
                master_df = pd.read_csv(master_path)
                if "region" in master_df.columns:
                    region_by_video = master_df.set_index(
                        master_df["video_id"].astype(str)
                    )["region"]
                    df["region"] = (
                        df["video_id"].astype(str).map(region_by_video).fillna("UNKNOWN")
                    )
            except Exception:
                logger.warning("Region join failed; segments served without region")

        numeric_cols = ("duration", "num_words", "asd_score",
                        "syncnet_conf", "whisper_conf", "wer")
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Per-row NaN→None scrub. `df.where(df.notna(), None)` leaves float NaN
        # in float-typed columns, which `jsonify` emits as `NaN` — invalid JSON
        # for strict parsers and the source of the "Failed to load" we saw.
        records = df.to_dict("records")
        for r in records:
            for k, v in r.items():
                if v is None:
                    continue
                if isinstance(v, float) and math.isnan(v):
                    r[k] = None
                elif isinstance(v, str) and v == "nan":
                    r[k] = None

        payload = {"segments": records, "total": len(records)}
        _segments_cache["mtime"] = mtime
        _segments_cache["payload"] = payload
        return jsonify(payload)
    except Exception as e:
        logger.exception("api_segments failed")
        return jsonify({"segments": [], "error": str(e)})


@app.route("/api/vocabulary")
def api_vocabulary():
    """Word frequency table with ACTUAL per-word duration summed from annotations.

    For each annotation file we parse the WORD / START / END table and add
    (end − start) per occurrence. Falls back to (segment_duration / num_words)
    when an annotation file is missing.
    """
    try:
        import pandas as pd
    except ImportError:
        return jsonify({"words": []})

    csv_path = _METADATA_DIR / "segments_index.csv"
    if not csv_path.exists():
        return jsonify({"words": []})

    try:
        # This endpoint walks EVERY annotation file for per-word durations —
        # by far the most expensive scan in the app. Cache on the CSV + review
        # mtimes (transcript/word edits touch both; approvals touch review).
        cache_key = (
            csv_path.stat().st_mtime,
            _REVIEW_FILE.stat().st_mtime if _REVIEW_FILE.exists() else None,
        )
        if _vocab_cache["key"] == cache_key and _vocab_cache["payload"] is not None:
            return jsonify(_vocab_cache["payload"])

        df = pd.read_csv(csv_path)
        if "text" not in df.columns:
            return jsonify({"words": []})

        dur_col = "duration" if "duration" in df.columns else None
        word_counts: dict = {}  # word -> {samples, occurrences, duration}

        def _bump(word_key: str, dur_delta: float):
            entry = word_counts.setdefault(
                word_key, {"samples": 0, "occurrences": 0, "duration": 0.0}
            )
            entry["occurrences"] += 1
            entry["duration"] += max(0.0, dur_delta)

        for _, row in df.iterrows():
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            seg_dur = float(row[dur_col]) if dur_col and pd.notna(row[dur_col]) else 0.0

            # Try to read the per-word timings from the annotation file.
            anno_rel = str(row.get("annotation_path", "")).strip().replace("\\", "/")
            anno_path = (_PROJECT_ROOT / anno_rel) if anno_rel else None
            per_word_durs = []  # list of (WORD_UPPER_CLEAN, dur_seconds)

            if anno_path and anno_path.exists():
                try:
                    in_words = False
                    for line in anno_path.read_text(encoding="utf-8").splitlines():
                        if line.startswith("WORD START END"):
                            in_words = True
                            continue
                        if not in_words or not line.strip():
                            continue
                        parts = line.split()
                        if len(parts) < 3:
                            continue
                        try:
                            ws, we = float(parts[1]), float(parts[2])
                        except ValueError:
                            continue
                        wc = parts[0].strip(".,;:!?\"'()[]").upper()
                        if wc:
                            per_word_durs.append((wc, max(0.0, we - ws)))
                except Exception:
                    per_word_durs = []

            if per_word_durs:
                # Use real per-word timings — one bump per occurrence
                samples_counted = set()
                for wc, d in per_word_durs:
                    _bump(wc, d)
                    samples_counted.add(wc)
                # Also bump sample-count (once per unique word per segment)
                for wc in samples_counted:
                    word_counts[wc]["samples"] += 1
            else:
                # Fallback: distribute the segment's duration evenly across
                # the words present in the transcript. Still counts one sample
                # per unique word per segment.
                words_in_seg = [
                    w.strip(".,;:!?\"'()[]").upper()
                    for w in text.split()
                    if w.strip(".,;:!?\"'()[]")
                ]
                if not words_in_seg:
                    continue
                per_word_dur = seg_dur / len(words_in_seg) if words_in_seg else 0.0
                seen = set()
                for wc in words_in_seg:
                    _bump(wc, per_word_dur)
                    if wc not in seen:
                        seen.add(wc)
                        word_counts[wc]["samples"] += 1

        words = [
            {
                "word": w,
                "samples": d["samples"],            # # of segments this word appears in
                "occurrences": d["occurrences"],    # total count across all segments
                "duration": round(d["duration"], 2) # total spoken seconds (from annotations)
            }
            for w, d in word_counts.items()
        ]
        words.sort(key=lambda x: x["samples"], reverse=True)
        payload = {"words": words, "total_unique": len(words)}
        _vocab_cache["key"] = cache_key
        _vocab_cache["payload"] = payload
        return jsonify(payload)
    except Exception as e:
        return jsonify({"words": [], "error": str(e)})


@app.route("/api/segment/<segment_id>")
def api_segment_detail(segment_id: str):
    try:
        import pandas as pd
    except ImportError:
        return jsonify({"error": "pandas not installed"}), 500

    csv_path = _METADATA_DIR / "segments_index.csv"
    if not csv_path.exists():
        return jsonify({"error": "No segments index found"}), 404

    try:
        df = pd.read_csv(csv_path).fillna("")
        row = df[df["segment_id"].astype(str) == segment_id]
        if row.empty:
            return jsonify({"error": "Segment not found"}), 404
        seg = {str(k): str(v) for k, v in row.iloc[0].to_dict().items()}
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    video_id  = seg.get("video_id", "")
    # Videos live in face_crop/ subdir (new layout)
    video_path = _PROCESSED_DIR / video_id / "face_crop" / f"{segment_id}.mp4"
    anno_path  = _annotation_path(video_id, segment_id)

    seg["has_video"] = video_path.exists()
    seg["has_annotation"] = anno_path.exists()

    if anno_path.exists():
        seg["annotation_raw"] = anno_path.read_text(encoding="utf-8")
        seg["annotation"] = _parse_annotation(anno_path)

    # Include review status
    review = _load_review()
    seg["review"] = review.get(segment_id, {"status": "pending"})

    return jsonify(seg)


_REVIEW_FILE = _METADATA_DIR / "review_status.json"


def _load_review() -> dict:
    try:
        if _REVIEW_FILE.exists():
            return json.loads(_REVIEW_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_review(data: dict):
    """Write review_status.json atomically (tmp + rename) under the lock."""
    with _METADATA_WRITE_LOCK:
        _METADATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _REVIEW_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(_REVIEW_FILE)


def _mark_checkpoint_rejected(video_id: str, segment_id: str) -> None:
    """Patch the per-video checkpoint to mark the clip that produced segment_id
    as rejected, so resume-batch skips it silently instead of warning about a
    missing segment file. No-op if the checkpoint or matching entry is absent.
    """
    cp_path = _PROJECT_ROOT / "data" / "clips" / video_id / ".checkpoint.json"
    if not cp_path.exists():
        return
    try:
        cp = json.loads(cp_path.read_text(encoding="utf-8"))
        changed = False
        for _clip_id, info in cp.get("processed_clips", {}).items():
            if info.get("segment_id") == segment_id:
                info["result"] = "rejected"
                info["rejected_at"] = datetime.now().isoformat()
                changed = True
                break
        if changed:
            tmp = cp_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(cp, indent=2), encoding="utf-8")
            tmp.replace(cp_path)
    except Exception as e:
        logger.warning(f"Checkpoint update for rejected {segment_id} failed: {e}")


@app.route("/api/review_status")
def api_review_status():
    return jsonify(_load_review())


@app.route("/api/segment/<segment_id>/review", methods=["POST"])
@_with_metadata_lock
def api_review_segment(segment_id: str):
    data   = request.json or {}
    action = data.get("action")  # "approve" | "reject" | "save"
    review = _load_review()

    if action == "approve":
        review[segment_id] = {**review.get(segment_id, {}), "status": "approved"}
        _save_review(review)
        # Approval = explicit human sign-off → promote Conf to 3.
        video_id = _get_video_id(segment_id)
        if video_id is not None:
            _set_conf_in_annotation(
                _annotation_path(video_id, segment_id),
                _HUMAN_REVIEWED_CONF_LEVEL,
            )
            _recompute_video_validation_status(video_id)
        return jsonify({"ok": True, "status": "approved"})

    elif action == "reject":
        # Find video_id from CSV, then delete artifacts + drop the row.
        try:
            video_id = _get_video_id(segment_id)
            if video_id is not None:
                # Delete face_crop, mouth_crop and annotation
                for f in [
                    _PROCESSED_DIR   / video_id / "face_crop"  / f"{segment_id}.mp4",
                    _PROCESSED_DIR   / video_id / "mouth_crop" / f"{segment_id}.mp4",
                    _annotation_path(video_id, segment_id),
                ]:
                    if f.exists():
                        f.unlink()
                # Remove from CSV
                delete_segment_row(segment_id)
                # Mark the producing clip as rejected in the checkpoint so
                # resume doesn't warn about a missing segment file.
                _mark_checkpoint_rejected(video_id, segment_id)
        except Exception as e:
            logger.warning(f"Reject cleanup error: {e}")

        review[segment_id] = {**review.get(segment_id, {}), "status": "rejected"}
        _save_review(review)
        # Re-evaluate validation: the rejected segment was removed from
        # segments_index, so the remaining survivors decide the video status.
        rejected_video = _get_video_id(segment_id)
        if rejected_video is not None:
            _recompute_video_validation_status(rejected_video)
        return jsonify({"ok": True, "status": "rejected"})

    elif action == "save":
        new_text = data.get("text", "").strip()
        if not new_text:
            return jsonify({"error": "text is required"}), 400

        video_id = _get_video_id(segment_id)
        if video_id is None:
            return jsonify({"error": "Segment not found in CSV"}), 404

        # Update CSV text column
        try:
            update_segment(segment_id, text=new_text)
        except Exception as e:
            logger.warning(f"CSV update error: {e}")

        # Rewrite annotation — update ONLY the Text: line, keep Original/Conf/words.
        # The condition l.startswith("Text:") would also match "Text: " but NOT
        # "Original:" (different prefix), so the original reference is preserved.
        anno_path = _annotation_path(video_id, segment_id)
        if anno_path.exists():
            lines = anno_path.read_text(encoding="utf-8").splitlines()
            anno_path.write_text(
                "\n".join(f"Text:  {new_text}" if l.startswith("Text:") else l
                          for l in lines),
                encoding="utf-8",
            )
            _recompute_wer_in_csv(
                segment_id,
                _read_original_from_annotation(anno_path),
                new_text,
            )
            # Manual save = human curator vouches for the text → Conf 3.
            _set_conf_in_annotation(anno_path, _HUMAN_REVIEWED_CONF_LEVEL)

        review[segment_id] = {**review.get(segment_id, {}), "transcript_edited": True}
        _save_review(review)
        return jsonify({"ok": True})

    elif action == "save_words":
        # Rewrite full annotation with updated word timings (and optionally new text)
        words    = data.get("words", [])   # [{word, start, end, score}, ...]
        new_text = data.get("text", "").strip()

        video_id = _get_video_id(segment_id)
        if video_id is None:
            return jsonify({"error": "Segment not found in CSV"}), 404

        anno_path = _annotation_path(video_id, segment_id)
        if not anno_path.exists():
            return jsonify({"error": "Annotation not found"}), 404

        parsed = _parse_annotation(anno_path)
        if not new_text:
            new_text = parsed["text"]
        # Editing words is a curator action → promote Conf to 3.
        conf = _HUMAN_REVIEWED_CONF_LEVEL
        original = parsed.get("original")

        lines = [f"Text:  {new_text}"]
        # Preserve the Whisper-original reference when the file has one;
        # legacy annotations without Original: simply skip this line.
        if original is not None:
            lines.append(f"Original:  {original}")
        lines.append(f"Conf:  {conf}")
        lines.append("WORD START END ASDSCORE")
        for w in words:
            asd = w.get('asd_score') or w.get('score') or 0
            lines.append(f"{w['word']} {float(w['start']):.2f} {float(w['end']):.2f} {float(asd):.1f}")
        anno_path.write_text("\n".join(lines), encoding="utf-8")

        # Update CSV
        try:
            update_segment(segment_id, text=new_text, num_words=len(words))
        except Exception as e:
            logger.warning(f"CSV update error: {e}")

        _recompute_wer_in_csv(segment_id, original, new_text)

        review[segment_id] = {**review.get(segment_id, {}), "transcript_edited": True}
        _save_review(review)
        return jsonify({"ok": True})

    elif action == "revert":
        # Set status back to pending (no file changes)
        entry = review.get(segment_id, {})
        entry["status"] = "pending"
        entry.pop("transcript_edited", None)
        review[segment_id] = entry
        _save_review(review)
        # Reverting can demote a video from validated → completed.
        reverted_video = _get_video_id(segment_id)
        if reverted_video is not None:
            _recompute_video_validation_status(reverted_video)
        return jsonify({"ok": True, "status": "pending"})

    return jsonify({"error": "Unknown action"}), 400


def _segments_csv() -> Path:
    return _METADATA_DIR / "segments_index.csv"


def load_segments_df():
    """Read segments_index.csv as a DataFrame, or None if it doesn't exist."""
    import pandas as pd
    csv_path = _segments_csv()
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def _get_video_id(segment_id: str):
    """Look up video_id for a segment from segments_index.csv."""
    try:
        df = load_segments_df()
        if df is not None:
            row = df[df["segment_id"].astype(str) == segment_id]
            if not row.empty:
                return str(row.iloc[0]["video_id"])
    except Exception:
        pass
    return None


def update_segment(segment_id: str, **fields) -> bool:
    """Set columns on one segment's row in segments_index.csv.

    Missing columns are created (None default) so legacy CSVs that predate a
    column still get updated. Returns True if a matching row was written.
    """
    with _METADATA_WRITE_LOCK:
        df = load_segments_df()
        if df is None:
            return False
        mask = df["segment_id"].astype(str) == str(segment_id)
        if not mask.any():
            return False
        for col, val in fields.items():
            if col not in df.columns:
                df[col] = None
            df.loc[mask, col] = val
        _write_segments_csv_atomic(df)
        return True


def _write_segments_csv_atomic(df) -> None:
    """Write segments_index.csv atomically (tmp + rename), so a crash or a
    concurrent reader can never observe a half-written CSV."""
    target = _segments_csv()
    tmp = target.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(target)


def delete_segment_row(segment_id: str) -> bool:
    """Remove the row for `segment_id` from segments_index.csv. Returns True if written."""
    with _METADATA_WRITE_LOCK:
        df = load_segments_df()
        if df is None:
            return False
        df = df[df["segment_id"].astype(str) != str(segment_id)]
        _write_segments_csv_atomic(df)
        return True


@app.route("/api/segment/<segment_id>/trim", methods=["POST"])
@_with_metadata_lock
def api_trim_segment(segment_id: str):
    """Trim face_crop and mouth_crop videos and update annotation timings.

    Body: {start: float, end: float}  — seconds relative to current clip start.
    """
    data  = request.json or {}
    t_in  = float(data.get("start", 0))
    t_out = float(data.get("end",   0))

    if t_out <= t_in or t_in < 0:
        return jsonify({"error": "Invalid trim range"}), 400

    video_id = _get_video_id(segment_id)
    if video_id is None:
        return jsonify({"error": "Segment not found"}), 404

    duration = round(t_out - t_in, 3)

    # Trim each video crop in-place
    for subdir in ("face_crop", "mouth_crop"):
        src = _PROCESSED_DIR / video_id / subdir / f"{segment_id}.mp4"
        if not src.exists():
            continue
        tmp = src.with_suffix(".trimming.mp4")
        cmd = [
            FFMPEG, "-y",
            "-ss", str(t_in), "-t", str(duration),
            "-i", str(src),
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "aac", "-ar", "16000", "-ac", "1",
            str(tmp),
            "-loglevel", "error",
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            tmp.replace(src)
        except subprocess.CalledProcessError as e:
            tmp.unlink(missing_ok=True)
            logger.warning(f"Trim failed for {subdir}/{segment_id}: {e.stderr.decode()}")

    # Update annotation: shift word timings by t_in, clamp to [0, duration]
    anno_path = _annotation_path(video_id, segment_id)
    if anno_path.exists():
        parsed = _parse_annotation(anno_path)
        kept_words = []
        for w in parsed["words"]:
            ws = round(w["start"] - t_in, 3)
            we = round(w["end"]   - t_in, 3)
            if we <= 0 or ws >= duration:
                continue          # word outside trimmed range
            ws = max(0.0, ws)
            we = min(duration, we)
            kept_words.append({**w, "start": ws, "end": we})

        original = parsed.get("original")
        # Trimming changes which words are kept. The Text: line is rebuilt
        # from the kept words so it stays consistent with the WORD table —
        # otherwise WER vs Original would be misleading after trim.
        new_text = " ".join(w["word"] for w in kept_words) if kept_words else parsed["text"]

        lines = [f"Text:  {new_text}"]
        if original is not None:
            lines.append(f"Original:  {original}")
        lines.append(f"Conf:  {parsed.get('conf', 2)}")
        lines.append("WORD START END ASDSCORE")
        for w in kept_words:
            lines.append(f"{w['word']} {w['start']:.2f} {w['end']:.2f} {w.get('score', 0):.1f}")
        anno_path.write_text("\n".join(lines), encoding="utf-8")

        # Update CSV
        try:
            update_segment(
                segment_id,
                duration=duration,
                num_words=len(kept_words),
                text=new_text,
            )
        except Exception as e:
            logger.warning(f"CSV update after trim: {e}")

        _recompute_wer_in_csv(segment_id, original, new_text)

    review = _load_review()
    review[segment_id] = {**review.get(segment_id, {}), "trimmed": True}
    _save_review(review)
    return jsonify({"ok": True, "duration": duration})


def _parse_annotation(path: Path) -> dict:
    """Parse an LRS2 annotation file into {text, original, conf, words}.

    `original` is None for legacy annotations written before the
    `Original:` line was introduced — callers treat that as
    "no reference to compute WER against".
    """
    result: dict = {"text": "", "original": None, "conf": 2, "words": []}
    in_words = False
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        # 'Original:' must be checked before 'Text:' (longer prefix).
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
                    "word":  parts[0],
                    "start": float(parts[1]),
                    "end":   float(parts[2]),
                    "score": float(parts[3]),
                })
    return result


def _read_original_from_annotation(path: Path) -> Optional[str]:
    """Return the `Original:` line if present, else None (legacy file)."""
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Original:"):
            return line[len("Original:"):].strip()
    return None


# Confidence level used when a curator has explicitly approved or saved a
# segment after manual review. The auto-derived value (1/2/3 from Whisper
# confidence at export time) loses its meaning once a human has touched
# the transcript, so we promote it to 3 = high.
_HUMAN_REVIEWED_CONF_LEVEL = 3


def _recompute_video_validation_status(video_id: str) -> None:
    """Promote a video to status=validated when every one of its surviving
    segments has been explicitly approved or rejected.

    Reject deletes the segment row from segments_index.csv and the file
    from disk, so a rejected segment does not appear in `seg_ids` here —
    it counts as "removed" rather than "pending". A video is validated
    once all segments still on disk are approved.

    Idempotent: if status is already correct, no write happens. Also
    handles the demote case (validated → completed) when a segment is
    reverted back to pending.
    """
    if not video_id:
        return
    try:
        import pandas as pd
        seg_csv = _METADATA_DIR / "segments_index.csv"
        master_csv = _METADATA_DIR / "videos_master.csv"
        if not (seg_csv.exists() and master_csv.exists()):
            return
        seg_df = pd.read_csv(seg_csv)
        if "video_id" not in seg_df.columns or "segment_id" not in seg_df.columns:
            return
        seg_ids = (
            seg_df[seg_df["video_id"].astype(str) == video_id]["segment_id"]
            .astype(str).tolist()
        )
        if not seg_ids:
            # No segments left on disk for this video — nothing to validate.
            return

        review = _load_review()
        all_done = all(
            review.get(sid, {}).get("status") in ("approved", "rejected")
            for sid in seg_ids
        )

        master_df = pd.read_csv(master_csv)
        for col in master_df.select_dtypes(include=["float64", "object"]).columns:
            master_df[col] = master_df[col].astype(object)
        mask = master_df["video_id"].astype(str) == video_id
        if not mask.any():
            return
        current_status = str(master_df.loc[mask, "status"].iloc[0])

        if all_done and current_status != "validated":
            master_df.loc[mask, "status"] = "validated"
            master_df.to_csv(master_csv, index=False)
            logger.info(f"Video {video_id} → validated (all {len(seg_ids)} segments reviewed)")
        elif (not all_done) and current_status == "validated":
            # Curator reverted a segment — drop back to completed so the
            # video re-enters the review queue.
            master_df.loc[mask, "status"] = "completed"
            master_df.to_csv(master_csv, index=False)
            logger.info(f"Video {video_id} → completed (some segments back to pending)")
    except Exception as e:
        logger.warning(f"Validation status recompute failed for {video_id}: {e}")


def _set_conf_in_annotation(anno_path: Path, new_conf: int) -> None:
    """Rewrite ONLY the `Conf:` line in an annotation file.

    No-op if the file or the Conf line is missing. Other lines (Text,
    Original, WORD table) are preserved byte-for-byte.
    """
    if not anno_path.exists():
        return
    try:
        lines = anno_path.read_text(encoding="utf-8").splitlines()
    except OSError as e:
        logger.warning(f"Cannot read annotation for Conf rewrite: {e}")
        return
    new_lines = [
        f"Conf:  {new_conf}" if l.startswith("Conf:") else l
        for l in lines
    ]
    anno_path.write_text("\n".join(new_lines), encoding="utf-8")


def _recompute_wer_in_csv(segment_id: str, original: Optional[str], current: str) -> None:
    """Compute WER for one segment and persist it to segments_index.csv.

    No-op (with a debug log) if `original` is None — legacy annotations
    cannot be scored without a reference. Reviewer-visible side effect
    only — does not touch the annotation file.
    """
    try:
        from vsr_shared.wer_utils import compute_wer
        wer_val, ref_words = compute_wer(original, current)
    except Exception as e:
        logger.warning(f"WER computation failed for {segment_id}: {e}")
        return

    if wer_val is None:
        # Nothing to write — keep existing CSV value (likely null).
        return

    try:
        fields = {"wer": wer_val, "wer_word_count_ref": ref_words}
        # Keep original_text in CSV in sync with annotation (cheap and safe).
        if original is not None:
            fields["original_text"] = original
        update_segment(segment_id, **fields)
    except Exception as e:
        logger.warning(f"WER CSV update failed for {segment_id}: {e}")


@app.route("/api/stats/distributions")
def api_stats_distributions():
    """Histogram buckets + dataset-health counters for the Stats tab.

    All work is one CSV pass + N annotation reads (only when needed for
    Conf=1 counting). Designed to stay snappy for ~10k segments.
    """
    try:
        import pandas as pd
    except ImportError:
        return jsonify({"error": "pandas not installed"}), 500

    seg_csv = _METADATA_DIR / "segments_index.csv"
    if not seg_csv.exists():
        return jsonify({"wer_buckets": [], "duration_buckets": [], "health": {}})

    try:
        df = pd.read_csv(seg_csv)
    except Exception as e:
        return jsonify({"error": f"failed to read segments_index.csv: {e}"}), 500

    n = len(df)

    # WER histogram in 5% buckets [0, 100].
    wer_buckets: list = []
    if "wer" in df.columns:
        wer_series = pd.to_numeric(df["wer"], errors="coerce").dropna()
        if not wer_series.empty:
            edges = [i / 20 for i in range(21)]  # 0.00 .. 1.00 step 0.05
            for i in range(20):
                lo, hi = edges[i], edges[i + 1]
                mask = (wer_series >= lo) & ((wer_series < hi) if i < 19 else (wer_series <= hi))
                wer_buckets.append({
                    "lo_pct": round(lo * 100, 1),
                    "hi_pct": round(hi * 100, 1),
                    "count": int(mask.sum()),
                })

    # Duration histogram in 1-second buckets [1, 15].
    duration_buckets: list = []
    if "duration" in df.columns:
        d = pd.to_numeric(df["duration"], errors="coerce").dropna()
        if not d.empty:
            for sec in range(1, 15):
                lo, hi = float(sec), float(sec + 1)
                mask = (d >= lo) & ((d < hi) if sec < 14 else (d <= hi))
                duration_buckets.append({"lo_s": sec, "hi_s": sec + 1, "count": int(mask.sum())})

    # Dataset health.
    def _missing(col: str) -> int:
        if col not in df.columns:
            return n
        s = df[col]
        return int(s.isna().sum() + (s.astype(str) == "").sum())

    # Conf=1 needs to read annotation files. Skip the scan when annotation_path is missing.
    conf1_count = 0
    annotations_scanned = 0
    if "annotation_path" in df.columns:
        for p in df["annotation_path"].dropna().astype(str):
            anno_path = (_PROJECT_ROOT / p) if not Path(p).is_absolute() else Path(p)
            if not anno_path.exists():
                continue
            annotations_scanned += 1
            for line in anno_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("Conf:"):
                    try:
                        if int(line[len("Conf:"):].strip()) == 1:
                            conf1_count += 1
                    except ValueError:
                        pass
                    break

    def _pct(v: int, denom: int) -> float:
        return round(100.0 * v / denom, 1) if denom else 0.0

    missing_speaker = _missing("speaker_id")
    missing_wer = int(pd.to_numeric(df.get("wer", pd.Series(dtype=float)), errors="coerce").isna().sum())

    health = {
        "n_segments": n,
        "missing_speaker_id": {"count": missing_speaker, "pct": _pct(missing_speaker, n)},
        "missing_wer": {"count": missing_wer, "pct": _pct(missing_wer, n)},
        "conf_1_low": {
            "count": conf1_count,
            "annotations_scanned": annotations_scanned,
            "pct_of_scanned": _pct(conf1_count, annotations_scanned),
        },
    }

    return jsonify({
        "wer_buckets": wer_buckets,
        "duration_buckets": duration_buckets,
        "health": health,
    })


@app.route("/api/speakers")
def api_speakers():
    """Return the speakers registry with fresh aggregates."""
    try:
        from vsr_shared.speakers_registry import load_speakers, recompute_aggregates
    except ImportError as e:
        return jsonify({"error": f"speakers_registry import failed: {e}"}), 500
    try:
        # Refresh aggregates from segments_index before returning — this keeps
        # the response cheap (single CSV pass) and saves the curator from
        # remembering to click a "recompute" button after every batch.
        recompute_aggregates(_METADATA_DIR)
        df = load_speakers(_METADATA_DIR).fillna("")
        records = [{str(k): (str(v) if v != "" else "") for k, v in r.items()}
                   for r in df.to_dict("records")]
        return jsonify({"speakers": records})
    except Exception as e:
        logger.exception("api_speakers failed")
        return jsonify({"speakers": [], "error": str(e)}), 500


_SPEAKER_THUMB_DIR = _PROJECT_ROOT / "data" / "cache" / "speaker_thumbs"


@app.route("/api/speaker/<speaker_id>/thumbnail")
def api_speaker_thumbnail(speaker_id: str):
    """Serve a small JPEG preview of a speaker's face (for the Stats UI).

    Picks the FIRST segment in segments_index.csv assigned to this
    speaker, extracts a mid-frame from its face_crop video via ffmpeg,
    and caches the result on disk. Subsequent requests just hit the cache.

    Returns 404 if the speaker has no surviving segment with a face_crop
    video on disk.
    """
    if not speaker_id or "/" in speaker_id or "\\" in speaker_id or ".." in speaker_id:
        # Path-traversal guard: speaker_id becomes part of a filename below.
        return jsonify({"error": "invalid speaker_id"}), 400

    # ?seg=N → sample the N-th segment (evenly across the speaker's catalogue)
    # so the modal can show several different snapshots of the same person.
    # Default 0 = first available segment.
    try:
        seg_index = max(0, int(request.args.get("seg", "0")))
    except ValueError:
        seg_index = 0

    _SPEAKER_THUMB_DIR.mkdir(parents=True, exist_ok=True)
    cached = _SPEAKER_THUMB_DIR / f"{speaker_id}_seg{seg_index}.jpg"

    # Cache invalidation: if the file is older than 30 days, regenerate
    # so a re-clustering pass eventually refreshes the visual.
    import time
    if cached.exists() and (time.time() - cached.stat().st_mtime) < 30 * 24 * 3600:
        return send_file(str(cached), mimetype="image/jpeg", conditional=True)

    # Find a segment for this speaker.
    try:
        import pandas as pd
        seg_csv = _METADATA_DIR / "segments_index.csv"
        if not seg_csv.exists():
            return jsonify({"error": "segments_index.csv not found"}), 404
        df = pd.read_csv(seg_csv)
        if "speaker_id" not in df.columns:
            return jsonify({"error": "no speaker_id column"}), 404
        rows = df[df["speaker_id"].astype(str) == speaker_id]
        if rows.empty:
            return jsonify({"error": "no segments for this speaker"}), 404

        # Build the pool of segments whose face_crop actually exists on disk.
        candidates: list[Path] = []
        for _, r in rows.iterrows():
            seg_id = str(r["segment_id"])
            vid = str(r["video_id"])
            cand = _PROCESSED_DIR / vid / "face_crop" / f"{seg_id}.mp4"
            if cand.exists():
                candidates.append(cand)
        if not candidates:
            return jsonify({"error": "no face_crop video on disk for any segment"}), 404

        # Pick the N-th sample, spread evenly. seg=0 → first, seg=k → ~k/(K-1)
        # of the way through. Out-of-range seg falls back to last sample.
        n = len(candidates)
        if seg_index <= 0 or n == 1:
            chosen_video = candidates[0]
        elif seg_index >= n:
            chosen_video = candidates[-1]
        else:
            # Spread: when the caller asks for 4 samples, they pass seg=0,1,2,3.
            # We map those onto evenly-spaced indices in the candidate list.
            step = max(1, n // 4)
            idx = min(n - 1, seg_index * step)
            chosen_video = candidates[idx]
    except Exception as e:
        logger.exception("thumbnail lookup failed")
        return jsonify({"error": str(e)}), 500

    # Extract one mid-frame as JPEG via ffmpeg.
    try:
        # Probe duration to get the midpoint timestamp.
        ffprobe = FFMPEG.replace("ffmpeg", "ffprobe").replace("ffmpeg.exe", "ffprobe.exe")
        try:
            duration = float(subprocess.run(
                [ffprobe, "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(chosen_video)],
                capture_output=True, text=True, check=True, timeout=10,
            ).stdout.strip() or 1.0)
        except Exception:
            duration = 1.0  # bail to first frame if probe fails

        ts = max(0.0, duration / 2.0)
        subprocess.run(
            [FFMPEG, "-y", "-ss", f"{ts:.3f}", "-i", str(chosen_video),
             "-vframes", "1", "-q:v", "2", str(cached), "-loglevel", "error"],
            check=True, timeout=15,
        )
        if not cached.exists() or cached.stat().st_size == 0:
            return jsonify({"error": "thumbnail extraction produced empty file"}), 500
        return send_file(str(cached), mimetype="image/jpeg", conditional=True)
    except subprocess.CalledProcessError as e:
        logger.warning(f"ffmpeg failed for thumbnail {speaker_id}: {e}")
        return jsonify({"error": "ffmpeg failed"}), 500


@app.route("/api/speaker/<speaker_id>", methods=["POST"])
@_with_metadata_lock
def api_speaker_update(speaker_id: str):
    """Curator edits to a speaker row (name / gender / age / accent)."""
    if not speaker_id:
        return jsonify({"error": "speaker_id is required"}), 400
    data = request.json or {}
    editable = {
        k: data[k] for k in ("speaker_name", "gender", "age_group", "accent_region")
        if k in data
    }
    try:
        from vsr_shared.speakers_registry import upsert_speaker
        upsert_speaker(_METADATA_DIR, speaker_id, editable)
        return jsonify({"ok": True, "speaker_id": speaker_id, "fields": editable})
    except Exception as e:
        logger.exception("api_speaker_update failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/segment/<segment_id>/speaker", methods=["POST"])
@_with_metadata_lock
def api_segment_set_speaker(segment_id: str):
    """Manually reassign a single segment to a different speaker_id.

    Used by the Review/Explorer UI when the automated clustering put a
    segment under the wrong speaker. If the target id doesn't exist yet,
    a stub registry row is created so it shows up in the Speakers panel.

    Body: {"speaker_id": "<existing_or_new_id>"}
    """
    data = request.json or {}
    new_sid = (data.get("speaker_id") or "").strip()
    if not new_sid:
        return jsonify({"error": "speaker_id is required"}), 400

    try:
        import pandas as pd
        from vsr_shared.speakers_registry import ensure_speaker_exists, recompute_aggregates

        seg_csv = _METADATA_DIR / "segments_index.csv"
        if not seg_csv.exists():
            return jsonify({"error": "segments_index.csv not found"}), 404
        df = pd.read_csv(seg_csv)
        mask = df["segment_id"].astype(str) == segment_id
        if not mask.any():
            return jsonify({"error": "segment not found"}), 404

        old_sid = str(df.loc[mask, "speaker_id"].iloc[0]) if "speaker_id" in df.columns else ""
        if "speaker_id" not in df.columns:
            df["speaker_id"] = ""
        df.loc[mask, "speaker_id"] = new_sid
        df.to_csv(seg_csv, index=False)

        # Make sure the new speaker has a registry stub, then refresh
        # aggregates so the move is reflected immediately in the UI.
        ensure_speaker_exists(_METADATA_DIR, new_sid)
        recompute_aggregates(_METADATA_DIR)

        return jsonify({
            "ok": True,
            "segment_id": segment_id,
            "old_speaker_id": old_sid,
            "speaker_id": new_sid,
        })
    except Exception as e:
        logger.exception("api_segment_set_speaker failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/media/<video_id>/<segment_id>")
def api_media(video_id: str, segment_id: str):
    """Serve a processed segment video.

    Query param ?type=mouth serves the mouth crop; default is face crop.
    """
    crop_type = request.args.get("type", "face")
    subdir = "mouth_crop" if crop_type == "mouth" else "face_crop"
    video_path = _PROCESSED_DIR / video_id / subdir / f"{segment_id}.mp4"
    if not video_path.exists():
        return jsonify({"error": "File not found"}), 404
    return _stream_h264(str(video_path))


_VIDEO_CACHE_DIR = _PROJECT_ROOT / "data" / "cache" / "h264"


def _stream_h264(filepath: str) -> Response:
    """Serve a browser-playable H.264 MP4 WITH seek support.

    We transcode once to disk (cached by source mtime) and then serve the
    cached file via `send_file(..., conditional=True)`, which adds
    `Accept-Ranges: bytes` + `Content-Length` headers so the <video>
    element can issue Range requests and honor `currentTime = X` seeks.

    The previous implementation streamed a live ffmpeg pipe which lacked
    length/ranges, so browsers silently refused to seek and word-play
    buttons seemed to "start from 0".
    """
    src = Path(filepath)
    if not src.exists():
        return jsonify({"error": "File not found"}), 404

    _VIDEO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Unique cache name: <video_id>_<filename>. Unique because each segment
    # ID already embeds the source video_id, and parent folder disambiguates
    # face vs mouth crops.
    crop_kind = src.parent.name  # "face_crop" or "mouth_crop"
    cache_name = f"{crop_kind}__{src.parent.parent.name}__{src.stem}.mp4"
    cached = _VIDEO_CACHE_DIR / cache_name

    needs_encode = (
        not cached.exists()
        or cached.stat().st_mtime < src.stat().st_mtime
        or cached.stat().st_size == 0
    )

    if needs_encode:
        cmd = [
            FFMPEG, "-y", "-i", str(src),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-loglevel", "error",
            str(cached),
        ]
        try:
            subprocess.run(cmd, check=True, timeout=60)
        except Exception as e:
            logger.warning(f"Transcode failed for {src}: {e}")
            # Fallback: serve the raw file directly (may not be seekable
            # in all browsers, but at least we try).
            if src.exists():
                return send_file(str(src), mimetype="video/mp4", conditional=True)
            return jsonify({"error": "Transcode failed"}), 500

    return send_file(str(cached), mimetype="video/mp4", conditional=True)


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  Romanian VSR Dataset Pipeline  →  http://localhost:5001\n")
    # IMPORTANT: keep `use_reloader=False` when running the pipeline.
    # Flask's debug auto-reloader watches every file under the project root
    # and restarts the server on ANY change. The pipeline writes to
    # videos_master.csv, segments_index.csv, checkpoints, log files,
    # face/mouth crops, annotations — so the reloader would kick in
    # constantly, killing the subprocess mid-run. We keep `debug=True` for
    # nicer error pages but disable the watcher.
    app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=False)
