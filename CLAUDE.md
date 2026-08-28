# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

A pipeline that builds a Romanian Visual Speech Recognition (VSR) dataset from YouTube videos. It downloads videos, splits them into sentence-level clips via VAD, runs face detection + active speaker detection, transcribes with WhisperX, and exports in LRS2-compatible format. A Tkinter GUI lets annotators review/curate the exported clips.

## Commands

```bash
# Setup (run once after cloning)
python -m venv .venv
source .venv/Scripts/activate       # Windows Git Bash
pip install -r requirements.txt

# Fetch model weights (manifest: config/models.yaml; --check / --pin / --only)
python backend/tools/fetch_models.py

# Machine readiness report (green/red; same checks as the future Docker healthcheck)
python backend/tools/doctor.py

# Initialize directory structure + dataset.db
python backend/orchestrator/cli.py init ./data

# Process a single video
python backend/orchestrator/cli.py single ro_001 "https://youtube.com/watch?v=..."

# Batch process all 'pending' videos from the Excel
python backend/orchestrator/cli.py batch data/catalog/videos_master.csv

# Filter by status or specific IDs
python backend/orchestrator/cli.py batch data/catalog/videos_master.csv --status failed
python backend/orchestrator/cli.py batch data/catalog/videos_master.csv --video-id ro_001 ro_005

# Resume interrupted runs
python backend/orchestrator/cli.py resume ro_003
python backend/orchestrator/cli.py resume-batch data/catalog/videos_master.csv

# Rebuild Excel stats from disk (no reprocessing)
python backend/orchestrator/cli.py sync-excel data/catalog/videos_master.csv

# Dataset statistics
python backend/orchestrator/cli.py stats data/catalog/videos_master.csv

# Review GUI
python backend/tools/review_gui.py --data ./data
python backend/tools/review_gui.py --data ./data --video ro_003
```

All commands accept `--config config/config.yaml` (default). `batch` and `single` also accept `--cookies FILE` or `--cookies-from-browser BROWSER` to bypass YouTube bot-detection.

**Windows / WSL Tk fix** (if GUI crashes on startup):
```bash
export TCL_LIBRARY="C:/Program Files/Python313/tcl/tcl8.6"
export TK_LIBRARY="C:/Program Files/Python313/tcl/tk8.6"
```

## Architecture

### Repository layout (microservices)

The backend is organized as one folder per pipeline stage under `backend/services/` (downloader, segmenter, face_tracker, speaker_detector, mouth_exporter, quality_indexer, transcript_refiner), each with its own `requirements.txt` — each service is a future Docker image. Shared code (CSV schemas, speakers registry, WER utils) lives in `backend/shared/vsr_shared/`. The orchestrator (`backend/orchestrator/`) currently imports the services **in-process** (modular monolith); services communicate through file contracts on `data/` (clips.json, checkpoints, CSVs), not HTTP. Curation utilities in `backend/tools/` are not services and never get containerized. Python imports resolve with `backend/` and `backend/shared/` on `sys.path` (the CLI and every tool bootstrap this themselves). The improvement roadmap lives in `plan.md` / `plan.html`; working notes in `PIPELINE_NOTES.md`.

### Data flow (v3 — sentence strategy, the default)

```
YouTube URL
  → backend/services/downloader/youtube_downloader.py   (yt-dlp, CC filter)
  → data/raw/{video_id}.mp4

  Segmentation (VSRPipeline._segment_video_into_sentences):
    → Silero VAD on the FULL audio               (speech regions + pauses)
    → WhisperX ONCE on the full audio            (all words + punctuation + timestamps)
    → backend/services/segmenter/sentence_segmenter.py
         sentence windows: punctuation + pauses; over-long sentences split
         at the largest inter-word pause — NEVER mid-word, no blind 15s cut
    → backend/services/segmenter/vad_splitter.py  cut_sentence_windows()
    → data/clips/{video_id}/{clip_id}.mp4 + .wav + clips.json (v2, with words)
    → Whisper unloaded from VRAM

  Per clip (VSRPipeline._process_clip — no per-clip Whisper anymore):
    → backend/services/face_tracker/face_processing.py    (RetinaFace + Kalman)
    → [face visibility filter]
    → transcript comes from the manifest words (shifted to clip time)
    → backend/services/speaker_detector/asd.py            (TalkNet — loud fallback policy)
    → backend/services/speaker_detector/syncnet.py        (SyncNet — loud fallback policy)
    → backend/services/mouth_exporter/lrs2_formatter.py
         face_crop 256×256 + mouth_crop 96×96 (MediaPipe dense lip landmarks,
         One-Euro smoothing, roll-aligned grayscale) via single-encode ffmpeg pipe
    → ArcFace identity evidence sampled from the face crop (speaker_identity)
    → quality_tier A/B/C computed (quality_indexer/quality_tiers.py)

  End of video:
    → backend/services/quality_indexer/speaker_identity.py
         DBSCAN clustering → one speaker_id per person (recurring anchor = one id),
         per-cluster gender + numeric age_estimate, optional cross-video re-ID
         via centroids in the speakers.centroid column of dataset.db

  → data/processed/{video_id}/{face_crop,mouth_crop}/{segment_id}.mp4
  → data/processed/{video_id}/audio/{segment_id}.wav + text/{segment_id}.txt
  → data/catalog/dataset.db  (SQLite WAL — videos, segments, words, speakers,
                              segment_embeddings, dropped_clips, jobs + views;
                              CSV mirrors in data/catalog/ until FastAPI)

  Later, standalone (second opinion on transcripts):
    → python backend/services/transcript_refiner/main.py   (large-v3 int8 → needs_review)
```

Legacy behaviour is still available with `segmentation.strategy: "vad"`.

### Key classes and their responsibilities

| Class | File | Role |
|---|---|---|
| `VSRPipeline` | `backend/orchestrator/pipeline.py` | Top-level orchestrator; owns lazy-loaded component singletons |
| `PipelineConfig` | `backend/orchestrator/pipeline.py` | Dataclass parsed from `config/config.yaml` via `from_yaml()` |
| `VADSplitter` | `backend/services/segmenter/vad_splitter.py` | Silero VAD → `VideoClip` list |
| `WhisperTranscriber` | `backend/services/segmenter/transcribe.py` | WhisperX → `TranscribedSegment` with word-level timings |
| `VideoProcessor` | `backend/services/face_tracker/face_processing.py` | RetinaFace + Kalman → `FaceTrack` list |
| `TalkNetASD` | `backend/services/speaker_detector/asd.py` | Per-frame active-speaker scores → `ASDResult` |
| `SyncNetVerifier` | `backend/services/speaker_detector/syncnet.py` | Audio-visual sync confidence → `SyncResult` |
| `LRS2Exporter` | `backend/services/mouth_exporter/lrs2_formatter.py` | Face-crops video + writes annotation file → `ExportedSegment` |

### Resume / checkpoint mechanism

After each clip is processed, the result is written to `data/clips/{video_id}/.checkpoint.json`. When `resume_video()` is called (or when `process_video()` finds existing clips), it reads this file to skip already-done clips and reconstruct already-exported `ExportedSegment` objects from disk without re-running any ML inference.

### Catalog (storage v2)

ALL metadata lives in `data/catalog/dataset.db` (SQLite; WAL, with automatic fallback to the classic rollback journal on Windows bind mounts) — written automatically by `CatalogWriter` through `vsr_shared/catalog_db.py` (the only file that knows SQL), one transaction per clip. CSVs are **exports only** — nothing in the pipeline reads or writes them; generate them on demand with `python backend/tools/export_catalog.py`. Batch/resume selection, stats, bulk import and the API all read the `videos`/`segments` tables directly. Other tools: `import_videos.py` (curator CSV → DB), `verify_dataset.py` (disk↔DB consistency report). The old Windows pending-updates queue is gone — SQLite doesn't lock the way Excel does. Automatic `dataset.db` backups (last 5) land in `data/catalog/backups/` after each video.

### External model dependencies

- **TalkNet-ASD** — clone into `TalkNet-ASD/` at the repo root (NO pip install — the repo has no setup.py; cli/worker put it on sys.path, Docker via PYTHONPATH). Weights at `models/talknet_asd.pth`.
- **SyncNet** — bundled in `backend/services/speaker_detector/syncnet.py`. Weights at `models/syncnet_v2.pth`.
- **WhisperX** — weights downloaded automatically on first use.
- **RetinaFace** — loaded via the `retinaface-pytorch` PyPI package.
- **Silero VAD** — loaded via `torch.hub` on first use.

### Annotation format

```
Text:  ACESTE ZILE CÂND GĂTEȘTI
Conf:  2
WORD START END ASDSCORE
ACESTE 0.00 0.42 8.1
...
```

`Conf`: 1=low, 2=medium, 3=high. All timings are relative to the clip start.

## Frontend (Web UI) + API + worker

The web UI is a React (Vite) app in `frontend/` served by the FastAPI backend
(`backend/api/` — torch-free) which talks to `data/catalog/dataset.db`. Pipeline
runs go through the **jobs queue**: the API inserts a row in the `jobs` table,
the worker (`backend/worker/` — the only GPU process) claims it atomically,
executes the existing `VSRPipeline` in-process, heartbeats progress back, and
streams its log to `data/logs/job_{id}.log`.

```bash
# Start the API (serves the React build + /api/* + OpenAPI at /docs).
# Launchers work from ANY directory (they bootstrap sys.path themselves):
python backend/run_api.py         # --port 9000 pentru alt port

# Start the queue worker (separate terminal; the GPU process)
python backend/run_worker.py

# Then open http://localhost:8000
```

`/api/start`, `/api/status`, `/api/stop`, `/api/bulk_import` keep the shapes
the React build already calls (compat layer over the queue). The old Flask app
was DELETED after UI parity was confirmed on the processing machine.

Four tabs:
- **Process** — select run mode (Batch Pending / Batch Failed / Resume / Single), fill in parameters, press START. Live log streams every 1.2 s.
- **Review** — one-segment-at-a-time manual curation: approve / reject / edit transcript. Keyboard: A=approve, R=reject, S=skip, E=edit, ←/→=navigate.
- **Explorer** — browse all exported segments, search by text/ID, filter by ASD score, preview face-cropped video.
- **Stats** — pipeline flow gauges, quality metrics, paginated video registry with status filter.

Review decisions live on the segment rows (`review_status` / `reviewed_at` / `transcript_edited` / `trimmed` in dataset.db; the old review_status.json imports once via `backend/tools/import_review_status.py`). Reject deletes the media files but KEEPS the DB row with `review_status='rejected'` — lists and aggregates filter it out.

## Configuration

All pipeline parameters live in `config/config.yaml`. Key knobs:

- `audio.whisper.model` — `medium` (default) through `large-v3`; trade speed for accuracy
- `audio.whisper.device` — `cuda` or `cpu` (auto-falls back to CPU if CUDA unavailable)
- `vad_splitting.split_threshold` — silence gap (s) that starts a new clip
- `vad_splitting.cleanup_clips` — delete `data/clips/` after processing to save space
- `clip_filters.face_visibility_threshold` — drop clips where face is visible in < this fraction of frames (default 0.80)
- `download.cookies_file` / `download.cookies_from_browser` — YouTube bot-detection bypass
