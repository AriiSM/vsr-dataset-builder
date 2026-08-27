"""
Catalog database — the single source of truth for ALL pipeline metadata.

Storage v2 (Faza 6.5): media files live on disk under processed/{video_id}/,
EVERYTHING measured about them lives here, in data/catalog/dataset.db.
The pipeline writes rows automatically (transaction per clip); CSVs become
on-demand exports (backend/tools/export_catalog.py).

Tables (see the ER diagram in plan.html):
    videos              one row per source video: state machine + provenance
    segments            one row per exported segment — THE dataset
    words               per-word timings + scores (queryable, indexed)
    speakers            identity + curator-edited demographics + centroid BLOB
    segment_embeddings  per-segment ArcFace evidence (512×float32 BLOB)
    dropped_clips       every rejection WITH its reason (survives cleanup)
    jobs                the API↔worker queue (Faza 7 packaging design)
Views:
    speaker_stats · video_stats · dataset_overview — aggregates computed
    live from segments: always correct, never recomputed by hand.

Design rules:
    - This module is the ONLY place that knows SQL.
    - WAL journal + busy_timeout=5000 + foreign_keys=ON on every connection.
    - Short transactions (one per clip / one per update).
    - No path columns for media: paths derive from segment_id by convention.
      (annotation_path/video_path survive only in the CSV *export* shape.)
    - stdlib sqlite3 only — no ORM, no new dependency.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id                 TEXT PRIMARY KEY,
    youtube_url              TEXT DEFAULT '',
    source                   TEXT DEFAULT '',
    source_channel           TEXT DEFAULT '',
    license                  TEXT DEFAULT '',
    region                   TEXT DEFAULT '',
    title                    TEXT DEFAULT '',
    duration_seconds         REAL,
    speaker_id               TEXT DEFAULT '',
    num_speakers             INTEGER,
    gender                   TEXT DEFAULT '',
    age_group                TEXT DEFAULT '',
    environment              TEXT DEFAULT '',
    background_noise         TEXT DEFAULT '',
    status                   TEXT DEFAULT 'pending',
    processed_date           TEXT DEFAULT '',
    total_segments           INTEGER,
    total_duration_extracted REAL,
    avg_asd_score            REAL,
    avg_syncnet_conf         REAL,
    error_message            TEXT DEFAULT '',
    pipeline_version         TEXT DEFAULT '',
    config_hash              TEXT DEFAULT '',
    whisper_model            TEXT DEFAULT '',
    segmentation_strategy    TEXT DEFAULT '',
    created_at               TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS segments (
    segment_id               TEXT PRIMARY KEY,
    video_id                 TEXT NOT NULL REFERENCES videos(video_id),
    clip_id                  TEXT DEFAULT '',
    speaker_id               TEXT DEFAULT '',
    track_id                 INTEGER,
    start_time               REAL,
    end_time                 REAL,
    duration                 REAL,
    text                     TEXT DEFAULT '',
    original_text            TEXT DEFAULT '',
    num_words                INTEGER,
    num_chars                INTEGER,
    asd_score                REAL,
    asd_method               TEXT DEFAULT '',
    syncnet_conf             REAL,
    syncnet_method           TEXT DEFAULT '',
    whisper_conf             REAL,
    whisper_conf_min         REAL,
    whisper_conf_p25         REAL,
    face_bbox                TEXT DEFAULT '',
    face_visibility_ratio    REAL,
    head_pose_avg            TEXT DEFAULT '',
    mouth_landmark_fail_rate REAL,
    mouth_roi_method         TEXT DEFAULT '',
    quality_tier             TEXT DEFAULT '',
    audio_speaker_label      TEXT DEFAULT '',
    av_speaker_mismatch      INTEGER,           -- NULL=not judged, 0/1
    boundary_start_type      TEXT DEFAULT '',
    boundary_end_type        TEXT DEFAULT '',
    text_largev3             TEXT DEFAULT '',
    wer_medium_vs_large      REAL,
    needs_review             INTEGER,
    wer                      REAL DEFAULT 0.0,
    wer_word_count_ref       INTEGER,
    review_status            TEXT DEFAULT '',   -- '' | approved | rejected
    reviewed_at              TEXT DEFAULT '',
    transcript_edited        INTEGER DEFAULT 0,
    trimmed                  INTEGER DEFAULT 0,
    split                    TEXT DEFAULT '',
    created_at               TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_segments_video   ON segments(video_id);
CREATE INDEX IF NOT EXISTS idx_segments_tier    ON segments(quality_tier);
CREATE INDEX IF NOT EXISTS idx_segments_speaker ON segments(speaker_id);

CREATE TABLE IF NOT EXISTS words (
    segment_id TEXT NOT NULL REFERENCES segments(segment_id) ON DELETE CASCADE,
    word_index INTEGER NOT NULL,
    word       TEXT NOT NULL,
    start_time REAL,
    end_time   REAL,
    confidence REAL,
    asd_score  REAL,
    PRIMARY KEY (segment_id, word_index)
);
CREATE INDEX IF NOT EXISTS idx_words_word ON words(word);

CREATE TABLE IF NOT EXISTS speakers (
    speaker_id        TEXT PRIMARY KEY,
    speaker_name      TEXT DEFAULT '',
    gender            TEXT DEFAULT '',
    gender_confidence REAL,
    age_estimate      REAL,
    age_std           REAL,
    age_group         TEXT DEFAULT '',
    accent_region     TEXT DEFAULT '',
    identity_match    TEXT DEFAULT '',
    centroid          BLOB
);

CREATE TABLE IF NOT EXISTS segment_embeddings (
    segment_id TEXT PRIMARY KEY REFERENCES segments(segment_id) ON DELETE CASCADE,
    embedding  BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS dropped_clips (
    video_id        TEXT NOT NULL REFERENCES videos(video_id),
    clip_id         TEXT NOT NULL,
    reason          TEXT DEFAULT '',
    start_time      REAL,
    end_time        REAL,
    face_visibility REAL,
    whisper_conf    REAL,
    asd_score       REAL,
    syncnet_conf    REAL,
    created_at      TEXT DEFAULT '',
    PRIMARY KEY (video_id, clip_id)
);

CREATE TABLE IF NOT EXISTS jobs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    type         TEXT NOT NULL,                -- single | batch | resume | sync
    params_json  TEXT DEFAULT '{}',
    status       TEXT NOT NULL DEFAULT 'pending',
        -- pending | running | done | failed | interrupted
        -- | cancel_requested | cancelled
    created_at   TEXT DEFAULT '',
    started_at   TEXT DEFAULT '',
    finished_at  TEXT DEFAULT '',
    claimed_by   TEXT DEFAULT '',
    heartbeat_at TEXT DEFAULT '',
    progress_json TEXT DEFAULT '{}',
    error        TEXT DEFAULT '',
    git_sha      TEXT DEFAULT '',
    config_hash  TEXT DEFAULT '',
    log_path     TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);

CREATE VIEW IF NOT EXISTS speaker_stats AS
    SELECT s.speaker_id,
           COUNT(DISTINCT s.video_id) AS num_videos,
           COUNT(*)                   AS num_segments,
           ROUND(SUM(s.duration), 2)  AS total_duration_s,
           ROUND(AVG(s.asd_score), 3) AS avg_asd,
           ROUND(AVG(s.wer), 4)       AS avg_wer
    FROM segments s
    WHERE s.speaker_id != ''
    GROUP BY s.speaker_id;

CREATE VIEW IF NOT EXISTS video_stats AS
    SELECT s.video_id,
           COUNT(*)                        AS num_segments,
           ROUND(SUM(s.duration), 2)       AS total_duration_s,
           ROUND(AVG(s.asd_score), 3)      AS avg_asd,
           ROUND(AVG(s.syncnet_conf), 3)   AS avg_syncnet,
           SUM(s.quality_tier = 'A')       AS tier_a,
           SUM(s.quality_tier = 'B')       AS tier_b,
           SUM(s.quality_tier = 'C')       AS tier_c
    FROM segments s
    GROUP BY s.video_id;

CREATE VIEW IF NOT EXISTS dataset_overview AS
    SELECT COUNT(*)                             AS num_segments,
           COUNT(DISTINCT video_id)             AS num_videos,
           COUNT(DISTINCT speaker_id)           AS num_speakers,
           ROUND(SUM(duration) / 3600.0, 2)     AS total_hours,
           SUM(quality_tier = 'A')              AS tier_a,
           SUM(quality_tier = 'B')              AS tier_b,
           SUM(quality_tier = 'C')              AS tier_c
    FROM segments;
"""


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class CatalogDatabase:
    """Owns the sqlite3 connection and the schema. One instance per process."""

    def __init__(self, db_path: Path, *, check_same_thread: bool = True):
        # check_same_thread=False is for callers that guarantee one-user-at-a-
        # time themselves (the API's connection pool); everyone else keeps the
        # sqlite3 default of binding the connection to its creating thread.
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=check_same_thread)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

        self.videos = VideosRepo(self._conn)
        self.segments = SegmentsRepo(self._conn)
        self.speakers = SpeakersRepo(self._conn)
        self.dropped = DroppedClipsRepo(self._conn)
        self.jobs = JobsRepo(self._conn)

    def _init_schema(self):
        with self._conn:
            self._conn.executescript(_SCHEMA)
            current = self._conn.execute("PRAGMA user_version").fetchone()[0]
            if current < SCHEMA_VERSION:
                self._migrate(current)
                self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def _migrate(self, from_version: int):
        """Additive migrations for databases created by older schema versions.
        (CREATE TABLE IF NOT EXISTS leaves existing tables untouched, so new
        columns must arrive via ALTER.)"""
        columns = {r[1] for r in self._conn.execute("PRAGMA table_info(segments)")}
        for name, ddl in [
            ("transcript_edited", "INTEGER DEFAULT 0"),
            ("trimmed", "INTEGER DEFAULT 0"),
        ]:
            if name not in columns:
                self._conn.execute(f"ALTER TABLE segments ADD COLUMN {name} {ddl}")

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    def backup_to(self, target_path: Path) -> None:
        """Consistent online backup (sqlite backup API) — safe under WAL."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(target_path)) as target:
            self._conn.backup(target)

    def close(self):
        self._conn.close()


class _Repo:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn


class VideosRepo(_Repo):
    def ensure_exists(self, video_id: str):
        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO videos (video_id, created_at) VALUES (?, ?)",
                (video_id, _now()),
            )

    def set_status(self, video_id: str, status: str):
        self.ensure_exists(video_id)
        with self._conn:
            self._conn.execute(
                "UPDATE videos SET status = ? WHERE video_id = ?",
                (status, video_id),
            )

    def update_fields(self, video_id: str, fields: Dict):
        """Update a subset of columns (keys validated against the table)."""
        if not fields:
            return
        self.ensure_exists(video_id)
        columns = {r[1] for r in self._conn.execute("PRAGMA table_info(videos)")}
        safe = {k: v for k, v in fields.items() if k in columns and k != "video_id"}
        if not safe:
            return
        assignments = ", ".join(f"{k} = ?" for k in safe)
        with self._conn:
            self._conn.execute(
                f"UPDATE videos SET {assignments} WHERE video_id = ?",
                (*safe.values(), video_id),
            )

    def get(self, video_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM videos WHERE video_id = ?", (video_id,)
        ).fetchone()
        return dict(row) if row else None

    def all(self) -> List[dict]:
        return [dict(r) for r in self._conn.execute(
            "SELECT * FROM videos ORDER BY video_id")]

    def region(self, video_id: str) -> str:
        row = self._conn.execute(
            "SELECT region FROM videos WHERE video_id = ?", (video_id,)
        ).fetchone()
        return (row["region"] or "").strip() if row else ""


class SegmentsRepo(_Repo):
    def upsert(self, row: Dict, words: Optional[List[Dict]] = None):
        """Insert or replace ONE segment + its words, in one transaction.

        Called per clip right after export — the row becomes visible to
        readers only when its files are already complete on disk.
        """
        columns = {r[1] for r in self._conn.execute("PRAGMA table_info(segments)")}
        safe = {k: v for k, v in row.items() if k in columns}
        safe.setdefault("created_at", _now())
        names = ", ".join(safe)
        placeholders = ", ".join("?" for _ in safe)
        with self._conn:
            self._conn.execute(
                f"INSERT OR REPLACE INTO segments ({names}) VALUES ({placeholders})",
                tuple(safe.values()),
            )
            if words is not None:
                segment_id = safe["segment_id"]
                self._conn.execute(
                    "DELETE FROM words WHERE segment_id = ?", (segment_id,)
                )
                self._conn.executemany(
                    "INSERT INTO words (segment_id, word_index, word, start_time,"
                    " end_time, confidence, asd_score) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        (segment_id, i, w.get("word", ""), w.get("start"),
                         w.get("end"), w.get("score"), w.get("asd_score"))
                        for i, w in enumerate(words)
                    ],
                )

    def replace_for_video(self, video_id: str, rows: Iterable[Dict]):
        """End-of-video authoritative rewrite (final speaker_id / tier / av)."""
        with self._conn:
            for row in rows:
                columns = {r[1] for r in self._conn.execute("PRAGMA table_info(segments)")}
                safe = {k: v for k, v in row.items() if k in columns}
                safe.setdefault("created_at", _now())
                assignments = ", ".join(f"{k} = ?" for k in safe if k != "segment_id")
                values = [v for k, v in safe.items() if k != "segment_id"]
                updated = self._conn.execute(
                    f"UPDATE segments SET {assignments} WHERE segment_id = ?",
                    (*values, safe["segment_id"]),
                ).rowcount
                if not updated:
                    names = ", ".join(safe)
                    placeholders = ", ".join("?" for _ in safe)
                    self._conn.execute(
                        f"INSERT INTO segments ({names}) VALUES ({placeholders})",
                        tuple(safe.values()),
                    )

    def delete_for_video(self, video_id: str):
        with self._conn:
            self._conn.execute("DELETE FROM segments WHERE video_id = ?", (video_id,))

    def for_video(self, video_id: str) -> List[dict]:
        return [dict(r) for r in self._conn.execute(
            "SELECT * FROM segments WHERE video_id = ? ORDER BY segment_id",
            (video_id,))]

    def all(self) -> List[dict]:
        return [dict(r) for r in self._conn.execute(
            "SELECT * FROM segments ORDER BY video_id, segment_id")]

    def words_for(self, segment_id: str) -> List[dict]:
        return [dict(r) for r in self._conn.execute(
            "SELECT * FROM words WHERE segment_id = ? ORDER BY word_index",
            (segment_id,))]

    def set_embedding(self, segment_id: str, vector: np.ndarray):
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO segment_embeddings (segment_id, embedding)"
                " VALUES (?, ?)",
                (segment_id, np.asarray(vector, dtype=np.float32).tobytes()),
            )

    def video_stats(self, video_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM video_stats WHERE video_id = ?", (video_id,)
        ).fetchone()
        return dict(row) if row else None


class SpeakersRepo(_Repo):
    _EDITABLE = {
        "speaker_name", "gender", "gender_confidence", "age_estimate",
        "age_std", "age_group", "accent_region", "identity_match",
    }

    def ensure_exists(self, speaker_id: str):
        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO speakers (speaker_id) VALUES (?)",
                (speaker_id,),
            )

    def upsert(self, speaker_id: str, fields: Dict):
        if not speaker_id:
            raise ValueError("speaker_id is required")
        self.ensure_exists(speaker_id)
        safe = {k: v for k, v in fields.items() if k in self._EDITABLE}
        if not safe:
            return
        assignments = ", ".join(f"{k} = ?" for k in safe)
        with self._conn:
            self._conn.execute(
                f"UPDATE speakers SET {assignments} WHERE speaker_id = ?",
                (*safe.values(), speaker_id),
            )

    def get(self, speaker_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM speakers WHERE speaker_id = ?", (speaker_id,)
        ).fetchone()
        return dict(row) if row else None

    def all_with_stats(self) -> List[dict]:
        """Speakers joined with the live aggregates view (CSV export shape)."""
        return [dict(r) for r in self._conn.execute(
            "SELECT sp.*, st.num_videos, st.num_segments,"
            " st.total_duration_s, st.avg_asd, st.avg_wer"
            " FROM speakers sp LEFT JOIN speaker_stats st USING (speaker_id)"
            " ORDER BY sp.speaker_id")]

    def set_centroid(self, speaker_id: str, vector: np.ndarray):
        self.ensure_exists(speaker_id)
        with self._conn:
            self._conn.execute(
                "UPDATE speakers SET centroid = ? WHERE speaker_id = ?",
                (np.asarray(vector, dtype=np.float32).tobytes(), speaker_id),
            )

    def centroids(self) -> Dict[str, np.ndarray]:
        """All stored centroids: speaker_id → normalized 512-d vector."""
        result = {}
        for row in self._conn.execute(
                "SELECT speaker_id, centroid FROM speakers"
                " WHERE centroid IS NOT NULL"):
            result[row["speaker_id"]] = np.frombuffer(
                row["centroid"], dtype=np.float32)
        return result


class DroppedClipsRepo(_Repo):
    def record(self, video_id: str, clip_id: str, reason: str,
               start_time: Optional[float] = None,
               end_time: Optional[float] = None,
               face_visibility: Optional[float] = None,
               whisper_conf: Optional[float] = None,
               asd_score: Optional[float] = None,
               syncnet_conf: Optional[float] = None):
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO dropped_clips (video_id, clip_id, reason,"
                " start_time, end_time, face_visibility, whisper_conf,"
                " asd_score, syncnet_conf, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (video_id, clip_id, reason, start_time, end_time,
                 face_visibility, whisper_conf, asd_score, syncnet_conf, _now()),
            )

    def for_video(self, video_id: str) -> List[dict]:
        return [dict(r) for r in self._conn.execute(
            "SELECT * FROM dropped_clips WHERE video_id = ? ORDER BY clip_id",
            (video_id,))]


class JobsRepo(_Repo):
    """The API↔worker queue. No consumer yet — foundation for Faza 7."""

    def create(self, job_type: str, params: Dict,
               git_sha: str = "", config_hash: str = "") -> int:
        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO jobs (type, params_json, status, created_at,"
                " git_sha, config_hash) VALUES (?, ?, 'pending', ?, ?, ?)",
                (job_type, json.dumps(params), _now(), git_sha, config_hash),
            )
            return cursor.lastrowid

    def claim_next(self, worker_name: str) -> Optional[dict]:
        """Atomically claim the oldest pending job (correct with N workers)."""
        with self._conn:
            row = self._conn.execute(
                "SELECT id FROM jobs WHERE status = 'pending'"
                " ORDER BY id LIMIT 1").fetchone()
            if row is None:
                return None
            updated = self._conn.execute(
                "UPDATE jobs SET status = 'running', claimed_by = ?,"
                " started_at = ?, heartbeat_at = ?"
                " WHERE id = ? AND status = 'pending'",
                (worker_name, _now(), _now(), row["id"]),
            ).rowcount
            if not updated:      # someone else won the race
                return None
        return self.get(row["id"])

    def heartbeat(self, job_id: int, progress: Optional[Dict] = None):
        with self._conn:
            if progress is not None:
                self._conn.execute(
                    "UPDATE jobs SET heartbeat_at = ?, progress_json = ?"
                    " WHERE id = ?",
                    (_now(), json.dumps(progress), job_id),
                )
            else:
                self._conn.execute(
                    "UPDATE jobs SET heartbeat_at = ? WHERE id = ?",
                    (_now(), job_id),
                )

    def finish(self, job_id: int, status: str, error: str = ""):
        with self._conn:
            self._conn.execute(
                "UPDATE jobs SET status = ?, error = ?, finished_at = ?"
                " WHERE id = ?",
                (status, error, _now(), job_id),
            )

    def request_cancel(self, job_id: int) -> bool:
        with self._conn:
            return self._conn.execute(
                "UPDATE jobs SET status = 'cancel_requested'"
                " WHERE id = ? AND status IN ('pending', 'running')",
                (job_id,),
            ).rowcount > 0

    def cancel_unclaimed(self) -> int:
        """Cancel-requested jobs nobody ever claimed → cancelled outright.
        (The claim query only takes 'pending', so without this sweep an
        early cancel would hang forever.)"""
        with self._conn:
            return self._conn.execute(
                "UPDATE jobs SET status = 'cancelled', finished_at = ?"
                " WHERE status = 'cancel_requested' AND started_at = ''",
                (_now(),),
            ).rowcount

    def is_cancel_requested(self, job_id: int) -> bool:
        row = self._conn.execute(
            "SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return bool(row) and row["status"] == "cancel_requested"

    def stamp_execution(self, job_id: int, git_sha: str,
                        config_hash: str, log_path: str):
        """Record exactly what code/config this job runs with + its log."""
        with self._conn:
            self._conn.execute(
                "UPDATE jobs SET git_sha = ?, config_hash = ?, log_path = ?"
                " WHERE id = ?",
                (git_sha, config_hash, log_path, job_id),
            )

    def recover_stale(self, heartbeat_older_than_seconds: int = 60) -> List[int]:
        """Mark running jobs with a stale heartbeat as 'interrupted'.

        A dead worker leaves its job 'running' forever otherwise. Called at
        worker startup (and safe to call any time — claims are atomic)."""
        cutoff = datetime.now().timestamp() - heartbeat_older_than_seconds
        stale = []
        for row in self._conn.execute(
                "SELECT id, heartbeat_at FROM jobs"
                " WHERE status IN ('running', 'cancel_requested')"):
            beat = row["heartbeat_at"]
            try:
                beat_ts = datetime.strptime(beat, "%Y-%m-%d %H:%M:%S").timestamp()
            except (TypeError, ValueError):
                beat_ts = 0.0
            if beat_ts < cutoff:
                stale.append(row["id"])
        with self._conn:
            for job_id in stale:
                self._conn.execute(
                    "UPDATE jobs SET status = 'interrupted',"
                    " error = 'worker heartbeat lost' WHERE id = ?",
                    (job_id,),
                )
        return stale

    def all(self, limit: int = 50) -> List[dict]:
        return [dict(r) for r in self._conn.execute(
            "SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,))]

    def get(self, job_id: int) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None
