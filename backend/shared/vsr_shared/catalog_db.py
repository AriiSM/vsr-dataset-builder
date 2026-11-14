"""
Catalog database — the single source of truth for ALL pipeline metadata.

Storage v4: MongoDB document model (service `vsr-mongo`, database
`vsr-dataset-md`); media files still live on disk under data/processed/.

Collections (native documents — see catalog_model.py for the field map):
    videos              one doc per source video: source{}, metadata{}, pipeline{},
                        stats{}, processed_at/created_at (dates) and the rejected
                        clips embedded as dropped_clips[] {clip_id, reason{code,value},
                        time{}, scores{}}
    segments            one doc per exported segment — THE dataset: time{},
                        quality{tier, asd{}, syncnet{}, whisper{}}, face{bbox{x,y,w,h},
                        head_pose{yaw,pitch,roll}}, mouth{}, transcripts{...},
                        wer{...}, review{...} and the words embedded as words[]
    speakers            identity: gender{}, age{}, centroid (512 floats)
    segment_embeddings  per-segment ArcFace vector (512 floats), _id = segment_id
    jobs                the API↔worker queue: params{}, progress{}, execution{}, dates
    counters · meta     jobs.id sequence · schema + data version
Views:
    speaker_stats · video_stats · dataset_overview (live aggregates).

Simulated relations: _id = natural key (video_id, segment_id, speaker_id, job id);
references keep their names (segments.video_id, segments.speaker_id) and are
indexed. One-to-many relations that are always read with their parent
(words, dropped clips) are embedded arrays instead of collections.

Design rules:
    - This module is the ONLY place that knows the database (queries live here).
    - Repositories still speak flat ROWS (historical column names) to the rest of
      the code; catalog_model converts rows ⇄ documents exactly. `.document()`
      methods return the native nested documents (API detail views).
    - Connection: MONGO_URI / MONGO_DB from the environment or the repo .env.
      MONGO_HOST (host:port) overrides the host inside containers.
    - A db_path under the system temp dir maps to an isolated throw-away
      database (tests); every other path maps to MONGO_DB.
"""

import gzip
import hashlib
import json
import os
import re
import tempfile
import threading
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from vsr_shared import catalog_model as M

SCHEMA_VERSION = 5
DEFAULT_DB = "vsr-dataset-md"
DEFAULT_URI = "mongodb://127.0.0.1:27018/"
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# The SQLite DDL is kept as the schema contract: column names, order and
# defaults of every collection are parsed from it.
_SCHEMA = """
CREATE TABLE videos (
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

CREATE TABLE segments (
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
    av_speaker_mismatch      INTEGER,
    boundary_start_type      TEXT DEFAULT '',
    boundary_end_type        TEXT DEFAULT '',
    text_largev3             TEXT DEFAULT '',
    wer_medium_vs_large      REAL,
    needs_review             INTEGER,
    wer                      REAL DEFAULT 0.0,
    wer_word_count_ref       INTEGER,
    review_status            TEXT DEFAULT '',
    reviewed_at              TEXT DEFAULT '',
    transcript_edited        INTEGER DEFAULT 0,
    trimmed                  INTEGER DEFAULT 0,
    split                    TEXT DEFAULT '',
    created_at               TEXT DEFAULT '',
    whisper_standard_mediu           TEXT,
    whisper_standard_mediu_conf      REAL,
    whisper_standard_mediu_conf_min  REAL,
    whisper_standard_mediu_conf_p25  REAL,
    whisper_dialectat_mediu          TEXT,
    whisper_dialectat_mediu_conf     REAL,
    whisper_dialectat_mediu_conf_min REAL,
    whisper_dialectat_mediu_conf_p25 REAL,
    whisper_dialectat_large          TEXT,
    whisper_dialectat_large_conf     REAL,
    whisper_dialectat_large_conf_min REAL,
    whisper_dialectat_large_conf_p25 REAL,
    wer_standard_vs_dialectat_mediu  REAL,
    wer_mediu_vs_large               REAL,
    wer_standard_vs_dialectat_large  REAL,
    transcript_consensus             TEXT,
    wer_dialectat_mediu_vs_large     REAL,
    transcript_source                TEXT
);

CREATE TABLE words (
    segment_id TEXT NOT NULL REFERENCES segments(segment_id) ON DELETE CASCADE,
    word_index INTEGER NOT NULL,
    word       TEXT NOT NULL,
    start_time REAL,
    end_time   REAL,
    confidence REAL,
    asd_score  REAL,
    PRIMARY KEY (segment_id, word_index)
);

CREATE TABLE speakers (
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

CREATE TABLE segment_embeddings (
    segment_id TEXT PRIMARY KEY REFERENCES segments(segment_id) ON DELETE CASCADE,
    embedding  BLOB NOT NULL
);

CREATE TABLE dropped_clips (
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

CREATE TABLE jobs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    type         TEXT NOT NULL,
    params_json  TEXT DEFAULT '{}',
    status       TEXT NOT NULL DEFAULT 'pending',
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
"""

# Columns added in the SQLite era by tools (transcript variants) that are not
# always present on older rows; they default to NULL like an ALTER TABLE did.
_NULLABLE_EXTRA = re.compile(r"^(whisper_|wer_|transcript_)")


def _parse_schema(ddl: str) -> Dict[str, "OrderedDict[str, object]"]:
    tables: Dict[str, OrderedDict] = {}
    for name, body in re.findall(r"CREATE TABLE (\w+) \((.*?)\n\);", ddl, re.S):
        cols: OrderedDict = OrderedDict()
        for line in body.splitlines():
            line = line.strip().rstrip(",")
            m = re.match(r"(\w+)\s+(TEXT|REAL|INTEGER|BLOB)\b(.*)", line)
            if not m or m.group(1) == "PRIMARY":
                continue
            default = None
            d = re.search(r"DEFAULT\s+('(?:[^']*)'|-?[\d.]+)", m.group(3))
            if d:
                raw = d.group(1)
                if raw.startswith("'"):
                    default = raw[1:-1]
                elif "." in raw:
                    default = float(raw)
                else:
                    default = int(raw)
            cols[m.group(1)] = default
        tables[name] = cols
    return tables


COLUMNS = _parse_schema(_SCHEMA)

# ------------------------------------------------------------- connection

def _load_dotenv():
    """MONGO_* from the repo .env when not already in the environment."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("MONGO_") and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def mongo_uri() -> str:
    _load_dotenv()
    uri = os.environ.get("MONGO_URI", DEFAULT_URI)
    host = os.environ.get("MONGO_HOST", "").strip()
    if host:  # containers: same credentials, different host (vsr-mongo:27017)
        parts = urlsplit(uri)
        creds = parts.netloc.rsplit("@", 1)[0] + "@" if "@" in parts.netloc else ""
        uri = urlunsplit(parts._replace(netloc=creds + host))
    return uri


def resolve_db_name(db_path=None, db_name: Optional[str] = None) -> str:
    if db_name:
        return db_name
    if db_path is not None:
        path = Path(db_path).resolve()
        tmp = Path(tempfile.gettempdir()).resolve()
        if tmp == path or tmp in path.parents:
            digest = hashlib.sha1(str(path).lower().encode()).hexdigest()[:16]
            return f"vsr_test_{digest}"
    _load_dotenv()
    return os.environ.get("MONGO_DB", DEFAULT_DB)


_CLIENTS: Dict[str, object] = {}
_CLIENTS_LOCK = threading.Lock()


def _client(uri: str):
    from pymongo import MongoClient
    with _CLIENTS_LOCK:
        client = _CLIENTS.get(uri)
        if client is None:
            client = MongoClient(uri, serverSelectionTimeoutMS=5000, tz_aware=False)
            _CLIENTS[uri] = client
        return client



# ------------------------------------------------------------- document model

# Collections and their indexes. Multikey indexes cover the embedded arrays
# (segments.words.word for vocabulary search, videos.dropped_clips.*).
INDEXES = {
    "videos": [([("status", 1)], {}), ([("region", 1)], {}),
               ([("dropped_clips.reason.code", 1)], {})],
    "segments": [
        ([("video_id", 1), ("segment_id", 1)], {}),   # list order of the UI
        ([("speaker_id", 1)], {}),
        ([("quality.tier", 1)], {}),
        ([("review.status", 1)], {"sparse": True}),
        ([("words.word", 1)], {}),
    ],
    "speakers": [],
    "segment_embeddings": [],
    "jobs": [([("status", 1), ("id", 1)], {})],
}
COLLECTIONS = list(INDEXES) + ["counters", "meta"]

# simulated foreign keys: (child collection, field, parent collection)
FKS = [
    ("segments", "video_id", "videos"),
    ("segments", "speaker_id", "speakers"),
    ("segment_embeddings", "_id", "segments"),
]

_DUR = "$time.duration"
_ASD = "$quality.asd.score"
_TIER = "$quality.tier"


def _tier_count(tier: str) -> dict:
    return {"$sum": {"$cond": [{"$eq": [_TIER, tier]}, 1, 0]}}


def _speaker_stats_pipeline(match: Optional[dict] = None) -> list:
    return [
        {"$match": {"speaker_id": {"$ne": ""}, **(match or {})}},
        {"$group": {"_id": "$speaker_id", "videos": {"$addToSet": "$video_id"},
                    "num_segments": {"$sum": 1}, "total_duration_s": {"$sum": _DUR},
                    "avg_asd": {"$avg": _ASD},
                    "avg_wer": {"$avg": {"$ifNull": ["$wer.value", 0.0]}}}},
        {"$project": {"_id": 0, "speaker_id": "$_id", "num_videos": {"$size": "$videos"},
                      "num_segments": 1,
                      "total_duration_s": {"$round": ["$total_duration_s", 2]},
                      "avg_asd": {"$round": ["$avg_asd", 3]},
                      "avg_wer": {"$round": ["$avg_wer", 4]}}},
    ]


def _video_stats_pipeline(match: Optional[dict] = None) -> list:
    return ([{"$match": match}] if match else []) + [
        {"$group": {"_id": "$video_id", "num_segments": {"$sum": 1},
                    "total_duration_s": {"$sum": _DUR}, "avg_asd": {"$avg": _ASD},
                    "avg_syncnet": {"$avg": "$quality.syncnet.conf"},
                    "tier_a": _tier_count("A"), "tier_b": _tier_count("B"),
                    "tier_c": _tier_count("C")}},
        {"$project": {"_id": 0, "video_id": "$_id", "num_segments": 1,
                      "total_duration_s": {"$round": ["$total_duration_s", 2]},
                      "avg_asd": {"$round": ["$avg_asd", 3]},
                      "avg_syncnet": {"$round": ["$avg_syncnet", 3]},
                      "tier_a": 1, "tier_b": 1, "tier_c": 1}},
    ]


_OVERVIEW_PIPELINE = [
    {"$group": {"_id": None, "num_segments": {"$sum": 1},
                "videos": {"$addToSet": "$video_id"},
                "speakers": {"$addToSet": "$speaker_id"},
                "dur": {"$sum": _DUR},
                "tier_a": _tier_count("A"), "tier_b": _tier_count("B"),
                "tier_c": _tier_count("C")}},
    {"$project": {"_id": 0, "num_segments": 1, "num_videos": {"$size": "$videos"},
                  "num_speakers": {"$size": "$speakers"},
                  "total_hours": {"$round": [{"$divide": ["$dur", 3600.0]}, 2]},
                  "tier_a": 1, "tier_b": 1, "tier_c": 1}},
]

VIEWS = {
    "speaker_stats": ("segments", _speaker_stats_pipeline()),
    "video_stats": ("segments", _video_stats_pipeline()),
    "dataset_overview": ("segments", _OVERVIEW_PIPELINE),
}

_NOT_REJECTED = {"review.status": {"$ne": "rejected"}}


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _row(table: str, doc: Optional[dict]) -> Optional[dict]:
    """Document → flat row in the historical column order."""
    if doc is None:
        return None
    r = M.to_row(table, doc, COLUMNS[table])
    return {c: r.get(c, COLUMNS[table][c]) for c in COLUMNS[table]}


def _doc(table: str, fields: Dict) -> dict:
    """Flat fields (missing columns = defaults) → document."""
    full = dict(COLUMNS[table])
    full.update({k: v for k, v in fields.items() if k in COLUMNS[table]})
    return M.to_doc(table, full, COLUMNS[table])


def _projection(table: str, fields: Optional[List[str]]) -> Optional[dict]:
    if not fields:
        return None
    return {M.path(table, f): 1 for f in fields}


def _pick(row: dict, fields: Optional[List[str]]) -> dict:
    return {f: row.get(f) for f in fields} if fields else row


class CatalogDatabase:
    """Owns the MongoDB handle and the schema. Cheap to create: the client
    (connection pool, thread-safe) is shared per URI inside the process.

    `db_path` is kept for call-site compatibility with the SQLite era; it only
    selects an isolated database when it points into the temp dir (tests)."""

    def __init__(self, db_path=None, *, check_same_thread: bool = True,
                 uri: Optional[str] = None, db_name: Optional[str] = None):
        self.db_path = Path(db_path) if db_path is not None else None
        if self.db_path is not None:
            # the catalog dir still hosts sidecars (backups, review_status.json)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.uri = uri or mongo_uri()
        self.db_name = resolve_db_name(db_path, db_name)
        self._db = _client(self.uri)[self.db_name]
        self._init_schema()

        self.videos = VideosRepo(self._db)
        self.segments = SegmentsRepo(self._db)
        self.speakers = SpeakersRepo(self._db)
        self.dropped = DroppedClipsRepo(self._db)
        self.jobs = JobsRepo(self._db)

    # -- schema ------------------------------------------------------------
    def _init_schema(self):
        meta = self._db["meta"].find_one({"_id": "schema"})
        if meta and meta.get("version", 0) >= SCHEMA_VERSION:
            return
        from pymongo.errors import CollectionInvalid, OperationFailure

        existing = set(self._db.list_collection_names())
        for name in COLLECTIONS:
            if name not in existing:
                try:
                    self._db.create_collection(name)
                except CollectionInvalid:
                    pass  # created concurrently
        for name, specs in INDEXES.items():
            for keys, opts in specs:
                self._db[name].create_index(keys, **opts)
        for name, (source, pipeline) in VIEWS.items():
            try:
                if name in existing:
                    self._db.command({"collMod": name, "viewOn": source,
                                      "pipeline": pipeline})
                else:
                    self._db.command({"create": name, "viewOn": source,
                                      "pipeline": pipeline})
            except OperationFailure:
                pass  # created concurrently
        if self._db["counters"].find_one({"_id": "jobs"}) is None:
            last = self._db["jobs"].find_one(sort=[("id", -1)])
            self._db["counters"].update_one(
                {"_id": "jobs"}, {"$setOnInsert": {"seq": int(last["id"]) if last else 0}},
                upsert=True)
        self._db["meta"].update_one(
            {"_id": "schema"}, {"$max": {"version": SCHEMA_VERSION},
                                "$set": {"model": "document"}}, upsert=True)

    def schema_version(self) -> int:
        meta = self._db["meta"].find_one({"_id": "schema"})
        return int(meta.get("version", 0)) if meta else 0

    def data_version(self) -> int:
        """Increments on every catalog write — cheap staleness check for caches."""
        doc = self._db["meta"].find_one({"_id": "data_version"})
        return int(doc.get("v", 0)) if doc else 0

    # -- info --------------------------------------------------------------
    @property
    def mongo(self):
        """The pymongo Database (for tools that need ad-hoc queries)."""
        return self._db

    @property
    def location(self) -> str:
        host = urlsplit(self.uri).netloc.rsplit("@", 1)[-1]
        return f"mongodb://{host}/{self.db_name}"

    def ping(self) -> bool:
        return bool(self._db.client.admin.command("ping").get("ok"))

    def overview(self) -> dict:
        row = next(self._db["segments"].aggregate(_OVERVIEW_PIPELINE), None)
        return row or {"num_segments": 0, "num_videos": 0, "num_speakers": 0,
                       "total_hours": 0, "tier_a": 0, "tier_b": 0, "tier_c": 0}

    # -- backup ------------------------------------------------------------
    def backup_to(self, target_path: Path) -> None:
        """Full logical backup: every collection as gzip-compressed BSON
        (one stream of {"c": collection, "d": document})."""
        import bson
        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = target_path.with_suffix(target_path.suffix + ".part")
        with gzip.open(tmp, "wb", compresslevel=1) as out:
            for name in COLLECTIONS:
                for doc in self._db[name].find():
                    out.write(bson.encode({"c": name, "d": doc}))
        tmp.replace(target_path)

    @classmethod
    def restore_backup(cls, backup_path: Path, *, db_name: str,
                       uri: Optional[str] = None) -> "CatalogDatabase":
        """Load a backup_to() file into the (empty) database `db_name`."""
        import bson
        db = _client(uri or mongo_uri())[db_name]
        batches: Dict[str, list] = {}
        with gzip.open(backup_path, "rb") as f:
            for item in bson.decode_file_iter(f):
                batch = batches.setdefault(item["c"], [])
                batch.append(item["d"])
                if len(batch) >= 5000:
                    db[item["c"]].insert_many(batch)
                    batch.clear()
        for name, batch in batches.items():
            if batch:
                db[name].insert_many(batch)
        return cls(uri=uri, db_name=db_name)

    def drop(self):
        """Delete this whole database (tests / throw-away DBs only)."""
        if self.db_name == resolve_db_name():
            raise RuntimeError(f"refusing to drop the main catalog '{self.db_name}'")
        self._db.client.drop_database(self.db_name)

    def close(self):
        """No-op: the client is shared per process (kept for compatibility)."""


class _Repo:
    def __init__(self, db):
        self._db = db

    def _touch(self):
        """Bump the catalog data version (API caches compare against it)."""
        self._db["meta"].update_one({"_id": "data_version"}, {"$inc": {"v": 1}},
                                    upsert=True)


class VideosRepo(_Repo):
    def ensure_exists(self, video_id: str):
        doc = _doc("videos", {"video_id": video_id, "created_at": _now()})
        self._db["videos"].update_one({"_id": video_id}, {"$setOnInsert": doc},
                                      upsert=True)

    def set_status(self, video_id: str, status: str):
        self.update_fields(video_id, {"status": status})

    def update_fields(self, video_id: str, fields: Dict):
        """Update a subset of columns (keys validated against the schema)."""
        if not fields:
            return
        self.ensure_exists(video_id)
        safe = {k: v for k, v in fields.items() if k in COLUMNS["videos"] and k != "video_id"}
        ops = M.set_ops("videos", safe, COLUMNS["videos"])
        if ops:
            self._db["videos"].update_one({"_id": video_id}, ops)
            self._touch()

    def document(self, video_id: str) -> Optional[dict]:
        """The native document (nested, with dropped_clips[])."""
        return self._db["videos"].find_one({"_id": video_id})

    def get(self, video_id: str) -> Optional[dict]:
        return _row("videos", self._db["videos"].find_one(
            {"_id": video_id}, {"dropped_clips": 0}))

    def all(self) -> List[dict]:
        return [_row("videos", d) for d in
                self._db["videos"].find({}, {"dropped_clips": 0}).sort("_id", 1)]

    def select_for_batch(self, status_filter=None, video_ids=None,
                         limit=None) -> List[dict]:
        """Batch selection.

        video_ids (explicit picks) bypass the status filter BUT always skip
        'completed' videos — prevents re-processing after a job restart."""
        rows = self.all()
        if video_ids:
            wanted = {str(v) for v in video_ids}
            rows = [r for r in rows if str(r["video_id"]) in wanted
                    and (r.get("status") or "") != "completed"]
        elif status_filter:
            allowed = set(status_filter)
            rows = [r for r in rows if (r.get("status") or "") in allowed]
        if limit:
            rows = rows[:int(limit)]
        return rows

    def region(self, video_id: str) -> str:
        doc = self._db["videos"].find_one({"_id": video_id}, {"region": 1})
        return (doc.get("region") or "").strip() if doc else ""

    def regions(self) -> Dict[str, str]:
        """video_id → region, '' / missing mapped to 'UNKNOWN'."""
        return {d["_id"]: (d.get("region") or "UNKNOWN")
                for d in self._db["videos"].find({}, {"region": 1})}


class SegmentsRepo(_Repo):
    def upsert(self, row: Dict, words: Optional[List[Dict]] = None):
        """Insert or replace ONE segment (+ its embedded words when given)."""
        fields = dict(row)
        fields.setdefault("created_at", _now())
        doc = _doc("segments", fields)
        sid = doc["segment_id"]
        if words is None:  # keep the words already embedded (REPLACE kept the words table)
            old = self._db["segments"].find_one({"_id": sid}, {"words": 1})
            if old and old.get("words"):
                doc["words"] = old["words"]
        else:
            doc["words"] = M.word_docs([
                {"word": w.get("word", ""), "start_time": w.get("start"),
                 "end_time": w.get("end"), "confidence": w.get("score"),
                 "asd_score": w.get("asd_score")} for w in words], COLUMNS["words"])
        self._db["segments"].replace_one({"_id": sid}, {"_id": sid, **doc}, upsert=True)
        self._touch()

    def replace_words(self, segment_id: str, words: List[Dict]):
        """Replace the embedded words (keys: word, start_time, end_time,
        confidence, asd_score; position = word_index)."""
        docs = M.word_docs(words, COLUMNS["words"])
        if docs:
            self._db["segments"].update_one({"_id": segment_id}, {"$set": {"words": docs}})
        else:
            self._db["segments"].update_one({"_id": segment_id}, {"$unset": {"words": ""}})
        self._touch()

    def replace_for_video(self, video_id: str, rows: Iterable[Dict]):
        """End-of-video authoritative rewrite (final speaker_id / tier / av)."""
        from pymongo import UpdateOne
        ops = []
        for row in rows:
            safe = {k: v for k, v in row.items() if k in COLUMNS["segments"]}
            safe.setdefault("created_at", _now())
            sid = safe["segment_id"]
            update = M.set_ops("segments", {k: v for k, v in safe.items()
                                            if k != "segment_id"}, COLUMNS["segments"])
            insert_doc = _doc("segments", safe)
            on_insert = {k: v for k, v in insert_doc.items()
                         if k not in {p.split(".")[0] for p in update.get("$set", {})}
                         and k not in {p.split(".")[0] for p in update.get("$unset", {})}}
            if on_insert:
                update["$setOnInsert"] = on_insert
            ops.append(UpdateOne({"_id": sid}, update, upsert=True))
        if ops:
            self._db["segments"].bulk_write(ops, ordered=True)
            self._touch()

    def delete_for_video(self, video_id: str):
        """Delete a video's segments (+ embedded words) and their embeddings."""
        ids = self._db["segments"].distinct("_id", {"video_id": video_id})
        if ids:
            self._db["segment_embeddings"].delete_many({"_id": {"$in": ids}})
        self._db["segments"].delete_many({"video_id": video_id})
        self._touch()

    def document(self, segment_id: str) -> Optional[dict]:
        """The native document (nested sub-objects + words[])."""
        return self._db["segments"].find_one({"_id": segment_id})

    def get(self, segment_id: str, fields: Optional[List[str]] = None) -> Optional[dict]:
        doc = self._db["segments"].find_one({"_id": segment_id},
                                            _projection("segments", fields) or {"words": 0})
        row = _row("segments", doc)
        return _pick(row, fields) if row else None

    def update_fields(self, segment_id: str, fields: Dict) -> bool:
        """Update a subset of columns; returns False when the segment is missing."""
        safe = {k: v for k, v in fields.items()
                if k in COLUMNS["segments"] and k != "segment_id"}
        ops = M.set_ops("segments", safe, COLUMNS["segments"])
        if not ops:
            return self._db["segments"].count_documents({"_id": segment_id}) > 0
        matched = self._db["segments"].update_one({"_id": segment_id}, ops).matched_count > 0
        self._touch()
        return matched

    def set_many(self, updates: Dict[str, Dict]):
        """{segment_id: {column: value}} in one bulk write."""
        from pymongo import UpdateOne
        ops = []
        for sid, f in updates.items():
            u = M.set_ops("segments", {k: v for k, v in f.items() if k != "segment_id"},
                          COLUMNS["segments"])
            if u:
                ops.append(UpdateOne({"_id": sid}, u))
        for i in range(0, len(ops), 5000):
            self._db["segments"].bulk_write(ops[i:i + 5000], ordered=False)
        if ops:
            self._touch()

    def query(self, *, fields: Optional[List[str]] = None, exclude_rejected: bool = False,
              video_id: Optional[str] = None, speaker_id: Optional[str] = None,
              with_region: bool = False) -> List[dict]:
        """Rows ordered by (video_id, segment_id); optional filters/columns.
        with_region adds the video's region ('UNKNOWN' when empty)."""
        flt: dict = {}
        if exclude_rejected:
            flt.update(_NOT_REJECTED)
        if video_id is not None:
            flt["video_id"] = video_id
        if speaker_id is not None:
            flt["speaker_id"] = speaker_id
        projection = _projection("segments", fields) or {"words": 0}
        if fields and with_region:
            projection["video_id"] = 1
        cursor = self._db["segments"].find(flt, projection).sort(
            [("video_id", 1), ("segment_id", 1)])
        regions = VideosRepo(self._db).regions() if with_region else None
        rows = []
        for d in cursor:
            row = _row("segments", d)
            out = _pick(row, fields)
            if regions is not None:
                out["region"] = regions.get(row["video_id"], "UNKNOWN") or "UNKNOWN"
            rows.append(out)
        return rows

    def for_video(self, video_id: str) -> List[dict]:
        return self.query(video_id=video_id)

    def all(self) -> List[dict]:
        return self.query()

    def review_flags(self) -> List[dict]:
        """Segments with a verdict or a manual edit (review_status sidecar)."""
        fields = ["segment_id", "review_status", "transcript_edited", "trimmed"]
        return [_pick(_row("segments", d), fields) for d in self._db["segments"].find(
            {"$or": [{"review.status": {"$nin": ["", None]}},
                     {"review.transcript_edited": {"$nin": [0, None]}},
                     {"review.trimmed": 1}]},
            _projection("segments", fields))]

    def rejected_ids(self) -> set:
        return set(self._db["segments"].distinct("_id", {"review.status": "rejected"}))

    def fresh_video_stats(self) -> Dict[str, dict]:
        """Per-video {n, dur, asd} over the surviving (non-rejected) segments."""
        out = {}
        for r in self._db["segments"].aggregate([
                {"$match": _NOT_REJECTED},
                {"$group": {"_id": "$video_id", "n": {"$sum": 1},
                            "dur": {"$sum": _DUR}, "asd": {"$avg": _ASD}}}]):
            out[r["_id"]] = {"video_id": r["_id"], "n": r["n"],
                             "dur": round(r["dur"] or 0, 2),
                             "asd": None if r["asd"] is None else round(r["asd"], 3)}
        return out

    def word_segment_counts(self, exclude_segment_ids: Optional[set] = None) -> List[dict]:
        """[{word, segment_id, occ, dur}] grouped by (word, segment);
        dur = Σ max(end - start, 0) — straight from the embedded words[]."""
        match = {"_id": {"$nin": list(exclude_segment_ids)}} if exclude_segment_ids else {}
        pipeline = ([{"$match": match}] if match else []) + [
            {"$project": {"words.word": 1, "words.start": 1, "words.end": 1}},
            {"$unwind": "$words"},
            {"$group": {"_id": {"w": "$words.word", "s": "$_id"}, "occ": {"$sum": 1},
                        "dur": {"$sum": {"$max": [
                            {"$subtract": [{"$ifNull": ["$words.end", 0]},
                                           {"$ifNull": ["$words.start", 0]}]}, 0]}}}},
            {"$project": {"_id": 0, "word": "$_id.w", "segment_id": "$_id.s",
                          "occ": 1, "dur": 1}},
        ]
        return list(self._db["segments"].aggregate(pipeline, allowDiskUse=True))

    def without_words(self, exclude_rejected: bool = True) -> List[dict]:
        """Segments that have no words (segment_id, text, duration)."""
        flt = {"words.0": {"$exists": False}}
        if exclude_rejected:
            flt.update(_NOT_REJECTED)
        fields = ["segment_id", "text", "duration"]
        return [_pick(_row("segments", d), fields) for d in
                self._db["segments"].find(flt, _projection("segments", fields))]

    def words_for(self, segment_id: str) -> List[dict]:
        doc = self._db["segments"].find_one({"_id": segment_id}, {"words": 1})
        return M.word_rows(segment_id, (doc or {}).get("words"), COLUMNS["words"])

    def segment_ids_with_words(self) -> set:
        return set(self._db["segments"].distinct("_id", {"words.0": {"$exists": True}}))

    def set_embedding(self, segment_id: str, vector: np.ndarray):
        self._db["segment_embeddings"].replace_one(
            {"_id": segment_id},
            {"_id": segment_id, **_doc("segment_embeddings",
                                       {"segment_id": segment_id, "embedding": vector})},
            upsert=True)

    def embeddings(self) -> Dict[str, np.ndarray]:
        """segment_id → 512-d float32 vector (all stored embeddings)."""
        return {d["_id"]: np.asarray(d["embedding"], dtype=np.float32)
                for d in self._db["segment_embeddings"].find()}

    def video_stats(self, video_id: str) -> Optional[dict]:
        return next(self._db["segments"].aggregate(
            _video_stats_pipeline({"video_id": video_id})), None)


class SpeakersRepo(_Repo):
    _EDITABLE = {
        "speaker_name", "gender", "gender_confidence", "age_estimate",
        "age_std", "age_group", "accent_region", "identity_match",
    }

    def ensure_exists(self, speaker_id: str):
        doc = _doc("speakers", {"speaker_id": speaker_id})
        self._db["speakers"].update_one({"_id": speaker_id}, {"$setOnInsert": doc},
                                        upsert=True)

    def upsert(self, speaker_id: str, fields: Dict):
        if not speaker_id:
            raise ValueError("speaker_id is required")
        self.ensure_exists(speaker_id)
        safe = {k: v for k, v in fields.items() if k in self._EDITABLE}
        ops = M.set_ops("speakers", safe, COLUMNS["speakers"])
        if ops:
            self._db["speakers"].update_one({"_id": speaker_id}, ops)
            self._touch()

    def document(self, speaker_id: str) -> Optional[dict]:
        return self._db["speakers"].find_one({"_id": speaker_id}, {"centroid": 0})

    def get(self, speaker_id: str) -> Optional[dict]:
        return _row("speakers", self._db["speakers"].find_one({"_id": speaker_id}))

    def all(self) -> List[dict]:
        return [_row("speakers", d) for d in self._db["speakers"].find().sort("_id", 1)]

    def all_with_stats(self) -> List[dict]:
        """Speakers joined with the live aggregates (CSV export shape)."""
        stats = {r["speaker_id"]: r for r in
                 self._db["segments"].aggregate(_speaker_stats_pipeline())}
        rows = []
        for sp in self.all():
            st = stats.get(sp["speaker_id"], {})
            for k in ("num_videos", "num_segments", "total_duration_s", "avg_asd", "avg_wer"):
                sp[k] = st.get(k)
            rows.append(sp)
        return rows

    def set_centroid(self, speaker_id: str, vector: np.ndarray):
        self.ensure_exists(speaker_id)
        self._db["speakers"].update_one(
            {"_id": speaker_id},
            {"$set": {"centroid": M.encode_value("speakers", "centroid", vector)}})

    def centroids(self) -> Dict[str, np.ndarray]:
        """All stored centroids: speaker_id → normalized 512-d vector."""
        return {d["_id"]: np.asarray(d["centroid"], dtype=np.float32)
                for d in self._db["speakers"].find(
                    {"centroid": {"$type": "array"}}, {"centroid": 1})}


class DroppedClipsRepo(_Repo):
    """Rejected clips, embedded in their video: videos.dropped_clips[]."""

    def record(self, video_id: str, clip_id: str, reason: str,
               start_time: Optional[float] = None,
               end_time: Optional[float] = None,
               face_visibility: Optional[float] = None,
               whisper_conf: Optional[float] = None,
               asd_score: Optional[float] = None,
               syncnet_conf: Optional[float] = None):
        VideosRepo(self._db).ensure_exists(video_id)
        entry = _doc("dropped_clips", {
            "clip_id": clip_id, "reason": reason,
            "start_time": start_time, "end_time": end_time,
            "face_visibility": face_visibility, "whisper_conf": whisper_conf,
            "asd_score": asd_score, "syncnet_conf": syncnet_conf,
            "created_at": _now()})
        videos = self._db["videos"]
        videos.update_one({"_id": video_id}, {"$pull": {"dropped_clips": {"clip_id": clip_id}}})
        videos.update_one({"_id": video_id}, {"$push": {"dropped_clips": {
            "$each": [entry], "$sort": {"clip_id": 1}}}})
        self._touch()

    def for_video(self, video_id: str) -> List[dict]:
        doc = self._db["videos"].find_one({"_id": video_id}, {"dropped_clips": 1}) or {}
        rows = []
        for d in doc.get("dropped_clips", []):
            r = _row("dropped_clips", d)
            r["video_id"] = video_id
            rows.append({c: r.get(c) for c in COLUMNS["dropped_clips"]})
        return sorted(rows, key=lambda r: r["clip_id"])


class JobsRepo(_Repo):
    """The API↔worker queue (atomic claims via find_one_and_update)."""

    def _update(self, job_id: int, fields: Dict, extra_filter: Optional[dict] = None):
        return self._db["jobs"].update_one({"_id": int(job_id), **(extra_filter or {})},
                                           M.set_ops("jobs", fields, COLUMNS["jobs"]))

    def create(self, job_type: str, params: Dict,
               git_sha: str = "", config_hash: str = "") -> int:
        from pymongo import ReturnDocument
        seq = self._db["counters"].find_one_and_update(
            {"_id": "jobs"}, {"$inc": {"seq": 1}}, upsert=True,
            return_document=ReturnDocument.AFTER)["seq"]
        doc = _doc("jobs", {
            "id": int(seq), "type": job_type, "params_json": json.dumps(params),
            "status": "pending", "created_at": _now(),
            "git_sha": git_sha, "config_hash": config_hash})
        self._db["jobs"].insert_one({"_id": int(seq), **doc})
        return int(seq)

    def claim_next(self, worker_name: str) -> Optional[dict]:
        """Atomically claim the oldest pending job (correct with N workers)."""
        from pymongo import ReturnDocument
        now = M.encode_value("jobs", "started_at", _now())
        return _row("jobs", self._db["jobs"].find_one_and_update(
            {"status": "pending"},
            {"$set": {"status": "running", "worker": worker_name,
                      "started_at": now, "heartbeat_at": now}},
            sort=[("id", 1)], return_document=ReturnDocument.AFTER))

    def heartbeat(self, job_id: int, progress: Optional[Dict] = None):
        fields = {"heartbeat_at": _now()}
        if progress is not None:
            fields["progress_json"] = json.dumps(progress)
        self._update(job_id, fields)

    def finish(self, job_id: int, status: str, error: str = ""):
        self._update(job_id, {"status": status, "error": error, "finished_at": _now()})

    def request_cancel(self, job_id: int) -> bool:
        return self._update(job_id, {"status": "cancel_requested"},
                            {"status": {"$in": ["pending", "running"]}}).matched_count > 0

    def cancel_unclaimed(self) -> int:
        """Cancel-requested jobs nobody ever claimed → cancelled outright."""
        return self._db["jobs"].update_many(
            {"status": "cancel_requested", "started_at": {"$exists": False}},
            {"$set": {"status": "cancelled",
                      "finished_at": M.encode_value("jobs", "finished_at", _now())}}
        ).modified_count

    def is_cancel_requested(self, job_id: int) -> bool:
        doc = self._db["jobs"].find_one({"_id": int(job_id)}, {"status": 1})
        return bool(doc) and doc["status"] == "cancel_requested"

    def stamp_execution(self, job_id: int, git_sha: str,
                        config_hash: str, log_path: str):
        """Record exactly what code/config this job runs with + its log."""
        self._update(job_id, {"git_sha": git_sha, "config_hash": config_hash,
                              "log_path": log_path})

    def recover_stale(self, heartbeat_older_than_seconds: int = 60) -> List[int]:
        """Mark running jobs with a stale heartbeat as 'interrupted'."""
        cutoff = datetime.now().timestamp() - heartbeat_older_than_seconds
        stale = []
        for doc in self._db["jobs"].find(
                {"status": {"$in": ["running", "cancel_requested"]}},
                {"id": 1, "heartbeat_at": 1}):
            beat = doc.get("heartbeat_at")
            beat_ts = beat.timestamp() if isinstance(beat, datetime) else 0.0
            if beat_ts < cutoff:
                stale.append(int(doc["id"]))
        for job_id in stale:
            self._update(job_id, {"status": "interrupted", "error": "worker heartbeat lost"})
        return stale

    def document(self, job_id: int) -> Optional[dict]:
        return self._db["jobs"].find_one({"_id": int(job_id)})

    def all(self, limit: int = 50) -> List[dict]:
        return [_row("jobs", d) for d in
                self._db["jobs"].find().sort("id", -1).limit(int(limit))]

    def get(self, job_id: int) -> Optional[dict]:
        return _row("jobs", self._db["jobs"].find_one({"_id": int(job_id)}))
