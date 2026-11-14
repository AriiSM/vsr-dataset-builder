"""
Document model of the catalog (database `vsr-dataset-md`).

The pipeline thinks in flat ROWS (the historical SQLite columns: face_bbox,
whisper_dialectat_mediu_conf, review_status, ...). MongoDB stores them as
native DOCUMENTS instead:

    - related columns grouped into sub-documents
        quality.asd.score, quality.syncnet.conf, face.head_pose.yaw,
        transcripts.dialectat_large.conf, review.status, source.url, ...
    - JSON-in-a-string columns become real structures
        face_bbox "[x, y, w, h]"        → face.bbox {x, y, w, h}
        head_pose_avg "[y, p, r]"        → face.head_pose {yaw, pitch, roll}
        params_json / progress_json      → params / progress objects
        dropped reason "face_visibility=0.20" → reason {code, value}
    - one-to-many as arrays of sub-documents
        segment.words[]                  (was the `words` table)
        video.dropped_clips[]            (was the `dropped_clips` table)
    - timestamps as BSON dates; float32 vectors as arrays of numbers
    - a value equal to its column default is simply absent (schemaless)

`to_doc(table, row)` and `to_row(table, doc)` convert both ways and are exact
inverses (the migration verifies row == to_row(to_doc(row)) for every row),
so every caller keeps receiving the same row dicts as before.
"""

import json
import re
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

# kinds: plain | keep (always stored, even when default) | datetime | date |
#        bbox | pose | json | reason | vector
_F = "%Y-%m-%d %H:%M:%S"

FIELD_MAP: Dict[str, List[tuple]] = {
    "videos": [
        ("video_id", "video_id", "keep"),
        ("title", "title", "plain"),
        ("region", "region", "plain"),
        ("youtube_url", "source.url", "plain"),
        ("source", "source.platform", "plain"),
        ("source_channel", "source.channel", "plain"),
        ("license", "source.license", "plain"),
        ("duration_seconds", "duration_s", "plain"),
        ("status", "status", "keep"),
        ("processed_date", "processed_at", "date"),
        ("speaker_id", "metadata.speaker_id", "plain"),
        ("num_speakers", "metadata.num_speakers", "plain"),
        ("gender", "metadata.gender", "plain"),
        ("age_group", "metadata.age_group", "plain"),
        ("environment", "metadata.environment", "plain"),
        ("background_noise", "metadata.background_noise", "plain"),
        ("total_segments", "stats.total_segments", "plain"),
        ("total_duration_extracted", "stats.total_duration_s", "plain"),
        ("avg_asd_score", "stats.avg_asd_score", "plain"),
        ("avg_syncnet_conf", "stats.avg_syncnet_conf", "plain"),
        ("error_message", "pipeline.error_message", "plain"),
        ("pipeline_version", "pipeline.version", "plain"),
        ("config_hash", "pipeline.config_hash", "plain"),
        ("whisper_model", "pipeline.whisper_model", "plain"),
        ("segmentation_strategy", "pipeline.segmentation_strategy", "plain"),
        ("created_at", "created_at", "datetime"),
    ],
    "segments": [
        ("segment_id", "segment_id", "keep"),
        ("video_id", "video_id", "keep"),
        ("clip_id", "clip_id", "plain"),
        ("speaker_id", "speaker_id", "keep"),
        ("track_id", "track_id", "plain"),
        ("start_time", "time.start", "keep"),
        ("end_time", "time.end", "keep"),
        ("duration", "time.duration", "keep"),
        ("text", "text", "keep"),
        ("original_text", "original_text", "plain"),
        ("num_words", "num_words", "plain"),
        ("num_chars", "num_chars", "plain"),
        ("quality_tier", "quality.tier", "keep"),
        ("asd_score", "quality.asd.score", "plain"),
        ("asd_method", "quality.asd.method", "plain"),
        ("syncnet_conf", "quality.syncnet.conf", "plain"),
        ("syncnet_method", "quality.syncnet.method", "plain"),
        ("whisper_conf", "quality.whisper.conf", "plain"),
        ("whisper_conf_min", "quality.whisper.conf_min", "plain"),
        ("whisper_conf_p25", "quality.whisper.conf_p25", "plain"),
        ("face_bbox", "face.bbox", "bbox"),
        ("face_visibility_ratio", "face.visibility_ratio", "plain"),
        ("head_pose_avg", "face.head_pose", "pose"),
        ("mouth_roi_method", "mouth.roi_method", "plain"),
        ("mouth_landmark_fail_rate", "mouth.landmark_fail_rate", "plain"),
        ("audio_speaker_label", "audio.speaker_label", "plain"),
        ("av_speaker_mismatch", "audio.av_speaker_mismatch", "plain"),
        ("boundary_start_type", "boundaries.start_type", "plain"),
        ("boundary_end_type", "boundaries.end_type", "plain"),
        ("whisper_standard_mediu", "transcripts.standard_mediu.text", "plain"),
        ("whisper_standard_mediu_conf", "transcripts.standard_mediu.conf", "plain"),
        ("whisper_standard_mediu_conf_min", "transcripts.standard_mediu.conf_min", "plain"),
        ("whisper_standard_mediu_conf_p25", "transcripts.standard_mediu.conf_p25", "plain"),
        ("whisper_dialectat_mediu", "transcripts.dialectat_mediu.text", "plain"),
        ("whisper_dialectat_mediu_conf", "transcripts.dialectat_mediu.conf", "plain"),
        ("whisper_dialectat_mediu_conf_min", "transcripts.dialectat_mediu.conf_min", "plain"),
        ("whisper_dialectat_mediu_conf_p25", "transcripts.dialectat_mediu.conf_p25", "plain"),
        ("whisper_dialectat_large", "transcripts.dialectat_large.text", "plain"),
        ("whisper_dialectat_large_conf", "transcripts.dialectat_large.conf", "plain"),
        ("whisper_dialectat_large_conf_min", "transcripts.dialectat_large.conf_min", "plain"),
        ("whisper_dialectat_large_conf_p25", "transcripts.dialectat_large.conf_p25", "plain"),
        ("text_largev3", "transcripts.large_v3.text", "plain"),
        ("transcript_consensus", "transcripts.consensus", "plain"),
        ("wer", "wer.value", "plain"),
        ("wer_word_count_ref", "wer.ref_word_count", "plain"),
        ("wer_medium_vs_large", "wer.medium_vs_large_v3", "plain"),
        ("wer_standard_vs_dialectat_mediu", "wer.standard_vs_dialectat_mediu", "plain"),
        ("wer_mediu_vs_large", "wer.mediu_vs_large", "plain"),
        ("wer_standard_vs_dialectat_large", "wer.standard_vs_dialectat_large", "plain"),
        ("wer_dialectat_mediu_vs_large", "wer.dialectat_mediu_vs_large", "plain"),
        ("review_status", "review.status", "plain"),
        ("reviewed_at", "review.reviewed_at", "datetime"),
        ("needs_review", "review.needs_review", "plain"),
        ("transcript_edited", "review.transcript_edited", "plain"),
        ("trimmed", "review.trimmed", "plain"),
        ("transcript_source", "review.transcript_source", "plain"),
        ("split", "split", "plain"),
        ("created_at", "created_at", "datetime"),
    ],
    "words": [   # embedded: segments.words[i]
        ("word", "word", "keep"),
        ("start_time", "start", "plain"),
        ("end_time", "end", "plain"),
        ("confidence", "confidence", "plain"),
        ("asd_score", "asd_score", "plain"),
    ],
    "speakers": [
        ("speaker_id", "speaker_id", "keep"),
        ("speaker_name", "name", "plain"),
        ("gender", "gender.value", "plain"),
        ("gender_confidence", "gender.confidence", "plain"),
        ("age_estimate", "age.estimate", "plain"),
        ("age_std", "age.std", "plain"),
        ("age_group", "age.group", "plain"),
        ("accent_region", "accent_region", "plain"),
        ("identity_match", "identity_match", "plain"),
        ("centroid", "centroid", "vector"),
    ],
    "segment_embeddings": [
        ("segment_id", "segment_id", "keep"),
        ("embedding", "embedding", "vector"),
    ],
    "dropped_clips": [   # embedded: videos.dropped_clips[]
        ("clip_id", "clip_id", "keep"),
        ("reason", "reason", "reason"),
        ("start_time", "time.start", "plain"),
        ("end_time", "time.end", "plain"),
        ("face_visibility", "scores.face_visibility", "plain"),
        ("whisper_conf", "scores.whisper_conf", "plain"),
        ("asd_score", "scores.asd_score", "plain"),
        ("syncnet_conf", "scores.syncnet_conf", "plain"),
        ("created_at", "created_at", "datetime"),
    ],
    "jobs": [
        ("id", "id", "keep"),
        ("type", "type", "keep"),
        ("status", "status", "keep"),
        ("params_json", "params", "json"),
        ("progress_json", "progress", "json"),
        ("created_at", "created_at", "datetime"),
        ("started_at", "started_at", "datetime"),
        ("finished_at", "finished_at", "datetime"),
        ("heartbeat_at", "heartbeat_at", "datetime"),
        ("claimed_by", "worker", "plain"),
        ("error", "error", "plain"),
        ("git_sha", "execution.git_sha", "plain"),
        ("config_hash", "execution.config_hash", "plain"),
        ("log_path", "execution.log_path", "plain"),
    ],
}

PATHS = {t: {col: path for col, path, _ in spec} for t, spec in FIELD_MAP.items()}
KINDS = {t: {col: kind for col, _, kind in spec} for t, spec in FIELD_MAP.items()}


def path(table: str, column: str) -> str:
    """Document path of a row column (e.g. review_status → review.status)."""
    return PATHS[table].get(column, column)


# ------------------------------------------------------------ converters

_REASON = re.compile(r"^([a-z_]+)=(-?\d+(?:\.\d+)?)$")


def _encode(kind: str, value):
    if kind == "datetime":
        try:
            return datetime.strptime(value, _F)
        except (TypeError, ValueError):
            return value
    if kind == "date":
        try:
            return datetime.strptime(value, "%Y-%m-%d")
        except (TypeError, ValueError):
            return value
    if kind == "bbox":
        try:
            x, y, w, h = json.loads(value)
            return {"x": x, "y": y, "w": w, "h": h}
        except (TypeError, ValueError):
            return value
    if kind == "pose":
        try:
            yaw, pitch, roll = json.loads(value)
            return {"yaw": yaw, "pitch": pitch, "roll": roll}
        except (TypeError, ValueError):
            return value
    if kind == "json":
        try:
            parsed = json.loads(value)
            return parsed if json.dumps(parsed) == value else value
        except (TypeError, ValueError):
            return value
    if kind == "reason":
        m = _REASON.match(value or "")
        if not m:
            return {"code": value}
        decimals = len(m.group(2).split(".")[1]) if "." in m.group(2) else 0
        return {"code": m.group(1), "value": float(m.group(2)), "decimals": decimals}
    if kind == "vector":
        if isinstance(value, (bytes, bytearray, memoryview)):
            return np.frombuffer(bytes(value), dtype=np.float32).astype(float).tolist()
        return [float(v) for v in np.asarray(value, dtype=np.float32)]
    return value


def _decode(kind: str, value):
    if kind == "datetime" and isinstance(value, datetime):
        return value.strftime(_F)
    if kind == "date" and isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    if kind == "bbox" and isinstance(value, dict):
        return json.dumps([value["x"], value["y"], value["w"], value["h"]])
    if kind == "pose" and isinstance(value, dict):
        return json.dumps([value["yaw"], value["pitch"], value["roll"]])
    if kind == "json" and not isinstance(value, str):
        return json.dumps(value)
    if kind == "reason" and isinstance(value, dict):
        if "value" not in value:
            return value.get("code", "")
        return f'{value["code"]}={value["value"]:.{value.get("decimals", 2)}f}'
    if kind == "vector" and isinstance(value, list):
        return np.asarray(value, dtype=np.float32).tobytes()
    return value


def encode_value(table: str, column: str, value):
    return _encode(KINDS[table].get(column, "plain"), value)


# ------------------------------------------------------------ dict helpers

def _set(doc: dict, dotted: str, value):
    parts = dotted.split(".")
    for p in parts[:-1]:
        doc = doc.setdefault(p, {})
    doc[parts[-1]] = value


def _get(doc: dict, dotted: str, missing):
    for p in dotted.split("."):
        if not isinstance(doc, dict) or p not in doc:
            return missing
        doc = doc[p]
    return doc


_MISSING = object()


def to_doc(table: str, row: Dict, defaults: Dict) -> dict:
    """Flat row → nested document (values equal to the default are omitted)."""
    doc: dict = {}
    for col, dotted, kind in FIELD_MAP[table]:
        if col not in row:
            continue
        value = row[col]
        if kind != "keep" and value == defaults.get(col) and \
                type(value) is type(defaults.get(col)):
            continue
        if kind == "reason" or value is None or value == "":
            stored = value if value in (None, "") else _encode(kind, value)
        else:
            stored = _encode(kind, value)
        _set(doc, dotted, stored)
    return doc


def to_row(table: str, doc: Optional[dict], defaults: Dict) -> Optional[dict]:
    """Nested document → flat row with EVERY column (defaults filled in)."""
    if doc is None:
        return None
    row = {}
    for col, dotted, kind in FIELD_MAP[table]:
        value = _get(doc, dotted, _MISSING)
        row[col] = defaults.get(col) if value is _MISSING else _decode(kind, value)
    return row


def set_ops(table: str, fields: Dict, defaults: Dict) -> dict:
    """Row fields → {$set, $unset} on document paths (default ⇒ unset)."""
    sets, unsets = {}, {}
    for col, value in fields.items():
        if col not in PATHS[table]:
            continue
        kind = KINDS[table][col]
        dotted = PATHS[table][col]
        if kind != "keep" and value == defaults.get(col) and \
                type(value) is type(defaults.get(col)):
            unsets[dotted] = ""
        elif value is None or value == "":
            sets[dotted] = value
        else:
            sets[dotted] = _encode(kind, value)
    ops = {}
    if sets:
        ops["$set"] = sets
    if unsets:
        ops["$unset"] = unsets
    return ops


def word_docs(words: List[Dict], defaults: Dict) -> List[dict]:
    """Row-shaped words (word, start_time, end_time, confidence, asd_score)."""
    return [to_doc("words", w, defaults) for w in words]


def word_rows(segment_id: str, docs: List[dict], defaults: Dict) -> List[dict]:
    rows = []
    for i, d in enumerate(docs or []):
        r = to_row("words", d, defaults)
        rows.append({"segment_id": segment_id, "word_index": i, **r})
    return rows
