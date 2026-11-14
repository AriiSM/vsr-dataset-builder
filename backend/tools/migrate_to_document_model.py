"""
Migrate the flat catalog (one collection per old SQLite table, database
`lip_reading`) into the MongoDB document model (database `vsr-dataset-md`).

    words          → embedded in segments.words[]
    dropped_clips  → embedded in videos.dropped_clips[]
    JSON strings   → objects (face.bbox, face.head_pose, jobs.params/progress)
    flat columns   → sub-documents (quality{}, transcripts{}, review{}, ...)
    timestamps     → BSON dates;  float32 BLOBs → arrays of numbers

The source database is only read. Every converted document is checked on the
fly: converting it back must give exactly the original row (same values and
types), the same for each embedded word and dropped clip.

Usage:
    python backend/tools/migrate_to_document_model.py [--source lip_reading]
           [--target vsr-dataset-md] [--drop]
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "backend"), str(REPO / "backend" / "shared")]

from vsr_shared import catalog_db as C  # noqa: E402
from vsr_shared import catalog_model as M  # noqa: E402

BATCH = 1000


def flat(doc: dict) -> dict:
    doc = dict(doc)
    doc.pop("_id", None)
    return doc


class Checker:
    def __init__(self):
        self.checked = 0
        self.errors = []

    def same(self, table, key, original: dict, back: dict):
        self.checked += 1
        for col, value in original.items():
            if col not in back:
                continue
            b = back[col]
            if b != value or type(b) is not type(value):
                if len(self.errors) < 20:
                    self.errors.append(f"{table}[{key}].{col}: {value!r} -> {b!r}")
                return False
        return True


def migrate(src, dst, check: Checker):
    defaults = C.COLUMNS

    # videos + embedded dropped clips --------------------------------------
    dropped = {}
    for d in src["dropped_clips"].find().sort([("video_id", 1), ("clip_id", 1)]):
        row = flat(d)
        entry = M.to_doc("dropped_clips", row, defaults["dropped_clips"])
        back = C._row("dropped_clips", entry)
        back["video_id"] = row["video_id"]
        check.same("dropped_clips", d["_id"], row, back)
        dropped.setdefault(row["video_id"], []).append(entry)
    docs = []
    for d in src["videos"].find().sort("_id", 1):
        row = flat(d)
        doc = {"_id": row["video_id"], **M.to_doc("videos", row, defaults["videos"])}
        check.same("videos", d["_id"], row, C._row("videos", doc))
        if row["video_id"] in dropped:
            doc["dropped_clips"] = dropped.pop(row["video_id"])
        docs.append(doc)
    for vid, entries in dropped.items():   # orphans: keep them, on a stub video
        print(f"  ! dropped clips for unknown video {vid}: {len(entries)} (stub created)")
        docs.append({"_id": vid, "video_id": vid, "status": "pending", "dropped_clips": entries})
    dst["videos"].insert_many(docs)
    print(f"videos              {len(docs):>8} docs ({sum(len(d.get('dropped_clips', [])) for d in docs)} dropped clips embedded)")

    # segments + embedded words (streaming merge join on segment_id) -------
    words_cur = src["words"].find().sort([("segment_id", 1), ("word_index", 1)])
    pending_word = next(words_cur, None)
    batch, n_seg, n_words = [], 0, 0
    for d in src["segments"].find().sort("_id", 1):
        row = flat(d)
        sid = row["segment_id"]
        doc = {"_id": sid, **M.to_doc("segments", row, defaults["segments"])}
        check.same("segments", sid, row, C._row("segments", doc))
        while pending_word is not None and pending_word["segment_id"] < sid:
            print(f"  ! orphan word for missing segment {pending_word['segment_id']}")
            pending_word = next(words_cur, None)
        words = []
        while pending_word is not None and pending_word["segment_id"] == sid:
            words.append(flat(pending_word))
            pending_word = next(words_cur, None)
        if words:
            if [w["word_index"] for w in words] != list(range(len(words))):
                print(f"  ! {sid}: non-contiguous word_index — order kept")
            doc["words"] = M.word_docs(words, defaults["words"])
            for w, b in zip(words, M.word_rows(sid, doc["words"], defaults["words"])):
                b["word_index"] = w["word_index"]
                check.same("words", f"{sid}#{w['word_index']}", w, b)
            n_words += len(words)
        batch.append(doc)
        if len(batch) >= BATCH:
            dst["segments"].insert_many(batch)
            n_seg += len(batch)
            batch.clear()
            print(f"\rsegments            {n_seg:>8}", end="", flush=True)
    if batch:
        dst["segments"].insert_many(batch)
        n_seg += len(batch)
    print(f"\rsegments            {n_seg:>8} docs ({n_words} words embedded)")

    # speakers / embeddings / jobs ------------------------------------------
    for table in ("speakers", "segment_embeddings", "jobs"):
        key = {"speakers": "speaker_id", "segment_embeddings": "segment_id", "jobs": "id"}[table]
        batch, n = [], 0
        for d in src[table].find().sort("_id", 1):
            row = flat(d)
            doc = {"_id": row[key], **M.to_doc(table, row, defaults[table])}
            check.same(table, d["_id"], row, C._row(table, doc))
            batch.append(doc)
            if len(batch) >= BATCH:
                dst[table].insert_many(batch)
                n += len(batch)
                batch.clear()
        if batch:
            dst[table].insert_many(batch)
            n += len(batch)
        print(f"{table:20s}{n:>8} docs")

    for d in src["counters"].find():
        dst["counters"].insert_one(d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="lip_reading")
    ap.add_argument("--target", default="vsr-dataset-md")
    ap.add_argument("--drop", action="store_true", help="recreate the target database")
    args = ap.parse_args()

    client = C._client(C.mongo_uri())
    src, dst = client[args.source], client[args.target]
    if args.target in client.list_database_names():
        if not args.drop:
            sys.exit(f"database '{args.target}' exists; rerun with --drop to recreate it")
        client.drop_database(args.target)

    check = Checker()
    migrate(src, dst, check)
    db = C.CatalogDatabase(db_name=args.target)   # indexes, views, meta

    print("\n-- verification --")
    print(f"round-trip checks: {check.checked:,} rows, {len(check.errors)} mismatches")
    for e in check.errors:
        print("  ", e)
    counts = {
        "videos": (src["videos"].count_documents({}), dst["videos"].count_documents({})),
        "segments": (src["segments"].count_documents({}), dst["segments"].count_documents({})),
        "words": (src["words"].count_documents({}),
                  next(dst["segments"].aggregate([{"$project": {"n": {"$size": {"$ifNull": ["$words", []]}}}},
                                                  {"$group": {"_id": None, "n": {"$sum": "$n"}}}]), {"n": 0})["n"]),
        "dropped_clips": (src["dropped_clips"].count_documents({}),
                          next(dst["videos"].aggregate([{"$project": {"n": {"$size": {"$ifNull": ["$dropped_clips", []]}}}},
                                                        {"$group": {"_id": None, "n": {"$sum": "$n"}}}]), {"n": 0})["n"]),
        "speakers": (src["speakers"].count_documents({}), dst["speakers"].count_documents({})),
        "segment_embeddings": (src["segment_embeddings"].count_documents({}),
                               dst["segment_embeddings"].count_documents({})),
        "jobs": (src["jobs"].count_documents({}), dst["jobs"].count_documents({})),
    }
    ok = not check.errors
    for name, (a, b) in counts.items():
        ok &= a == b
        print(f"{name:20s} source={a:>8} target={b:>8} {'OK' if a == b else 'FAIL'}")
    ov_src = next(src["dataset_overview"].find({}, {"_id": 0}), {})
    ov_dst = db.overview()
    print("overview source:", ov_src)
    print("overview target:", ov_dst)
    ok &= ov_src == ov_dst
    print("RESULT:", "OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
