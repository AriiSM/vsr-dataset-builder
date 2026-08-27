"""
Verify the dataset: does the DB agree with the files on disk?

Cross-checks data/catalog/dataset.db against processed/{video_id}/ and
prints a green/red report:
    - DB segments whose files are missing (face_crop / mouth_crop / audio / text)
    - files on disk with no DB row (orphans)
    - per-video counts vs videos.total_segments
    - words coverage (segments with no word rows)
    - tier / review distribution summary

Read-only: prints, never fixes. Exit code 1 when problems were found.

Usage (from the repo root):
    python backend/tools/verify_dataset.py
    python backend/tools/verify_dataset.py --video-id md_001 md_002
"""

import argparse
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _BACKEND_DIR.parent
sys.path[:0] = [str(_BACKEND_DIR), str(_BACKEND_DIR / "shared")]

from vsr_shared.catalog_db import CatalogDatabase  # noqa: E402

_ARTIFACTS = [
    ("face_crop", ".mp4", True),    # required
    ("mouth_crop", ".mp4", True),   # required
    ("audio", ".wav", False),       # optional (export.save_segment_audio)
    ("text", ".txt", True),         # required
]


def verify(catalog_dir: Path, processed_dir: Path, video_ids=None) -> int:
    db = CatalogDatabase(catalog_dir / "dataset.db")
    problems = 0

    segments = db.segments.all()
    if video_ids:
        segments = [s for s in segments if s["video_id"] in set(video_ids)]
    print(f"Catalog: {len(segments)} segment(s) în dataset.db")

    # 1. DB → disk: every segment's artifacts exist
    missing = []
    for seg in segments:
        base = processed_dir / seg["video_id"]
        for folder, ext, required in _ARTIFACTS:
            path = base / folder / f"{seg['segment_id']}{ext}"
            if required and not path.exists():
                missing.append(f"{seg['segment_id']}: lipsă {folder}{ext}")
    if missing:
        problems += len(missing)
        print(f"✗ {len(missing)} artefact(e) lipsă pe disc:")
        for m in missing[:20]:
            print(f"    {m}")
        if len(missing) > 20:
            print(f"    … și încă {len(missing) - 20}")
    else:
        print("✓ toate artefactele segmentelor din DB există pe disc")

    # 2. disk → DB: orphan files
    known = {s["segment_id"] for s in db.segments.all()}
    orphans = []
    if processed_dir.exists():
        for video_dir in sorted(processed_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            if video_ids and video_dir.name not in set(video_ids):
                continue
            face_dir = video_dir / "face_crop"
            for mp4 in sorted(face_dir.glob("*.mp4")) if face_dir.exists() else []:
                if mp4.stem not in known:
                    orphans.append(str(mp4.relative_to(processed_dir)))
    if orphans:
        problems += len(orphans)
        print(f"✗ {len(orphans)} fișier(e) orfane (pe disc, fără rând în DB):")
        for o in orphans[:20]:
            print(f"    {o}")
    else:
        print("✓ zero fișiere orfane")

    # 3. per-video counts vs videos.total_segments
    mismatches = []
    for video in db.videos.all():
        if video_ids and video["video_id"] not in set(video_ids):
            continue
        if video.get("total_segments") in (None, ""):
            continue
        stats = db.segments.video_stats(video["video_id"])
        actual = stats["num_segments"] if stats else 0
        if int(video["total_segments"]) != actual:
            mismatches.append(
                f"{video['video_id']}: videos.total_segments="
                f"{video['total_segments']} vs segments={actual}")
    if mismatches:
        problems += len(mismatches)
        print(f"✗ {len(mismatches)} nepotriviri de numărători:")
        for m in mismatches:
            print(f"    {m}")
    else:
        print("✓ numărătorile per video corespund")

    # 4. words coverage
    no_words = [
        s["segment_id"] for s in segments
        if not db.segments.words_for(s["segment_id"])
    ]
    if no_words:
        print(f"⚠ {len(no_words)} segment(e) fără rânduri în words "
              f"(annotation neparsabil la momentul scrierii)")
    else:
        print("✓ fiecare segment are cuvinte în words")

    # 5. summary
    overview = db.connection.execute("SELECT * FROM dataset_overview").fetchone()
    if overview and overview["num_segments"]:
        o = dict(overview)
        print(
            f"— total: {o['num_segments']} segmente · {o['num_videos']} video-uri"
            f" · {o['num_speakers']} vorbitori · {o['total_hours']} ore"
            f" · tiers A/B/C: {o['tier_a']}/{o['tier_b']}/{o['tier_c']}"
        )

    db.close()
    print("REZULTAT:", "✓ CURAT" if problems == 0 else f"✗ {problems} probleme")
    return 0 if problems == 0 else 1


def main():
    parser = argparse.ArgumentParser(description="Verify dataset.db ↔ disk")
    parser.add_argument("--catalog", type=Path,
                        default=_PROJECT_ROOT / "data" / "catalog")
    parser.add_argument("--processed", type=Path,
                        default=_PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--video-id", nargs="*", default=None)
    args = parser.parse_args()
    sys.exit(verify(args.catalog, args.processed, args.video_id))


if __name__ == "__main__":
    main()
