"""
cluster_speakers.py — per-video speaker clustering via ArcFace embeddings.

Why: pipeline.py auto-assigns speaker_id="{video_id}_spk0" assuming one
speaker per video. For interview / talk-show content (the MD corpus)
this is wrong — each video typically has 3-7 distinct speakers.

This script fixes the speaker_id column retroactively:
  1. For every segment in segments_index.csv, sample N frames from its
     face_crop video.
  2. Compute an ArcFace embedding (insightface buffalo_l, 512-d) per frame
     and average them per segment to get one robust embedding.
  3. Per video, run DBSCAN on cosine distance over those embeddings.
     Each cluster becomes a distinct speaker → "{video_id}_spk{N}".
     Outliers (DBSCAN label = -1) get a per-segment unique id.
  4. Write the new speaker_id back to segments_index.csv. Mirror into
     videos_master.csv (set "multiple" if >1 speaker found). Refresh
     speakers_registry.csv via ensure_speaker_exists + recompute_aggregates.

Defaults are tuned for ArcFace 512-d L2-normalised embeddings on
interview footage:
  --eps 0.40    (cosine distance)
  --min-samples 2

Increase eps if too many clusters; decrease if speakers get merged.

Usage:
    python backend/tools/cluster_speakers.py --dry-run
    python backend/tools/cluster_speakers.py
    python backend/tools/cluster_speakers.py --video-id md_001 md_005
    python backend/tools/cluster_speakers.py --eps 0.45 --frames-per-segment 3
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# Make `src.*` imports work regardless of cwd.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path[:0] = [str(_PROJECT_ROOT / "backend"), str(_PROJECT_ROOT / "backend" / "shared")]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vsr_shared.speakers_registry import ensure_speaker_exists, recompute_aggregates  # noqa: E402


def _extract_frames(video_path: Path, n: int) -> List[np.ndarray]:
    """Extract n frames uniformly from a video as BGR numpy arrays.

    Uses cv2.VideoCapture directly (no temp files, no ffmpeg subprocess
    overhead). Picks frames at positions total*(i+1)/(n+1) so the first
    and last frames (which can be black/transition) are skipped.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return []
        if n == 1:
            positions = [total // 2]
        else:
            positions = [int(total * (i + 1) / (n + 1)) for i in range(n)]
        frames: List[np.ndarray] = []
        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                frames.append(frame)
        return frames
    finally:
        cap.release()


def _direct_arcface_embedding(face_app, frame: np.ndarray) -> Optional[np.ndarray]:
    """Embed a face_crop frame DIRECTLY through ArcFace, skipping detection.

    Why: pipeline.py exports `face_crop` videos that are already a
    centered, smoothed face crop (RetinaFace + Kalman + Gaussian
    smoothing). Running RetinaFace again on a 256×256 close-up of a
    face frequently fails — the detector's anchors expect a face that
    occupies a small fraction of the frame, not 90%. Bypassing the
    detector entirely gives a usable embedding for every frame.

    Pipeline:
        BGR frame (any size) → resize to 112×112 → ArcFace recognition
        → 512-d unnormalised feature → L2-normalise → return.
    """
    import cv2

    if frame is None or frame.size == 0:
        return None
    aligned = cv2.resize(frame, (112, 112), interpolation=cv2.INTER_LINEAR)
    rec_model = face_app.models.get("recognition")
    if rec_model is None:
        return None
    feat = rec_model.get_feat(aligned)
    feat = np.asarray(feat).flatten().astype(np.float32)
    norm = float(np.linalg.norm(feat))
    return feat / norm if norm > 0 else None


def _embed_segment(
    face_app,
    frames: List[np.ndarray],
    use_detection: bool,
) -> Optional[np.ndarray]:
    """Average ArcFace embedding over `frames`. None if all frames failed.

    By default (use_detection=False) we trust pipeline.py's face crops and
    feed them straight to ArcFace. With --use-detection RetinaFace is run
    first; that path exists for completeness but is fragile on close-up
    crops and not recommended.
    """
    import cv2

    embs: List[np.ndarray] = []
    for img in frames:
        if not use_detection:
            emb = _direct_arcface_embedding(face_app, img)
            if emb is not None:
                embs.append(emb)
            continue

        # Detection path (legacy, optional). Upscale so RetinaFace anchors line up.
        h, w = img.shape[:2]
        if max(h, w) < 640:
            scale = 640.0 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_CUBIC)
        results = face_app.get(img)
        if not results:
            continue
        results.sort(key=lambda r: (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]),
                     reverse=True)
        if results[0].normed_embedding is not None:
            embs.append(results[0].normed_embedding)

    if not embs:
        return None
    avg = np.mean(np.stack(embs), axis=0)
    norm = float(np.linalg.norm(avg))
    return avg / norm if norm > 0 else None


def _build_face_app():
    """Initialise insightface FaceAnalysis app.

    Models are stored under the project's `models/insightface/` directory
    (alongside talknet_asd.pth / syncnet_v2.pth) instead of insightface's
    default `~/.insightface/`. First run auto-downloads buffalo_l (~250MB)
    into `models/insightface/models/buffalo_l/`.
    """
    from insightface.app import FaceAnalysis

    # Project-local model store. insightface appends "models/" to `root`,
    # so the final on-disk path is models/insightface/models/buffalo_l/*.
    models_root = _PROJECT_ROOT / "models" / "insightface"
    models_root.mkdir(parents=True, exist_ok=True)

    weights_dir = models_root / "models" / "buffalo_l"
    if not weights_dir.exists():
        print(f"[insightface] First-run download → {weights_dir} (~250MB)")
    else:
        print(f"[insightface] Using cached weights at {weights_dir}")

    # Try CUDA first, fall back to CPU. onnxruntime silently picks the next
    # available provider when CUDAExecutionProvider isn't installed.
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    app = FaceAnalysis(name="buffalo_l", root=str(models_root), providers=providers)
    # det_size 640x640 matches RetinaFace's training resolution; lower
    # resolutions (e.g. 256) make the detector miss faces that fill the
    # whole frame. det_thresh 0.3 (default 0.5) is more permissive — we
    # don't care about false positives because the largest detection is
    # picked per frame.
    app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)
    return app


def _cluster_per_video(
    embeddings_by_video: Dict[str, Dict[str, np.ndarray]],
    eps: float,
    min_samples: int,
) -> Dict[str, str]:
    """Return {segment_id: assigned_speaker_id} for every embedded segment.

    DBSCAN on cosine distance per video. Outliers (label -1) get a unique
    fallback speaker id of the form "{video_id}_spkOUT_{seg_id_short}".
    """
    from sklearn.cluster import DBSCAN

    assignments: Dict[str, str] = {}
    for video_id, seg_to_emb in embeddings_by_video.items():
        if not seg_to_emb:
            continue
        seg_ids = list(seg_to_emb.keys())
        X = np.stack([seg_to_emb[s] for s in seg_ids])

        # Single segment: trivially one speaker.
        if len(seg_ids) == 1:
            assignments[seg_ids[0]] = f"{video_id}_spk0"
            continue

        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(X)
        labels = clustering.labels_

        # Re-number cluster labels so they start at 0 and skip outliers.
        cluster_remap: Dict[int, int] = {}
        next_id = 0
        for lab in labels:
            if lab == -1:
                continue
            if lab not in cluster_remap:
                cluster_remap[lab] = next_id
                next_id += 1

        for seg_id, lab in zip(seg_ids, labels):
            if lab == -1:
                # Lone face that didn't cluster — give it its own bucket so it
                # doesn't pollute existing speakers. Suffix uses last 6 chars
                # of segment_id for uniqueness without being giant.
                suffix = seg_id[-6:] if len(seg_id) >= 6 else seg_id
                assignments[seg_id] = f"{video_id}_spkOUT_{suffix}"
            else:
                assignments[seg_id] = f"{video_id}_spk{cluster_remap[lab]}"
    return assignments


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data", default="./data",
                   help="Project data directory (contains metadata/, processed/).")
    p.add_argument("--video-id", nargs="+", metavar="ID",
                   help="Cluster only these video IDs (default: every video in segments_index).")
    p.add_argument("--eps", type=float, default=0.40,
                   help="DBSCAN eps in cosine distance (default 0.40 — typical for ArcFace).")
    p.add_argument("--min-samples", type=int, default=2,
                   help="DBSCAN min_samples (default 2 — every speaker should appear in ≥2 segments).")
    p.add_argument("--frames-per-segment", type=int, default=3,
                   help="Frames sampled per segment for embedding (default 3 — covers head movement). "
                        "Increase to 5+ for very long clips, decrease to 1 for speed.")
    p.add_argument("--use-detection", action="store_true",
                   help="Run RetinaFace before ArcFace. Off by default: face_crop videos are "
                        "already centered/aligned so the detector is redundant and often fails.")
    p.add_argument("--debug-dump-frames", action="store_true",
                   help="Save the first 6 extracted frames to data/cache/cluster_debug/ for visual inspection.")
    p.add_argument("--dry-run", action="store_true",
                   help="Embed + cluster but do not write anything.")
    p.add_argument("--limit", type=int, default=None,
                   help="Hard cap on segments processed (for quick smoke tests).")
    args = p.parse_args()

    data_dir = Path(args.data)
    metadata_dir = data_dir / "metadata"
    processed_dir = data_dir / "processed"
    seg_csv = metadata_dir / "segments_index.csv"
    master_csv = metadata_dir / "videos_master.csv"

    if not seg_csv.exists():
        print(f"ERROR: {seg_csv} not found", file=sys.stderr)
        return 1

    df = pd.read_csv(seg_csv)
    if "speaker_id" not in df.columns:
        df["speaker_id"] = ""

    # Filter target rows.
    target = df.copy()
    if args.video_id:
        target = target[target["video_id"].astype(str).isin(args.video_id)]
    if args.limit:
        target = target.head(args.limit)
    if target.empty:
        print("No segments match the filter — nothing to cluster.")
        return 0

    print(f"Will embed {len(target)} segments across {target['video_id'].nunique()} video(s).")

    # Build the face app once (heavy: model load + GPU init).
    print("Loading insightface buffalo_l model...")
    try:
        face_app = _build_face_app()
    except Exception as e:
        print(f"ERROR: insightface init failed: {e}", file=sys.stderr)
        print("Try: pip install insightface onnxruntime", file=sys.stderr)
        return 1

    embeddings_by_video: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
    n_no_face = 0
    n_no_video = 0
    n_no_frames = 0

    debug_dir = _PROJECT_ROOT / "data" / "cache" / "cluster_debug"
    if args.debug_dump_frames:
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"[debug] Dumping first 6 frames to {debug_dir}")

    for i, (_, row) in enumerate(target.iterrows(), 1):
        seg_id = str(row["segment_id"])
        vid = str(row["video_id"])
        face_video = processed_dir / vid / "face_crop" / f"{seg_id}.mp4"

        if not face_video.exists():
            n_no_video += 1
            if i % 100 == 0 or i == len(target):
                print(f"  [{i}/{len(target)}] {seg_id}: face_crop missing — skipped")
            continue

        frames = _extract_frames(face_video, args.frames_per_segment)
        if not frames:
            n_no_frames += 1
            print(f"  [{i}/{len(target)}] {seg_id}: cv2 failed to read any frame")
            continue

        # First call is also the right place to dump debug frames.
        if args.debug_dump_frames and (sum(len(v) for v in embeddings_by_video.values()) +
                                       n_no_face + n_no_frames) < 6:
            import cv2
            for j, fr in enumerate(frames):
                p_out = debug_dir / f"{seg_id}_frame{j}.jpg"
                cv2.imwrite(str(p_out), fr)

        emb = _embed_segment(face_app, frames, use_detection=args.use_detection)
        if emb is None:
            n_no_face += 1
            if i % 50 == 0:
                print(f"  [{i}/{len(target)}] {seg_id}: embedding failed (frames OK)")
            continue

        embeddings_by_video[vid][seg_id] = emb
        if i % 50 == 0 or i == len(target):
            print(f"  [{i}/{len(target)}] embedded — running totals: "
                  f"{sum(len(v) for v in embeddings_by_video.values())} ok, "
                  f"{n_no_face} no-face, {n_no_video} missing-video, {n_no_frames} no-frames")

    # Cluster and report.
    assignments = _cluster_per_video(embeddings_by_video, args.eps, args.min_samples)
    print(f"\nClustering done: {len(assignments)} segments assigned.")

    # Aggregate: per video → set of distinct speaker_ids, and per speaker → segment list.
    per_video_speakers: Dict[str, set] = {}
    per_speaker_segments: Dict[str, List[str]] = defaultdict(list)
    for seg_id, sid in assignments.items():
        if "_spk" in sid:
            video_part = sid.split("_spk")[0]
            per_video_speakers.setdefault(video_part, set()).add(sid)
        per_speaker_segments[sid].append(seg_id)

    print("\n" + "=" * 60)
    print("CLUSTERING SUMMARY")
    print("=" * 60)
    for vid, sids in sorted(per_video_speakers.items()):
        genuine_ids = sorted(s for s in sids if "_spkOUT_" not in s)
        outlier_ids = [s for s in sids if "_spkOUT_" in s]
        n_outlier_segs = sum(len(per_speaker_segments[s]) for s in outlier_ids)

        outlier_note = (
            f"  (+ {len(outlier_ids)} outlier cluster(s) covering {n_outlier_segs} segment(s))"
            if outlier_ids else ""
        )
        print(f"\n  {vid}: {len(genuine_ids)} speaker(s){outlier_note}")
        # Per-speaker line: id, count, sample segment IDs.
        for sid in genuine_ids:
            segs = per_speaker_segments[sid]
            sample = ", ".join(segs[:3])
            more = f" (+ {len(segs) - 3} more)" if len(segs) > 3 else ""
            print(f"    {sid:>30s}  {len(segs):>4} segments  e.g. {sample}{more}")
        if outlier_ids:
            print(f"    {'<outliers>':>30s}  {n_outlier_segs:>4} segments  → review manually in UI")
    print()

    if args.dry_run:
        print("\n(--dry-run: no files written)")
        return 0

    # ── Write segments_index ───────────────────────────────────────────
    for seg_id, new_sid in assignments.items():
        df.loc[df["segment_id"].astype(str) == seg_id, "speaker_id"] = new_sid
    df.to_csv(seg_csv, index=False)
    print(f"\nUpdated {len(assignments)} rows in {seg_csv.name}")

    # ── Mirror into videos_master ──────────────────────────────────────
    if master_csv.exists():
        try:
            mdf = pd.read_csv(master_csv)
            for col in mdf.select_dtypes(include=["float64", "object"]).columns:
                mdf[col] = mdf[col].astype(object)
            if "speaker_id" not in mdf.columns:
                mdf["speaker_id"] = ""
            for vid, sids in per_video_speakers.items():
                m_mask = mdf["video_id"].astype(str) == vid
                if not m_mask.any():
                    continue
                # 1 speaker → put the id; >1 → "multiple" sentinel.
                genuine_ids = [s for s in sids if "_spkOUT_" not in s]
                if len(genuine_ids) == 1:
                    mdf.loc[m_mask, "speaker_id"] = genuine_ids[0]
                else:
                    mdf.loc[m_mask, "speaker_id"] = "multiple"
            mdf.to_csv(master_csv, index=False)
            print(f"Mirrored speaker_id into {master_csv.name}")
        except PermissionError:
            print(f"WARNING: {master_csv.name} is open — skipped master mirror.")

    # ── Registry stubs + aggregates ────────────────────────────────────
    unique_speakers = sorted(set(assignments.values()))
    for sid in unique_speakers:
        ensure_speaker_exists(metadata_dir, sid)
    n_agg = recompute_aggregates(metadata_dir)
    print(f"Ensured {len(unique_speakers)} speaker(s) in registry; aggregated {n_agg} row(s).")
    print(f"\nDone. Refresh the Stats tab to see the new speakers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
