"""
predict_speaker_metadata.py — auto-fill gender + age_group + accent_region
for every row in speakers_registry.csv.

Pipeline per speaker:
  1. Pull up to N segments from segments_index.csv whose speaker_id matches.
  2. Extract one mid-frame from each segment's face_crop video (cv2).
  3. Run buffalo_l face detection + genderage on each frame
     (insightface auto-uses the genderage.onnx already in
      models/insightface/models/buffalo_l/, so no extra download).
  4. Aggregate across samples:
       - gender:    majority vote (M/F)
       - age:       median across detections
       - age_group: bucket the median (18-30 / 31-50 / 51+)
  5. accent_region: majority vote of `region` column from videos_master.csv
     across the videos this speaker appears in.
  6. Write back to speakers_registry.csv.

Default: SKIPS any row where the field is already filled in (manual work
is preserved). Pass --overwrite to force-replace existing values.

Speaker name is NOT predicted — there is no offline / free way to map a
face to a real Romanian person's name. That column stays manual.

Usage:
    python backend/tools/predict_speaker_metadata.py --dry-run
    python backend/tools/predict_speaker_metadata.py
    python backend/tools/predict_speaker_metadata.py --speaker-id md_001_spk0 md_001_spk1
    python backend/tools/predict_speaker_metadata.py --overwrite
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path[:0] = [str(_PROJECT_ROOT / "backend"), str(_PROJECT_ROOT / "backend" / "shared")]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from vsr_shared.speakers_registry import load_speakers, save_speakers  # noqa: E402


def _build_face_app():
    """Same insightface bootstrap as cluster_speakers.py — shared model dir."""
    from insightface.app import FaceAnalysis
    models_root = _PROJECT_ROOT / "models" / "insightface"
    models_root.mkdir(parents=True, exist_ok=True)
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    app = FaceAnalysis(name="buffalo_l", root=str(models_root), providers=providers)
    # det_size 640x640 + relaxed threshold so detection succeeds on close-up crops.
    app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.3)
    return app


def _extract_mid_frame(video_path: Path) -> Optional[np.ndarray]:
    """Read one frame from the middle of `video_path` via cv2."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        ret, frame = cap.read()
        return frame if ret and frame is not None and frame.size > 0 else None
    finally:
        cap.release()


def _detect_gender_age(face_app, frame: np.ndarray) -> Optional[Tuple[int, float]]:
    """Run RetinaFace + genderage on `frame`. Returns (gender_int, age_years).

    `gender_int` follows insightface convention: 0 = female, 1 = male.
    Returns None if no face is detected — the speaker contributes one
    fewer sample to the aggregate.
    """
    import cv2
    h, w = frame.shape[:2]
    if max(h, w) < 640:
        scale = 640.0 / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_CUBIC)
    results = face_app.get(frame)
    if not results:
        return None
    # Pick the largest detection (close-up faces dominate face_crop output).
    results.sort(key=lambda r: (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]),
                 reverse=True)
    face = results[0]
    g = getattr(face, "gender", None)
    a = getattr(face, "age", None)
    if g is None or a is None:
        return None
    return int(g), float(a)


def _bucket_age(age: float) -> str:
    """Map a numeric age to the schema's age_group enum."""
    if age < 31:
        return "18-30"
    if age < 51:
        return "31-50"
    return "51+"


def _aggregate(samples: List[Tuple[int, float]]) -> Optional[Tuple[str, str, float]]:
    """Majority-vote gender, median age, bucket age_group."""
    if not samples:
        return None
    genders = [g for g, _ in samples]
    ages = [a for _, a in samples]
    male_count = sum(genders)
    gender = "M" if male_count > len(genders) / 2 else "F"
    age_med = float(np.median(ages))
    return gender, _bucket_age(age_med), age_med


def _majority_region(videos: List[str], master_df: pd.DataFrame) -> Optional[str]:
    """Majority vote of `region` column across the speaker's videos."""
    if master_df.empty or "video_id" not in master_df.columns or "region" not in master_df.columns:
        return None
    regions = []
    for vid in videos:
        row = master_df[master_df["video_id"].astype(str) == vid]
        if row.empty:
            continue
        r = row.iloc[0].get("region", "")
        if pd.notna(r) and str(r).strip():
            regions.append(str(r).strip())
    if not regions:
        return None
    return Counter(regions).most_common(1)[0][0]


def _is_blank(value) -> bool:
    """True if a cell from speakers_registry should be treated as empty."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() in ("", "nan", "None")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--data", default="./data",
                   help="Project data directory.")
    p.add_argument("--speaker-id", nargs="+", metavar="ID",
                   help="Predict only for these speaker IDs (default: all).")
    p.add_argument("--samples-per-speaker", type=int, default=3,
                   help="How many segments to sample per speaker for face detection (default 3).")
    p.add_argument("--overwrite", action="store_true",
                   help="Force-replace fields that are already filled. Default: skip filled cells.")
    p.add_argument("--force-region", metavar="CODE",
                   help="Bypass the majority-vote-from-videos_master logic and set "
                        "accent_region to this value for EVERY speaker (e.g. MD). "
                        "Useful when the whole dataset comes from the same region.")
    p.add_argument("--dry-run", action="store_true",
                   help="Run prediction + report but do not write speakers_registry.csv.")
    args = p.parse_args()

    data_dir = Path(args.data)
    metadata_dir = data_dir / "metadata"
    processed_dir = data_dir / "processed"

    seg_csv = metadata_dir / "segments_index.csv"
    master_csv = metadata_dir / "videos_master.csv"

    if not seg_csv.exists():
        print(f"ERROR: {seg_csv} not found", file=sys.stderr)
        return 1

    seg_df = pd.read_csv(seg_csv)
    if "speaker_id" not in seg_df.columns:
        print("ERROR: segments_index.csv has no speaker_id column. "
              "Run cluster_speakers.py first.", file=sys.stderr)
        return 1

    master_df = pd.read_csv(master_csv) if master_csv.exists() else pd.DataFrame()

    # ── Self-sync: make sure registry has a stub for every speaker_id that
    # appears in segments_index. Covers the case where cluster_speakers.py
    # crashed before its ensure_speaker_exists / recompute_aggregates pass
    # (so the new clusters live in segments_index but not in the registry).
    from vsr_shared.speakers_registry import ensure_speaker_exists, recompute_aggregates
    speakers_df = load_speakers(metadata_dir)
    unique_in_segments = set(
        seg_df["speaker_id"].dropna().astype(str).tolist()
    ) - {"", "nan"}
    existing_in_registry = set(speakers_df["speaker_id"].astype(str).tolist())
    missing = sorted(unique_in_segments - existing_in_registry)
    if missing:
        print(f"Sync: adding {len(missing)} missing speaker stub(s) to registry...")
        for sid in missing:
            ensure_speaker_exists(metadata_dir, sid)
        print("Sync: recomputing aggregates...")
        recompute_aggregates(metadata_dir)
        speakers_df = load_speakers(metadata_dir)  # reload with new rows
    if speakers_df.empty:
        print("speakers_registry.csv is empty — run cluster_speakers.py first.")
        return 0

    if args.speaker_id:
        speakers_df = speakers_df[speakers_df["speaker_id"].astype(str).isin(args.speaker_id)]
        if speakers_df.empty:
            print(f"None of the requested speaker_ids exist in registry.")
            return 1

    # Cast destination columns to object NOW so that string assignments inside
    # the loop don't trip the pandas FutureWarning ("Setting an item of
    # incompatible dtype is deprecated...") when writing into a NaN/float64 cell.
    for col in ("gender", "age_group", "accent_region"):
        if col in speakers_df.columns:
            speakers_df[col] = speakers_df[col].astype(object)

    print(f"Predicting metadata for {len(speakers_df)} speaker(s)...")
    print("Loading insightface buffalo_l (with genderage)...")
    face_app = _build_face_app()
    print()

    n_updated = 0
    n_no_data = 0
    n_already_done = 0
    summary_rows: List[Tuple[str, Optional[str], Optional[str], Optional[float], Optional[str]]] = []

    for _, sp_row in speakers_df.iterrows():
        sid = str(sp_row["speaker_id"])

        # Resume support: if every editable column is already filled and the
        # user did NOT pass --overwrite, skip the speaker entirely (saves the
        # 5-15s of detection per speaker on a re-run after Ctrl+C).
        if not args.overwrite:
            g_cur  = sp_row.get("gender")
            ag_cur = sp_row.get("age_group")
            ac_cur = sp_row.get("accent_region")
            if not _is_blank(g_cur) and not _is_blank(ag_cur) and not _is_blank(ac_cur):
                n_already_done += 1
                continue
        speaker_segs = seg_df[seg_df["speaker_id"].astype(str) == sid]
        if speaker_segs.empty:
            print(f"  {sid}: no segments — skipped")
            n_no_data += 1
            continue

        # Collect samples until we have N successful detections (or run out).
        samples: List[Tuple[int, float]] = []
        candidate_rows = speaker_segs.head(args.samples_per_speaker * 3)
        for _, srow in candidate_rows.iterrows():
            if len(samples) >= args.samples_per_speaker:
                break
            seg_id = str(srow["segment_id"])
            video_id = str(srow["video_id"])
            face_video = processed_dir / video_id / "face_crop" / f"{seg_id}.mp4"
            if not face_video.exists():
                continue
            frame = _extract_mid_frame(face_video)
            if frame is None:
                continue
            det = _detect_gender_age(face_app, frame)
            if det is not None:
                samples.append(det)

        agg = _aggregate(samples)
        if args.force_region:
            # Hard override — every speaker gets the same accent code.
            accent = args.force_region
        else:
            videos_for_speaker = sorted(set(speaker_segs["video_id"].astype(str).tolist()))
            accent = _majority_region(videos_for_speaker, master_df)

        if agg is None:
            print(f"  {sid:30s} no face on any sample (tried {len(candidate_rows)}) — only accent")
            gender, age_group, age_med = None, None, None
        else:
            gender, age_group, age_med = agg

        summary_rows.append((sid, gender, age_group, age_med, accent))

        if agg is None and accent is None:
            n_no_data += 1
            continue

        # Write back, respecting --overwrite policy.
        sp_mask = speakers_df["speaker_id"].astype(str) == sid
        wrote_anything = False
        for col, val in (("gender", gender), ("age_group", age_group), ("accent_region", accent)):
            if val is None:
                continue
            cur = speakers_df.loc[sp_mask, col].iloc[0] if col in speakers_df.columns else None
            if not args.overwrite and not _is_blank(cur):
                continue
            speakers_df.loc[sp_mask, col] = val
            wrote_anything = True
        if wrote_anything:
            n_updated += 1
            # INCREMENTAL SAVE: persist progress after every speaker so that
            # Ctrl+C / crashes / power loss don't throw away the work done so
            # far. Cheap (~few KB CSV write) compared to detection cost.
            if not args.dry_run:
                try:
                    save_speakers(metadata_dir, speakers_df)
                except Exception as e:
                    print(f"  WARN: incremental save failed for {sid}: {e}",
                          file=sys.stderr)
            print(f"  ✓ {sid:30s} g={gender or '—':3s} age={age_group or '—':6s} accent={accent or '—'}  (saved)")

    # ── Report ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"Predicted metadata for {len(summary_rows)} speaker(s):")
    print(f"  {'speaker_id':30s} {'gender':6s} {'age':6s} {'age_group':10s} {'accent'}")
    for sid, g, ag, am, ac in summary_rows:
        g_str  = g  or "—"
        am_str = f"{am:.0f}" if am is not None else "—"
        ag_str = ag or "—"
        ac_str = ac or "—"
        print(f"  {sid[:30]:30s} {g_str:6s} {am_str:>6s} {ag_str:10s} {ac_str}")

    print(f"\nSummary: {n_updated} row(s) updated"
          f"{' (saved incrementally)' if not args.dry_run else ''}, "
          f"{n_already_done} already done (skipped), "
          f"{n_no_data} no usable data.")

    if args.dry_run:
        print("\n(--dry-run: speakers_registry.csv NOT modified)")
        return 0

    if n_updated == 0:
        print("Nothing to write.")
        return 0

    # Re-load full registry and merge our changes (avoid losing rows that
    # weren't in the --speaker-id filter).
    if args.speaker_id:
        full = load_speakers(metadata_dir)
        for sid in speakers_df["speaker_id"].astype(str):
            mask_full = full["speaker_id"].astype(str) == sid
            mask_part = speakers_df["speaker_id"].astype(str) == sid
            for col in ("gender", "age_group", "accent_region"):
                if col in speakers_df.columns and col in full.columns:
                    full.loc[mask_full, col] = speakers_df.loc[mask_part, col].iloc[0]
        save_speakers(metadata_dir, full)
    else:
        save_speakers(metadata_dir, speakers_df)
    print(f"Wrote speakers_registry.csv ({n_updated} row(s) modified).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
