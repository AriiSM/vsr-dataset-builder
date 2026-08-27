"""
backfill_syncnet.py — compute SyncNet confidence for segments that were
exported before `syncnet.enabled` was flipped on.

The pipeline stored syncnet_conf=0.0 for every segment while the flag was
off. This script re-scores those segments by running SyncNet directly on
the already face-cropped 256×256 mp4 in data/processed/{video_id}/face_crop/.
No re-download, no re-detect — just load frames + mfcc from the existing
file and run the model.

By default it targets segments that are "pregătite pentru review" — present
in segments_index.csv but not yet decided in review_status.json
(status approved/rejected). Pass --force to re-score everything, or
--video-id / --limit to scope it down.

Idempotent. Checkpoints to a .tmp file every --checkpoint-every rows
(atomic replace) so Ctrl+C is safe.

Usage:
    python backend/tools/backfill_syncnet.py                       # pending-review only
    python backend/tools/backfill_syncnet.py --video-id md_001     # one video
    python backend/tools/backfill_syncnet.py --force               # ALL rows
    python backend/tools/backfill_syncnet.py --limit 50            # smoke test
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path[:0] = [str(_PROJECT_ROOT / "backend"), str(_PROJECT_ROOT / "backend" / "shared")]

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402

from services.speaker_detector.syncnet_model import SyncNetModel  # noqa: E402


# ---- SyncNet windowed scoring (no track object needed) -------------------

V_WIN = 5      # video frames per window
A_WIN = 20     # audio frames per window (0.2s at 100fps mfcc)
BATCH = 32
DEFAULT_OFFSETS = list(range(-15, 16))   # ±15 frames, 31 candidates


def _read_face_crop_frames(mp4_path: Path, size: int = 240) -> np.ndarray:
    """Read every frame from an already-face-cropped mp4 and resize to size×size."""
    cap = cv2.VideoCapture(str(mp4_path))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(fr, (size, size)))
    cap.release()
    return np.array(frames)


def _extract_mfcc(mp4_path: Path, sample_rate: int = 16000, n_mfcc: int = 13) -> np.ndarray:
    """Extract MFCC at 100fps from the audio stream inside the mp4."""
    import python_speech_features
    import subprocess
    from scipy.io import wavfile

    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg = "ffmpeg"

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t:
        wav_path = t.name
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", str(mp4_path),
             "-vn", "-acodec", "pcm_s16le",
             "-ar", str(sample_rate), "-ac", "1", wav_path],
            capture_output=True, check=True,
        )
        sr, audio = wavfile.read(wav_path)
        return python_speech_features.mfcc(
            audio, sr, numcep=n_mfcc, winlen=0.025, winstep=0.010
        )
    finally:
        Path(wav_path).unlink(missing_ok=True)


def _score_segment(
    model: SyncNetModel,
    device: str,
    frames: np.ndarray,
    mfcc: np.ndarray,
    offsets: List[int] = DEFAULT_OFFSETS,
) -> float:
    """Return SyncNet confidence in [0, 1] (max cosine across offsets, normalised)."""
    if len(frames) < V_WIN or len(mfcc) < A_WIN:
        return 0.0

    scores = []
    for off in offsets:
        v_off = max(off, 0)
        a_off = max(-off * 4, 0)
        T_v = len(frames) - v_off
        T_a = len(mfcc) - a_off
        if T_v < V_WIN or T_a < A_WIN:
            scores.append(-1.0)
            continue
        steps = min(T_v // V_WIN, T_a // A_WIN)

        v_windows, a_windows = [], []
        for i in range(steps):
            v_windows.append(
                torch.from_numpy(frames[v_off + i*V_WIN : v_off + (i+1)*V_WIN])
                     .float().permute(3, 0, 1, 2) / 255.0
            )
            a_windows.append(
                torch.from_numpy(mfcc[a_off + i*A_WIN : a_off + (i+1)*A_WIN].T)
                     .float().unsqueeze(0)
            )
        if not v_windows:
            scores.append(-1.0)
            continue

        sims = []
        for b in range(0, len(v_windows), BATCH):
            vb = torch.stack(v_windows[b:b+BATCH]).to(device)
            ab = torch.stack(a_windows[b:b+BATCH]).to(device)
            with torch.no_grad():
                out = model(vb, ab)
            sims.extend(out.detach().cpu().numpy().tolist())
        scores.append(float(np.mean(sims)) if sims else -1.0)

    raw = max(scores)
    return (raw + 1.0) / 2.0


# ---- Selection helpers ---------------------------------------------------

def _pending_review_ids(review_status_path: Path, all_ids: pd.Series) -> set:
    """Segment IDs that are NOT yet decided (approved/rejected)."""
    if not review_status_path.exists():
        return set(all_ids)
    with open(review_status_path, encoding="utf-8") as f:
        rs = json.load(f)
    decided = {sid for sid, info in rs.items()
               if info.get("status") in ("approved", "rejected")}
    return set(all_ids) - decided


def _backup_csv(csv_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = csv_path.with_suffix(f".csv.bak.{stamp}.syncnet-backfill")
    shutil.copy2(csv_path, bak)
    return bak


def _atomic_write_csv(df: pd.DataFrame, csv_path: Path) -> None:
    tmp = csv_path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(csv_path)


# ---- Main ----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Backfill syncnet_conf in segments_index.csv")
    ap.add_argument("--csv", default="data/metadata/segments_index.csv")
    ap.add_argument("--review-status", default="data/metadata/review_status.json")
    ap.add_argument("--processed-root", default="data/processed")
    ap.add_argument("--weights", default="models/syncnet_v2.pth")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--video-id", nargs="+", help="Restrict to these video_id values")
    ap.add_argument("--limit", type=int, help="Stop after N segments (smoke test)")
    ap.add_argument("--force", action="store_true",
                    help="Re-score every row, even reviewed ones / already non-zero")
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--epsilon", type=float, default=1e-6,
                    help="Treat |syncnet_conf| <= epsilon as 'unset'")
    args = ap.parse_args()

    csv_path = (_PROJECT_ROOT / args.csv).resolve()
    rs_path = (_PROJECT_ROOT / args.review_status).resolve()
    proc_root = (_PROJECT_ROOT / args.processed_root).resolve()
    weights = (_PROJECT_ROOT / args.weights).resolve()

    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")
    if not weights.exists():
        sys.exit(f"Weights not found: {weights}")

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    logger.info(f"device={device}  csv={csv_path.name}")

    df = pd.read_csv(csv_path)
    if "syncnet_conf" not in df.columns:
        df["syncnet_conf"] = 0.0
    n_total = len(df)
    logger.info(f"loaded {n_total} rows")

    # Build candidate mask
    mask = pd.Series(True, index=df.index)
    if not args.force:
        pending = _pending_review_ids(rs_path, df["segment_id"])
        mask &= df["segment_id"].isin(pending)
        unset = df["syncnet_conf"].abs().fillna(0.0) <= args.epsilon
        mask &= unset
    if args.video_id:
        mask &= df["video_id"].isin(args.video_id)

    todo_idx = df.index[mask].tolist()
    if args.limit:
        todo_idx = todo_idx[: args.limit]

    logger.info(f"candidates: {len(todo_idx)} / {n_total}")
    if not todo_idx:
        logger.info("nothing to do")
        return

    backup = _backup_csv(csv_path)
    logger.info(f"backup: {backup.name}")

    # Load model once
    model = SyncNetModel()
    state = torch.load(weights, map_location=device)
    model.load_state_dict(state)
    model = model.to(device).eval()
    logger.info(f"model loaded ({weights.name})")

    n_done = n_skipped = n_failed = 0
    confs = []
    t_start = time.time()

    for k, idx in enumerate(todo_idx, start=1):
        seg_id = df.at[idx, "segment_id"]
        vid_id = df.at[idx, "video_id"]
        mp4 = proc_root / vid_id / "face_crop" / f"{seg_id}.mp4"

        if not mp4.exists():
            logger.warning(f"[{k}/{len(todo_idx)}] {seg_id} — face_crop missing, skipping")
            n_skipped += 1
            continue

        try:
            frames = _read_face_crop_frames(mp4)
            if len(frames) < V_WIN:
                logger.warning(f"[{k}/{len(todo_idx)}] {seg_id} — too few frames ({len(frames)})")
                df.at[idx, "syncnet_conf"] = 0.0
                n_skipped += 1
                continue
            mfcc = _extract_mfcc(mp4)
            conf = _score_segment(model, device, frames, mfcc)
            df.at[idx, "syncnet_conf"] = float(round(conf, 4))
            confs.append(conf)
            n_done += 1
        except Exception as e:
            logger.warning(f"[{k}/{len(todo_idx)}] {seg_id} FAILED: {e}")
            n_failed += 1
            continue

        if k % args.checkpoint_every == 0:
            _atomic_write_csv(df, csv_path)
            elapsed = time.time() - t_start
            rate = k / elapsed
            eta = (len(todo_idx) - k) / rate if rate > 0 else 0
            logger.info(
                f"[{k}/{len(todo_idx)}] checkpoint  "
                f"rate={rate:.2f}/s  eta={eta/60:.1f}min  "
                f"mean_conf={np.mean(confs):.3f}"
            )

    _atomic_write_csv(df, csv_path)
    elapsed = time.time() - t_start

    logger.info("=" * 60)
    logger.info(f"done: {n_done} scored | {n_skipped} skipped | {n_failed} failed")
    logger.info(f"elapsed: {elapsed/60:.1f} min  ({n_done/max(elapsed,1):.2f} clips/s)")
    if confs:
        arr = np.array(confs)
        logger.info(
            f"conf stats: mean={arr.mean():.3f} med={np.median(arr):.3f} "
            f"p25={np.percentile(arr,25):.3f} p75={np.percentile(arr,75):.3f} "
            f"min={arr.min():.3f} max={arr.max():.3f}"
        )


if __name__ == "__main__":
    main()
