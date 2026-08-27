"""
Auto-tag `environment` + `background_noise` for videos_master.csv.

Phase 1 (this script):
  For every video that is still missing env/noise, download:
    * 1 thumbnail JPG  (used by a human/Claude to decide `environment`)
    * 10s of audio centred on the video midpoint  (used to compute noise metrics)
  Compute per-video audio features (RMS, spectral flatness, voice-band ratio) and
  derive `suggested_noise` from them. Write everything to a proposals CSV that a
  follow-up step can review and then merge back into videos_master.csv.

Phase 2 (separate): a human or Claude inspects the thumbnails and fills
`suggested_environment`, then `apply_proposals.py` merges into videos_master.

Run:
    python backend/tools/auto_tag_conditions.py
    python backend/tools/auto_tag_conditions.py --limit 5         # sanity-check first
    python backend/tools/auto_tag_conditions.py --video-id md_007 md_008
    python backend/tools/auto_tag_conditions.py --resume          # skip rows already in proposals.csv
"""
from __future__ import annotations

import argparse
import io
import shutil
import sys
import tempfile

# Force UTF-8 on Windows stdout/stderr so Romanian/Moldovan titles don't crash logging.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yt_dlp

ROOT = Path(__file__).resolve().parent.parent.parent
CSV_PATH = ROOT / "data" / "catalog" / "videos_master.csv"
WORKDIR = ROOT / "data" / "catalog" / "auto_tag_workdir"
THUMB_DIR = WORKDIR / "thumbnails"
AUDIO_DIR = WORKDIR / "audio_samples"
PROPOSALS_CSV = WORKDIR / "proposals.csv"
LOG_PATH = WORKDIR / "auto_tag.log"

AUDIO_SAMPLE_SECONDS = 10
AUDIO_TARGET_SR = 16000

# Channel-based environment prior. Derived from already-tagged videos in the
# dataset (md_001-006 are all studio). Used as a starting hint only — the
# thumbnail review step is what produces the final `suggested_environment`.
CHANNEL_ENV_PRIOR: dict[str, str] = {
    "Privesc.Eu Moldova": "studio",
    "Moldova.org": "studio",
    "HotNews Romania": "studio",
    "Teleradio Moldova": "studio",
    "Europa FM": "studio",
}


@dataclass
class AudioFeatures:
    rms_db: float           # overall loudness, dBFS
    spectral_flatness: float  # 0 (tonal) .. 1 (white noise)
    voice_band_ratio: float  # energy in 300-3400 Hz / total energy
    suggested_noise: str    # none / low / moderate / high


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def ensure_dirs() -> None:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def download_thumbnail(url: str, video_id: str) -> Path | None:
    """Download the YouTube thumbnail (any size) and save as <video_id>.jpg.
    Returns the path on success, None on failure."""
    out_path = THUMB_DIR / f"{video_id}.jpg"
    if out_path.exists() and out_path.stat().st_size > 1024:
        return out_path

    with tempfile.TemporaryDirectory() as tmp:
        opts = {
            "skip_download": True,
            "writethumbnail": True,
            "outtmpl": str(Path(tmp) / f"{video_id}.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "postprocessors": [
                {"key": "FFmpegThumbnailsConvertor", "format": "jpg"},
            ],
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
        except Exception as e:
            log(f"  thumbnail FAIL {video_id}: {e}")
            return None

        for cand in Path(tmp).iterdir():
            if cand.suffix.lower() in (".jpg", ".jpeg", ".webp", ".png"):
                shutil.copy2(cand, out_path)
                return out_path
    log(f"  thumbnail FAIL {video_id}: no image produced")
    return None


def download_audio_sample(url: str, video_id: str, duration_seconds: float | None) -> Path | None:
    """Download a 10s mono 16kHz WAV centred at the video midpoint.

    Uses yt-dlp's `download_ranges` so only the relevant fragment is fetched
    (avoids YT blocking a bare ffmpeg request against a cookie-signed URL)."""
    out_path = AUDIO_DIR / f"{video_id}.wav"
    if out_path.exists() and out_path.stat().st_size > 1024:
        return out_path

    # Pick the midpoint, clamped to a sensible range.
    if duration_seconds and duration_seconds > AUDIO_SAMPLE_SECONDS * 4:
        start = max(30.0, duration_seconds / 2 - AUDIO_SAMPLE_SECONDS / 2)
    else:
        start = 10.0
    end = start + AUDIO_SAMPLE_SECONDS

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / video_id
        opts = {
            "format": "bestaudio[abr<=64]/bestaudio/best",
            "quiet": True,
            "no_warnings": True,
            "outtmpl": str(tmp_path) + ".%(ext)s",
            "download_ranges": yt_dlp.utils.download_range_func(None, [(start, end)]),
            "force_keyframes_at_cuts": True,
            # Use android/ios player client → unsigned URLs that are not throttled
            # to ~50 B/s by YouTube's SABR rollout. Web client returns throttled URLs.
            "extractor_args": {
                "youtube": {
                    "player_client": ["android", "ios", "web"],
                },
            },
            "concurrent_fragment_downloads": 4,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "0",
                },
            ],
            "postprocessor_args": {
                "extractaudio": ["-ac", "1", "-ar", str(AUDIO_TARGET_SR)],
            },
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
        except Exception as e:
            log(f"  audio FAIL {video_id}: {e}")
            return None

        wav_files = list(Path(tmp).glob(f"{video_id}.wav"))
        if not wav_files:
            wav_files = list(Path(tmp).glob(f"{video_id}.*.wav"))
        if not wav_files:
            log(f"  audio: no wav produced for {video_id}")
            return None
        shutil.copy2(wav_files[0], out_path)

    if not out_path.exists() or out_path.stat().st_size < 1024:
        return None
    return out_path


def compute_audio_features(wav_path: Path) -> AudioFeatures | None:
    """Read a WAV and extract noise-discriminating features.

    The three features chosen:
      - rms_db: overall loudness. Studio recordings are usually normalised loud.
      - spectral_flatness: ratio of geometric to arithmetic mean of the magnitude
        spectrum. Noise-like signals approach 1; tonal speech sits around 0.05-0.15.
      - voice_band_ratio: fraction of energy in 300-3400 Hz (telephone band).
        Speech-dominated clips are >0.6; background-noise-dominated clips drift
        below 0.4 because the noise pushes energy outside the voice band.
    Returns None on read failure."""
    try:
        audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    except Exception as e:
        log(f"  read FAIL {wav_path.name}: {e}")
        return None

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.size == 0:
        return None

    # RMS in dBFS, with a floor to avoid -inf.
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    rms_db = 20.0 * np.log10(max(rms, 1e-6))

    # Magnitude spectrum over the whole clip.
    n = audio.shape[0]
    # Window with hann to reduce spectral leakage.
    window = np.hanning(n).astype(np.float32)
    spec = np.abs(np.fft.rfft(audio * window))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mag2 = spec ** 2 + 1e-12

    # Spectral flatness (computed in linear power domain).
    geo_mean = np.exp(np.mean(np.log(mag2)))
    arith_mean = np.mean(mag2)
    flatness = float(geo_mean / arith_mean) if arith_mean > 0 else 0.0

    # Voice-band energy ratio.
    voice_mask = (freqs >= 300.0) & (freqs <= 3400.0)
    total_e = float(np.sum(mag2))
    voice_e = float(np.sum(mag2[voice_mask]))
    voice_ratio = voice_e / total_e if total_e > 0 else 0.0

    suggested = _classify_noise(rms_db, flatness, voice_ratio)

    return AudioFeatures(
        rms_db=round(rms_db, 2),
        spectral_flatness=round(flatness, 4),
        voice_band_ratio=round(voice_ratio, 4),
        suggested_noise=suggested,
    )


def _classify_noise(rms_db: float, flatness: float, voice_ratio: float) -> str:
    """Map the three features to a noise bucket.

    Calibrated on the assumption that the source pool is mostly Romanian/Moldovan
    interviews. Studio interviews show flatness ~0.02-0.08 and voice_ratio >0.7;
    outdoor or muffled audio drops voice_ratio and raises flatness.

    These thresholds are deliberately conservative — the auto-suggestion can be
    overridden during the apply step."""
    if flatness >= 0.18 or voice_ratio < 0.35:
        return "high"
    if flatness >= 0.10 or voice_ratio < 0.55:
        return "moderate"
    if flatness >= 0.05 or voice_ratio < 0.75:
        return "low"
    return "none"


def load_existing_proposals() -> set[str]:
    if not PROPOSALS_CSV.exists():
        return set()
    try:
        df = pd.read_csv(PROPOSALS_CSV)
        return set(df["video_id"].astype(str).tolist())
    except Exception:
        return set()


def append_proposal(row: dict) -> None:
    df = pd.DataFrame([row])
    header = not PROPOSALS_CSV.exists()
    df.to_csv(PROPOSALS_CSV, mode="a", header=header, index=False, encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="Process only N videos (for testing).")
    p.add_argument("--video-id", nargs="+", default=None, help="Process only specific video IDs.")
    p.add_argument("--resume", action="store_true",
                   help="Skip videos already present in proposals.csv (default: True; pass --no-resume to override).")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.set_defaults(resume=True)
    args = p.parse_args()

    if not CSV_PATH.exists():
        print(f"ERROR: not found: {CSV_PATH}", file=sys.stderr)
        return 1

    ensure_dirs()
    log(f"=== auto_tag_conditions start ===")
    log(f"workdir: {WORKDIR}")

    df = pd.read_csv(CSV_PATH)

    if args.video_id:
        queue = df[df["video_id"].isin(args.video_id)]
    else:
        mask = df["environment"].isna() | df["background_noise"].isna()
        queue = df.loc[mask]

    already = load_existing_proposals() if args.resume else set()
    if already:
        queue = queue[~queue["video_id"].astype(str).isin(already)]
        log(f"resume: {len(already)} already processed, skipping")

    if args.limit:
        queue = queue.head(args.limit)

    if queue.empty:
        log("nothing to do.")
        return 0

    log(f"queue: {len(queue)} videos")

    ok = fail = 0
    for i, (_, row) in enumerate(queue.iterrows(), start=1):
        vid = str(row["video_id"])
        url = str(row.get("youtube_url", "") or "")
        title = str(row.get("title", "") or "")[:140]
        channel = str(row.get("source_channel", "") or "")
        duration = row.get("duration_seconds")
        try:
            duration = float(duration) if pd.notna(duration) else None
        except (TypeError, ValueError):
            duration = None

        log(f"[{i}/{len(queue)}] {vid} - {channel} - {title}")

        if not url:
            log("  no url, skipping")
            fail += 1
            continue

        thumb = download_thumbnail(url, vid)
        audio = download_audio_sample(url, vid, duration)
        feats = compute_audio_features(audio) if audio else None

        proposal = {
            "video_id": vid,
            "youtube_url": url,
            "source_channel": channel,
            "title": title,
            "thumbnail_path": str(thumb.relative_to(ROOT)) if thumb else "",
            "audio_path": str(audio.relative_to(ROOT)) if audio else "",
            "rms_db": feats.rms_db if feats else None,
            "spectral_flatness": feats.spectral_flatness if feats else None,
            "voice_band_ratio": feats.voice_band_ratio if feats else None,
            "suggested_noise": feats.suggested_noise if feats else "",
            "channel_env_prior": CHANNEL_ENV_PRIOR.get(channel, ""),
            "suggested_environment": "",  # filled in by review step
        }
        append_proposal(proposal)

        if thumb and feats:
            ok += 1
            log(f"  OK  flatness={feats.spectral_flatness:.3f} "
                f"voice_ratio={feats.voice_band_ratio:.3f} "
                f"→ noise={feats.suggested_noise}")
        else:
            fail += 1
            log(f"  PARTIAL thumb={'Y' if thumb else 'N'} audio={'Y' if feats else 'N'}")

    log(f"=== done: {ok} ok, {fail} partial/failed, proposals at {PROPOSALS_CSV} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
