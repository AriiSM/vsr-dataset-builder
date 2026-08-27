#!/usr/bin/env python3
"""
Romanian VSR Dataset Pipeline v2 - CLI Entry Point

Usage:
    # Initialize project structure
    python backend/orchestrator/cli.py init ./data

    # Process single video
    python backend/orchestrator/cli.py single ro_001 "https://youtube.com/watch?v=..."

    # Process batch from Excel
    python backend/orchestrator/cli.py batch data/catalog/videos_master.csv --limit 10

    # Show stats
    python backend/orchestrator/cli.py stats data/catalog/videos_master.csv
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional

# ── Redirect all model caches to models/ BEFORE any ML imports ──────────────
# This ensures Silero VAD, WhisperX, and HuggingFace alignment models are
# stored inside the project rather than scattered across the user profile.
project_root = Path(__file__).resolve().parent.parent.parent
_models_dir = project_root / "models"
_models_dir.mkdir(exist_ok=True)
os.environ.setdefault("TORCH_HOME",        str(_models_dir / "torch"))
os.environ.setdefault("HF_HOME",           str(_models_dir / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_models_dir / "huggingface" / "hub"))

warnings.filterwarnings("ignore", message=".*pretrained.*deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*weights.*deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*old format.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torchcodec.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*load_with_torchcodec.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*upgrade_checkpoint.*", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")

sys.path[:0] = [str(project_root / "backend"), str(project_root / "backend" / "shared")]

# Add TalkNet-ASD to path if present (project root or models/)
for _talknet_candidate in [project_root / "TalkNet-ASD", _models_dir / "TalkNet-ASD"]:
    if _talknet_candidate.exists():
        sys.path.insert(0, str(_talknet_candidate))
        break

# Cache redirection BEFORE any ML import — every model lands under models/.
from vsr_shared.model_env import apply_model_env
apply_model_env(project_root / "models")

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
logger.add(
    project_root / "logs" / f"pipeline_{datetime.now():%Y%m%d}.log",
    level="DEBUG",
    rotation="100 MB",
)


def _apply_cookie_overrides(pipeline, args):
    """Push --cookies / --cookies-from-browser CLI flags into the pipeline config."""
    if getattr(args, "cookies_from_browser", None):
        pipeline.config.cookies_from_browser = args.cookies_from_browser
        pipeline.services.invalidate_downloader()  # force re-init with new settings
    if getattr(args, "cookies", None):
        from pathlib import Path as _Path
        pipeline.config.cookies_file = _Path(args.cookies)
        pipeline.services.invalidate_downloader()


def cmd_single(args):
    from orchestrator.pipeline import VSRPipeline

    pipeline = VSRPipeline.from_config(args.config)
    _apply_cookie_overrides(pipeline, args)
    result = pipeline.process_video(
        args.video_id,
        args.url,
        skip_download=args.skip_download,
        verify_cc=not args.no_cc_check,
    )

    print(f"\n{'='*50}")
    print(f"Video:            {result.video_id}")
    print(f"Status:           {result.status.value}")
    print(f"Segments:         {len(result.segments)}")
    print(f"Total Duration:   {result.total_duration:.1f}s")
    print(f"Processing Time:  {result.processing_time:.1f}s")

    if result.error_message:
        print(f"Error: {result.error_message}")

    if result.segments:
        print(f"\nExported segments:")
        for seg in result.segments[:5]:
            print(f"  {seg.segment_id}: {seg.text[:60]}")
        if len(result.segments) > 5:
            print(f"  ... and {len(result.segments) - 5} more")


def cmd_batch(args):
    from orchestrator.pipeline import VSRPipeline
    from vsr_shared.excel_schema import ProcessingStatus

    excel_path = Path(args.excel)
    if not excel_path.exists():
        print(f"Error: Excel file not found: {excel_path}")
        sys.exit(1)

    pipeline = VSRPipeline.from_config(args.config)
    _apply_cookie_overrides(pipeline, args)
    results = pipeline.process_batch(
        excel_path,
        status_filter=args.status,
        limit=args.limit,
        video_ids=args.video_id or None,
    )

    completed = sum(1 for r in results if r.status == ProcessingStatus.COMPLETED)
    failed = sum(1 for r in results if r.status == ProcessingStatus.FAILED)
    total_segs = sum(len(r.segments) for r in results)
    total_dur = sum(r.total_duration for r in results)
    total_time = sum(r.processing_time for r in results)

    print(f"\n{'='*50}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*50}")
    print(f"Videos processed: {len(results)}")
    print(f"  Completed: {completed}")
    print(f"  Failed:    {failed}")
    print(f"Total segments:   {total_segs}")
    print(f"Total duration:   {total_dur/60:.1f} min ({total_dur/3600:.2f} h)")
    print(f"Processing time:  {total_time/60:.1f} min")

    if failed:
        print(f"\nFailed videos:")
        for r in results:
            if r.status == ProcessingStatus.FAILED:
                print(f"  {r.video_id}: {r.error_message}")


def cmd_resume(args):
    """Resume a single interrupted video."""
    import pandas as pd
    from orchestrator.pipeline import VSRPipeline

    # Resolve URL: prefer CLI argument, fall back to Excel
    url = getattr(args, "url", None) or ""
    if not url:
        import yaml
        config_path = Path(args.config)
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        excel_path = (
            Path(cfg["paths"]["base_dir"]) / "catalog" / "videos_master.csv"
        )
        if excel_path.exists():
            df = pd.read_csv(excel_path)
            row = df[df["video_id"].astype(str) == args.video_id]
            if not row.empty:
                url = str(row.iloc[0].get("youtube_url", ""))
    if not url:
        print(f"Error: could not find a URL for '{args.video_id}' in the master CSV.")
        sys.exit(1)

    pipeline = VSRPipeline.from_config(args.config)
    result = pipeline.resume_video(args.video_id, url)

    print(f"\n{'='*50}")
    print(f"Video:            {result.video_id}")
    print(f"Status:           {result.status.value}")
    print(f"Segments:         {len(result.segments)}")
    print(f"Total Duration:   {result.total_duration:.1f}s")
    print(f"Processing Time:  {result.processing_time:.1f}s")
    if result.error_message:
        print(f"Error:            {result.error_message}")


def cmd_resume_batch(args):
    """Resume all interrupted videos that still have clips or partial exports."""
    from orchestrator.pipeline import VSRPipeline

    excel_path = Path(args.excel)
    if not excel_path.exists():
        print(f"Error: Excel file not found: {excel_path}")
        sys.exit(1)

    pipeline = VSRPipeline.from_config(args.config)
    results = pipeline.process_batch_resume(
        excel_path,
        limit=args.limit,
        video_ids=args.video_id or None,
    )

    completed = sum(1 for r in results if r.status.value == "completed")
    failed    = sum(1 for r in results if r.status.value == "failed")
    total_segs = sum(len(r.segments) for r in results)
    total_dur  = sum(r.total_duration for r in results)

    print(f"\n{'='*50}")
    print(f"RESUME BATCH COMPLETE")
    print(f"{'='*50}")
    print(f"Videos resumed:   {len(results)}")
    print(f"  Completed:      {completed}")
    print(f"  Failed:         {failed}")
    print(f"Total segments:   {total_segs}")
    print(f"Total duration:   {total_dur/60:.1f} min")

    if failed:
        print("\nFailed videos:")
        for r in results:
            if r.status.value == "failed":
                print(f"  {r.video_id}: {r.error_message}")


def cmd_sync_excel(args):
    """Rebuild Excel stats from segments on disk (no reprocessing)."""
    from orchestrator.pipeline import VSRPipeline

    excel_path = Path(args.excel)
    if not excel_path.exists():
        print(f"Error: Excel file not found: {excel_path}")
        sys.exit(1)

    pipeline = VSRPipeline.from_config(args.config)
    updated = pipeline.sync_excel_from_disk(
        excel_path,
        video_ids=args.video_id or None,
    )

    print(f"\n{'='*50}")
    if updated:
        print(f"Updated {updated} video(s) in the Excel.")
    else:
        print("Nothing to update - all matching rows already have stats, "
              "or no segment files were found.")


def cmd_stats(args):
    """Extended dataset statistics — text or --json output.

    Sections:
      1. Videos by status (counts + %)
      2. Per region (RO / MD / DIASPORA): videos, segments, duration
      3. Per speaker (top 10): name, segments, duration
      4. Quality: distributions (min/mean/median/p95/max) for asd, whisper_conf,
         wer, duration, num_words
      5. Top vocabulary: top 50 words with count + total spoken duration
      6. Dataset health: segments missing speaker_id / WER / Conf=1 (counts + %)
      7. Train/val/test: split distribution when populated
    """
    import json
    import pandas as pd
    import yaml

    excel_path = Path(args.excel)
    if not excel_path.exists():
        print(f"Error: CSV file not found: {excel_path}")
        sys.exit(1)

    df = pd.read_csv(excel_path)

    config_path = Path(args.config)
    base_dir = Path(".")
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        base_dir = Path(cfg["paths"]["base_dir"])

    metadata_dir = base_dir / "catalog"
    seg_path = metadata_dir / "segments_index.csv"
    speakers_path = metadata_dir / "speakers_registry.csv"

    seg_df = pd.read_csv(seg_path) if seg_path.exists() else pd.DataFrame()
    speakers_df = pd.read_csv(speakers_path) if speakers_path.exists() else pd.DataFrame()

    # ── Section 1: Videos by status ────────────────────────────────────
    n_videos = len(df)
    by_status: dict = {}
    if "status" in df.columns:
        vc = df["status"].value_counts()
        by_status = {
            str(k): {
                "count": int(v),
                "pct": round(100.0 * v / n_videos, 1) if n_videos else 0.0,
            }
            for k, v in vc.items()
        }

    # ── Section 2: Per region ──────────────────────────────────────────
    per_region: dict = {}
    if "region" in df.columns:
        for reg, sub in df.groupby(df["region"].fillna("UNKNOWN")):
            vids = sub["video_id"].astype(str).tolist()
            segs = seg_df[seg_df["video_id"].astype(str).isin(vids)] if not seg_df.empty else pd.DataFrame()
            per_region[str(reg)] = {
                "n_videos": int(len(sub)),
                "n_segments": int(len(segs)),
                "total_duration_h": (
                    round(float(segs["duration"].sum()) / 3600, 3)
                    if "duration" in segs.columns and not segs.empty else 0.0
                ),
            }

    # ── Section 3: Per speaker (top 10 by # segments) ──────────────────
    per_speaker_top: list = []
    if not seg_df.empty and "speaker_id" in seg_df.columns:
        grp = seg_df.groupby(seg_df["speaker_id"].fillna("(unknown)"))
        rows = []
        for sid, sub in grp:
            row = {
                "speaker_id": str(sid),
                "n_segments": int(len(sub)),
                "duration_s": round(float(sub["duration"].fillna(0).sum()), 2)
                              if "duration" in sub.columns else 0.0,
            }
            if not speakers_df.empty:
                meta = speakers_df[speakers_df["speaker_id"].astype(str) == str(sid)]
                if not meta.empty:
                    nm = meta.iloc[0].get("speaker_name", "")
                    row["speaker_name"] = "" if pd.isna(nm) else str(nm)
            rows.append(row)
        rows.sort(key=lambda r: r["n_segments"], reverse=True)
        per_speaker_top = rows[:10]

    # ── Section 4: Quality distributions ───────────────────────────────
    def _dist(series: pd.Series) -> Optional[dict]:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return None
        return {
            "n": int(len(s)),
            "min": round(float(s.min()), 3),
            "mean": round(float(s.mean()), 3),
            "median": round(float(s.median()), 3),
            "p95": round(float(s.quantile(0.95)), 3),
            "max": round(float(s.max()), 3),
        }

    quality: dict = {}
    if not seg_df.empty:
        for col in ("asd_score", "whisper_conf", "wer", "duration", "num_words"):
            if col in seg_df.columns:
                d = _dist(seg_df[col])
                if d is not None:
                    quality[col] = d

    # ── Section 5: Top vocabulary ──────────────────────────────────────
    top_vocab: list = []
    if not seg_df.empty and "text" in seg_df.columns:
        word_durs: dict = {}  # word -> [count, total_dur]
        for _, row in seg_df.iterrows():
            text = str(row.get("text", "") or "")
            seg_dur = float(row.get("duration", 0) or 0)
            tokens = [w.strip(".,;:!?\"'()[]").upper() for w in text.split()]
            tokens = [t for t in tokens if t]
            if not tokens:
                continue
            per_word_dur = seg_dur / len(tokens)
            for t in tokens:
                rec = word_durs.setdefault(t, [0, 0.0])
                rec[0] += 1
                rec[1] += per_word_dur
        top_vocab = sorted(
            ({"word": w, "count": c, "duration_s": round(d, 2)} for w, (c, d) in word_durs.items()),
            key=lambda r: r["count"],
            reverse=True,
        )[:50]

    # ── Section 6: Dataset health ──────────────────────────────────────
    health: dict = {}
    if not seg_df.empty:
        n = len(seg_df)
        def _pct(v: int) -> float:
            return round(100.0 * v / n, 1) if n else 0.0
        missing_speaker = (
            int(seg_df["speaker_id"].isna().sum() + (seg_df["speaker_id"].astype(str) == "").sum())
            if "speaker_id" in seg_df.columns else n
        )
        missing_wer = (
            int(seg_df["wer"].isna().sum())
            if "wer" in seg_df.columns else n
        )
        # Conf=1 must be counted from annotations because the CSV doesn't
        # store the level directly. Skip if annotation_path missing.
        conf1_count = 0
        anno_total = 0
        if "annotation_path" in seg_df.columns:
            for p in seg_df["annotation_path"].dropna().astype(str):
                anno_path = (project_root / p) if not Path(p).is_absolute() else Path(p)
                if not anno_path.exists():
                    continue
                anno_total += 1
                for line in anno_path.read_text(encoding="utf-8").splitlines():
                    if line.startswith("Conf:"):
                        try:
                            if int(line[len("Conf:"):].strip()) == 1:
                                conf1_count += 1
                        except ValueError:
                            pass
                        break
        health = {
            "n_segments": n,
            "missing_speaker_id": {"count": missing_speaker, "pct": _pct(missing_speaker)},
            "missing_wer": {"count": missing_wer, "pct": _pct(missing_wer)},
            "conf_1_low": {
                "count": conf1_count,
                "annotations_scanned": anno_total,
                "pct_of_scanned": round(100.0 * conf1_count / anno_total, 1) if anno_total else 0.0,
            },
        }

    # ── Section 7: Train/val/test split ────────────────────────────────
    splits: dict = {}
    if not seg_df.empty and "split" in seg_df.columns:
        sub = seg_df["split"].dropna().astype(str)
        sub = sub[sub != ""]
        if not sub.empty:
            splits = {str(k): int(v) for k, v in sub.value_counts().items()}

    payload = {
        "videos": {
            "total": n_videos,
            "by_status": by_status,
        },
        "per_region": per_region,
        "per_speaker_top10": per_speaker_top,
        "quality": quality,
        "top_vocabulary": top_vocab,
        "health": health,
        "split": splits,
    }

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    # ── Pretty text output ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"DATASET STATISTICS")
    print(f"{'='*60}")

    print(f"\nVideos: {n_videos}")
    for st, info in by_status.items():
        print(f"  {st:12s} {info['count']:>4}  ({info['pct']:>4.1f}%)")

    if per_region:
        print(f"\nPer region:")
        print(f"  {'region':12s} {'videos':>6} {'segments':>9} {'duration_h':>11}")
        for reg, info in per_region.items():
            print(f"  {reg:12s} {info['n_videos']:>6} {info['n_segments']:>9} {info['total_duration_h']:>11.3f}")

    if per_speaker_top:
        print(f"\nTop 10 speakers:")
        print(f"  {'speaker_id':30s} {'name':25s} {'segs':>5} {'dur(s)':>9}")
        for r in per_speaker_top:
            name = r.get("speaker_name", "") or "(unnamed)"
            print(f"  {r['speaker_id'][:30]:30s} {name[:25]:25s} {r['n_segments']:>5} {r['duration_s']:>9.1f}")

    if quality:
        print(f"\nQuality distributions:")
        print(f"  {'metric':16s} {'n':>6} {'min':>8} {'mean':>8} {'median':>8} {'p95':>8} {'max':>8}")
        for metric, d in quality.items():
            print(f"  {metric:16s} {d['n']:>6} {d['min']:>8.3f} {d['mean']:>8.3f} {d['median']:>8.3f} {d['p95']:>8.3f} {d['max']:>8.3f}")

    if top_vocab:
        print(f"\nTop 20 vocabulary (of {len(top_vocab)} shown):")
        for w in top_vocab[:20]:
            print(f"  {w['word']:25s} {w['count']:>5}  ({w['duration_s']:>6.1f}s)")

    if health:
        print(f"\nDataset health (n={health['n_segments']}):")
        for k in ("missing_speaker_id", "missing_wer"):
            v = health[k]
            print(f"  {k:25s} {v['count']:>5}  ({v['pct']:>5.1f}%)")
        c1 = health["conf_1_low"]
        print(f"  {'conf_1_low':25s} {c1['count']:>5}  ({c1['pct_of_scanned']:>5.1f}% of {c1['annotations_scanned']} scanned)")

    if splits:
        print(f"\nSplit distribution:")
        for k, v in splits.items():
            print(f"  {k:8s} {v}")


def cmd_bulk_import(args):
    """Download a list of YouTube URLs and add rows to videos_master.csv.

    For each URL:
      - Assigns next video_id using the given prefix
      - Downloads via yt-dlp (honors CC check / cookies from CLI flags)
      - On success: appends row with status='pending', filled metadata
      - On failure: appends row with status='failed' and error_message
    """
    import re
    import pandas as pd
    from orchestrator.pipeline import VSRPipeline
    from services.downloader.youtube_downloader import YouTubeDownloader
    from vsr_shared.excel_schema import ProcessingStatus, VIDEOS_MASTER_SCHEMA

    # Read URLs
    urls: list = []
    if args.urls:
        urls.extend(args.urls)
    if args.from_file:
        txt = Path(args.from_file).read_text(encoding="utf-8")
        for line in txt.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    urls = [u.strip() for u in urls if u and u.strip()]
    if not urls:
        print("No URLs provided. Pass --urls ... or --from-file <path>.")
        sys.exit(1)

    csv_path = Path(args.excel)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path} — run `init` first.")
        sys.exit(1)

    prefix = args.prefix or "vid"
    region = args.region or "UNKNOWN"
    source = args.source or "YouTube_CC"
    verify_cc = not args.no_cc_check

    # Load pipeline just to get a ready-to-use downloader with correct config
    pipeline = VSRPipeline.from_config(args.config)
    _apply_cookie_overrides(pipeline, args)
    downloader: YouTubeDownloader = pipeline.services.downloader

    df = pd.read_csv(csv_path)
    existing_ids = set(df["video_id"].astype(str).tolist()) if "video_id" in df.columns else set()
    existing_urls = set(df["youtube_url"].astype(str).tolist()) if "youtube_url" in df.columns else set()

    # Build a set of YouTube video IDs already in the CSV so we can dedup
    # across URL variants (watch?v=, youtu.be/, with-or-without-www,
    # extra &t=XX query params, playlist params, etc.).
    existing_yt_ids: set = set()
    for u in existing_urls:
        try:
            existing_yt_ids.add(downloader._extract_youtube_id(u))
        except Exception:
            # Row's URL isn't parseable; fall back to exact-string matching
            pass

    # Determine next numeric suffix for the prefix
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    used_nums = []
    for vid in existing_ids:
        m = pat.match(str(vid))
        if m:
            used_nums.append(int(m.group(1)))
    next_num = (max(used_nums) + 1) if used_nums else 1

    total = len(urls)
    added, failed, skipped = 0, 0, 0

    # Track IDs we imported in this run so duplicate URLs inside the same
    # input list are also caught.
    imported_yt_ids: set = set()

    for i, url in enumerate(urls, 1):
        logger.info(f"[{i}/{total}] Importing: {url}")

        # Normalize: extract the 11-char YouTube ID. Any URL variant that
        # points to the same video will yield the same ID → reliable dedup.
        try:
            yt_id = downloader._extract_youtube_id(url)
        except Exception as e:
            logger.error(f"  Invalid URL ({e}): {url} — skipped")
            failed += 1
            continue

        if yt_id in existing_yt_ids or yt_id in imported_yt_ids:
            logger.warning(f"  Already in CSV (same YouTube ID {yt_id}): skipped")
            skipped += 1
            continue
        if url in existing_urls:
            # Fallback exact-URL check in case the CSV row had an unparseable URL
            logger.warning(f"  Already in CSV (exact URL match): skipped")
            skipped += 1
            continue

        video_id = f"{prefix}_{next_num:03d}"
        while video_id in existing_ids:
            next_num += 1
            video_id = f"{prefix}_{next_num:03d}"

        row = {col: "" for col in df.columns}
        for col in VIDEOS_MASTER_SCHEMA.keys():
            if col not in row:
                row[col] = ""
        row["video_id"] = video_id
        row["youtube_url"] = url
        row["region"] = region
        row["source"] = source
        row["license"] = "unverified"

        # ── Fetch metadata (title / duration / license / channel) ──────────
        info_obj = None
        try:
            info_obj = downloader.get_video_info(url)
            row["title"] = (info_obj.title or "")[:500]
            row["source_channel"] = info_obj.channel or ""
            row["duration_seconds"] = float(info_obj.duration) if info_obj.duration else ""
            # Log the raw license string so "unverified" verdicts are traceable
            logger.info(f"  raw license: {info_obj.license!r}")
            if info_obj.is_creative_commons:
                row["license"] = "CC-BY"
        except Exception as e:
            logger.warning(f"  metadata fetch failed: {e}")

        # ── Download — `verify_cc=True` also runs CC check, which succeeds
        # only for CC videos. A successful download under verify_cc=True
        # therefore *guarantees* CC-BY even if our upfront metadata call failed.
        try:
            video_path = downloader.download(url, video_id, verify_cc=verify_cc)
            if video_path is None:
                row["status"] = ProcessingStatus.FAILED.value
                row["error_message"] = "download returned no path (CC check failed?)"
                failed += 1
                logger.error(f"  {video_id} download returned None")
            else:
                row["status"] = ProcessingStatus.PENDING.value
                # If the download was gated on verify_cc=True and succeeded,
                # the video IS Creative Commons — update the license column.
                if verify_cc and row["license"] == "unverified":
                    row["license"] = "CC-BY"
                    logger.info(f"  license set to CC-BY (verified during download)")
                added += 1
                logger.info(f"  {video_id} downloaded OK → marked pending")
        except Exception as e:
            row["status"] = ProcessingStatus.FAILED.value
            row["error_message"] = str(e)[:500]
            failed += 1
            logger.error(f"  {video_id} download error: {e}")

        # Append row to CSV immediately so the UI can reflect progress in real time
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(csv_path, index=False)
        existing_ids.add(video_id)
        existing_urls.add(url)
        imported_yt_ids.add(yt_id)
        next_num += 1

    print(f"\n{'='*50}")
    print(f"Bulk import summary")
    print(f"  URLs processed : {total}")
    print(f"  Downloaded     : {added}")
    print(f"  Failed         : {failed}")
    print(f"  Skipped (dup.) : {skipped}")
    print(f"  CSV updated    : {csv_path}")


def cmd_backfill_speakers(args):
    """Populate speaker_id={video_id}_spk0 for legacy segments + refresh registry.

    Older segments (exported before speaker_id became part of the pipeline)
    have an empty speaker_id in segments_index.csv. Without it, the speakers
    registry never gets aggregates and the Stats panel stays empty.

    This command:
      1. Reads segments_index.csv.
      2. For every row with a missing/empty speaker_id, fills in a default
         "{video_id}_spk0" (single-speaker assumption). Rows that already
         have a speaker_id are left untouched (idempotent).
      3. Mirrors the speaker_id into videos_master.csv so the master table
         stays consistent with the segments index.
      4. Calls ensure_speaker_exists() so each speaker has a stub row in
         the registry (curator can later fill in name/gender/age/accent).
      5. Runs recompute_aggregates() so the Stats tab shows real numbers.

    Use --video-id to limit scope; --dry-run to preview without writing.
    """
    import pandas as pd
    import yaml
    from vsr_shared.speakers_registry import ensure_speaker_exists, recompute_aggregates

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config not found: {config_path}")
        sys.exit(1)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    metadata_dir = Path(cfg["paths"]["base_dir"]) / "catalog"

    seg_csv = metadata_dir / "segments_index.csv"
    master_csv = metadata_dir / "videos_master.csv"
    if not seg_csv.exists():
        print(f"Error: {seg_csv} not found")
        sys.exit(1)

    # ── 1+2: scan segments_index, fill missing speaker_id ──────────────
    seg_df = pd.read_csv(seg_csv)
    if "speaker_id" not in seg_df.columns:
        seg_df["speaker_id"] = ""
    seg_df["speaker_id"] = seg_df["speaker_id"].fillna("").astype(str)

    target_mask = (seg_df["speaker_id"] == "") | (seg_df["speaker_id"].str.lower() == "nan")
    if args.video_id:
        target_mask &= seg_df["video_id"].astype(str).isin(args.video_id)

    n_target = int(target_mask.sum())
    n_already = int(len(seg_df) - target_mask.sum())

    if n_target == 0:
        print(f"All {len(seg_df)} segments already have speaker_id — nothing to backfill.")
        # Still refresh aggregates: maybe the registry itself is stale.
        if not args.dry_run:
            recompute_aggregates(metadata_dir)
        return

    # Compute the assigned id per row: "{video_id}_spk0".
    new_ids = seg_df.loc[target_mask, "video_id"].astype(str) + "_spk0"
    speakers_to_create = sorted(set(new_ids.tolist()))

    print(f"Backfill plan:")
    print(f"  segments to update: {n_target}  (already populated: {n_already})")
    print(f"  unique speakers to create/touch: {len(speakers_to_create)}")
    if args.video_id:
        print(f"  scope limited to video_id(s): {args.video_id}")

    if args.dry_run:
        print("\n(--dry-run: no files written)")
        # Show first 5 sample mappings.
        for sid in speakers_to_create[:5]:
            print(f"    will create stub: {sid}")
        if len(speakers_to_create) > 5:
            print(f"    ... and {len(speakers_to_create) - 5} more")
        return

    # ── Write segments_index ───────────────────────────────────────────
    seg_df.loc[target_mask, "speaker_id"] = new_ids
    seg_df.to_csv(seg_csv, index=False)
    print(f"Updated {n_target} rows in {seg_csv.name}")

    # ── 3: mirror into videos_master ───────────────────────────────────
    if master_csv.exists():
        try:
            mdf = pd.read_csv(master_csv)
            for col in mdf.select_dtypes(include=["float64", "object"]).columns:
                mdf[col] = mdf[col].astype(object)
            if "speaker_id" not in mdf.columns:
                mdf["speaker_id"] = ""
            mdf["speaker_id"] = mdf["speaker_id"].fillna("").astype(str)

            # For every video in scope, set its master speaker_id if empty.
            for vid in seg_df.loc[target_mask, "video_id"].astype(str).unique():
                m_mask = mdf["video_id"].astype(str) == vid
                if m_mask.any():
                    cur = str(mdf.loc[m_mask, "speaker_id"].iloc[0]).strip()
                    if cur == "" or cur.lower() == "nan":
                        mdf.loc[m_mask, "speaker_id"] = f"{vid}_spk0"
            mdf.to_csv(master_csv, index=False)
            print(f"Mirrored speaker_id into {master_csv.name}")
        except PermissionError:
            print(f"WARNING: {master_csv.name} is open — skipped master mirror.")

    # ── 4+5: registry stubs + aggregates ───────────────────────────────
    for sid in speakers_to_create:
        ensure_speaker_exists(metadata_dir, sid)
    print(f"Ensured {len(speakers_to_create)} speaker(s) in registry")

    n_updated = recompute_aggregates(metadata_dir)
    print(f"Aggregated {n_updated} speaker row(s) in registry")
    print(f"\nDone. Refresh the Stats tab to see the speakers panel populated.")


def cmd_init(args):
    from vsr_shared.excel_schema import create_empty_videos_master, create_empty_segments_index

    base_dir = Path(args.base_dir)

    dirs = [
        base_dir / "raw",
        base_dir / "clips",
        base_dir / "processed",
        base_dir / "catalog",
        base_dir / "catalog" / "exports",
        base_dir / "catalog" / "backups",
        base_dir / "logs",
        project_root / "models",
        project_root / "temp",
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"Created: {d}")

    # Storage v2: the catalog database is the source of truth.
    from vsr_shared.catalog_db import CatalogDatabase
    CatalogDatabase(base_dir / "catalog" / "dataset.db").close()
    print(f"Created: {base_dir / 'catalog' / 'dataset.db'} (schema v2)")

    excel_path = base_dir / "catalog" / "videos_master.csv"
    if not excel_path.exists() or args.force:
        create_empty_videos_master(excel_path)
        print(f"Created: {excel_path}")
    else:
        print(f"Exists:  {excel_path} (use --force to overwrite)")

    segments_path = base_dir / "catalog" / "segments_index.csv"
    if not segments_path.exists() or args.force:
        create_empty_segments_index(segments_path)
        print(f"Created: {segments_path}")

    print(f"\nProject initialized at: {base_dir}")
    print(f"Next steps:")
    print(f"  1. Add videos to: {excel_path}")
    print(f"  2. Run: python backend/orchestrator/cli.py batch {excel_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Romanian VSR Dataset Pipeline v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backend/orchestrator/cli.py init ./data
  python backend/orchestrator/cli.py single ro_001 "https://youtube.com/watch?v=..."
  python backend/orchestrator/cli.py batch data/catalog/videos_master.csv
  python backend/orchestrator/cli.py stats data/catalog/videos_master.csv
        """,
    )

    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    subs = parser.add_subparsers(dest="command")

    # init
    p_init = subs.add_parser("init", help="Initialize project structure")
    p_init.add_argument("base_dir", help="Base data directory")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing files")

    # single
    p_single = subs.add_parser("single", help="Process one video")
    p_single.add_argument("video_id", help="Unique video identifier")
    p_single.add_argument("url", help="YouTube URL")
    p_single.add_argument("--skip-download", action="store_true")
    p_single.add_argument("--no-cc-check", action="store_true")
    p_single.add_argument(
        "--cookies-from-browser",
        metavar="BROWSER",
        help="Pass cookies from a browser (chrome, firefox, edge, …)",
    )
    p_single.add_argument(
        "--cookies",
        metavar="FILE",
        help="Path to a Netscape-format cookies.txt exported from your browser",
    )

    # batch
    p_batch = subs.add_parser("batch", help="Process batch from Excel")
    p_batch.add_argument("excel", help="Path to videos_master.csv")
    p_batch.add_argument("--limit", type=int)
    p_batch.add_argument("--status", nargs="+", default=["pending"])
    p_batch.add_argument(
        "--video-id",
        nargs="+",
        metavar="ID",
        help="Process only these video IDs (ignores --status when set)",
    )
    p_batch.add_argument(
        "--cookies-from-browser",
        metavar="BROWSER",
        help="Pass cookies from a browser (chrome, firefox, edge, …)",
    )
    p_batch.add_argument(
        "--cookies",
        metavar="FILE",
        help="Path to a Netscape-format cookies.txt",
    )

    # stats
    p_stats = subs.add_parser("stats", help="Show dataset statistics")
    p_stats.add_argument("excel", help="Path to videos_master.csv")
    p_stats.add_argument("--json", action="store_true",
                         help="Emit machine-readable JSON instead of pretty text.")

    # resume (single video)
    p_resume = subs.add_parser(
        "resume",
        help="Resume a single interrupted video",
    )
    p_resume.add_argument("video_id", help="Video identifier (e.g. ro_003)")
    p_resume.add_argument(
        "url", nargs="?", default="",
        help="YouTube URL - looked up from the Excel automatically if omitted",
    )

    # resume-batch (all interrupted videos)
    p_resume_batch = subs.add_parser(
        "resume-batch",
        help="Resume all interrupted videos (status=processing or failed)",
    )
    p_resume_batch.add_argument("excel", help="Path to videos_master.csv")
    p_resume_batch.add_argument("--limit", type=int, help="Max videos to resume")
    p_resume_batch.add_argument(
        "--video-id",
        nargs="+",
        metavar="ID",
        help="Resume only these video IDs (bypasses the status filter)",
    )

    # bulk-import — download a list of YouTube URLs and seed videos_master.csv
    p_bulk = subs.add_parser(
        "bulk-import",
        help="Download many YouTube URLs and add rows to videos_master.csv",
    )
    p_bulk.add_argument("excel", help="Path to videos_master.csv")
    p_bulk.add_argument(
        "--urls", nargs="+", metavar="URL",
        help="YouTube URLs to import (one or more)",
    )
    p_bulk.add_argument(
        "--from-file", metavar="PATH",
        help="Text file with one URL per line (lines starting with # are comments)",
    )
    p_bulk.add_argument("--prefix", default="vid",
                        help="Video ID prefix (default: vid). IDs will be prefix_001, prefix_002, …")
    p_bulk.add_argument("--region", default="UNKNOWN",
                        choices=["RO", "MD", "DIASPORA", "UNKNOWN"],
                        help="Speaker region column (default: UNKNOWN)")
    p_bulk.add_argument("--source", default="YouTube_CC",
                        choices=["TEDx", "YouTube_CC", "Interview", "Lecture",
                                 "Podcast", "News", "Other"],
                        help="Source category (default: YouTube_CC)")
    p_bulk.add_argument("--no-cc-check", action="store_true",
                        help="Skip Creative-Commons license verification")
    p_bulk.add_argument(
        "--cookies-from-browser", metavar="BROWSER",
        help="Pass cookies from a browser (chrome, firefox, edge, …)",
    )
    p_bulk.add_argument(
        "--cookies", metavar="FILE",
        help="Path to a Netscape-format cookies.txt",
    )

    # sync-excel
    p_sync = subs.add_parser(
        "sync-excel",
        help="Rebuild Excel stats from segment files on disk (no reprocessing)",
    )
    p_sync.add_argument("excel", help="Path to videos_master.csv")
    p_sync.add_argument(
        "--video-id",
        nargs="+",
        metavar="ID",
        help="Only sync these video IDs (default: all rows missing stats)",
    )

    # backfill-speakers
    p_bsp = subs.add_parser(
        "backfill-speakers",
        help="Populate speaker_id for legacy segments and refresh the speakers registry",
    )
    p_bsp.add_argument(
        "--video-id",
        nargs="+",
        metavar="ID",
        help="Only backfill these video IDs (default: every segment with empty speaker_id).",
    )
    p_bsp.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing any files.",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    (project_root / "logs").mkdir(exist_ok=True)

    if args.command == "init":
        cmd_init(args)
    elif args.command == "single":
        cmd_single(args)
    elif args.command == "batch":
        cmd_batch(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "sync-excel":
        cmd_sync_excel(args)
    elif args.command == "resume":
        cmd_resume(args)
    elif args.command == "resume-batch":
        cmd_resume_batch(args)
    elif args.command == "bulk-import":
        cmd_bulk_import(args)
    elif args.command == "backfill-speakers":
        cmd_backfill_speakers(args)


if __name__ == "__main__":
    main()
