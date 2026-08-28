#!/usr/bin/env python3
"""
Romanian VSR Dataset Pipeline v2 - CLI Entry Point

Usage:
    # Initialize project structure
    python backend/orchestrator/cli.py init ./data

    # Process single video
    python backend/orchestrator/cli.py single ro_001 "https://youtube.com/watch?v=..."

    # Process batch from Excel
    python backend/orchestrator/cli.py batch --limit 10

    # Show stats
    python backend/orchestrator/cli.py stats
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
        from vsr_shared.catalog_db import CatalogDatabase
        db = CatalogDatabase(
            Path(cfg["paths"]["base_dir"]) / "catalog" / "dataset.db")
        row = db.videos.get(args.video_id)
        db.close()
        if row:
            url = str(row.get("youtube_url") or "")
    if not url:
        print(f"Error: could not find a URL for '{args.video_id}' in the catalog.")
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

    # args.excel acceptat pentru compatibilitate — IGNORAT (DB e sursa)
    pipeline = VSRPipeline.from_config(args.config)
    updated = pipeline.sync_excel_from_disk(
        video_ids=args.video_id or None,
    )

    print(f"\n{'='*50}")
    if updated:
        print(f"Updated {updated} video row(s) in dataset.db.")
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

    # args.excel acceptat pentru compatibilitate — IGNORAT (DB e sursa)

    # Storage v2 final: everything reads dataset.db (CSVs are exports only).
    config_path = Path(args.config)
    base_dir = Path(".")
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        base_dir = Path(cfg["paths"]["base_dir"])
    metadata_dir = base_dir / "catalog"

    from vsr_shared.catalog_db import CatalogDatabase
    db = CatalogDatabase(metadata_dir / "dataset.db")
    df = pd.DataFrame(db.videos.all())
    seg_rows = db.connection.execute(
        "SELECT s.*, COALESCE(NULLIF(v.region, ''), 'UNKNOWN') AS region"
        " FROM segments s LEFT JOIN videos v USING (video_id)"
        " WHERE COALESCE(s.review_status, '') != 'rejected'").fetchall()
    seg_df = pd.DataFrame([dict(r) for r in seg_rows])
    speakers_rows = db.speakers.all_with_stats()
    for r in speakers_rows:
        r.pop("centroid", None)
    speakers_df = pd.DataFrame(speakers_rows)
    db.close()

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


def parse_import_line(entry: str):
    """One bulk-import line → (video_id | None, url | None).

    Formats:  "https://..."            → (None, url)      classic
              "md_001 https://..."     → ("md_001", url)  pre-downloaded
    Returns (None, None) for an unrecognized line.
    """
    import re as _re
    parts = entry.split()
    if len(parts) == 1 and "://" in parts[0]:
        return None, parts[0]
    if (len(parts) == 2 and "://" in parts[1]
            and _re.fullmatch(r"[A-Za-z0-9][\w-]*", parts[0])):
        return parts[0], parts[1]
    return None, None


def _upsert_video_row(pipeline, video_id: str, row: dict) -> None:
    """Mirror one registered video into dataset.db (videos table)."""
    try:
        fields = {k: v for k, v in row.items() if v not in ("", None)}
        pipeline.catalog.db.videos.ensure_exists(video_id)
        pipeline.catalog.db.videos.update_fields(video_id, fields)
    except Exception as e:
        logger.debug(f"DB upsert skipped for {video_id}: {e}")


def cmd_bulk_import(args):
    """Register a list of YouTube URLs: download (or map) + add catalog rows.

    Two line formats, auto-detected:
      "https://..."           classic — next {prefix}_NNN id + download;
      "md_001 https://..."    pre-downloaded — uses the GIVEN id; when
                              data/raw/{id}.mp4 exists, NO download happens:
                              metadata is fetched from the link, the raw file
                              is integrity-checked with ffprobe (duration vs
                              metadata), and the row is registered pending.
    --pre-downloaded makes the pair format MANDATORY (strict validation).
    Rows land in dataset.db (CSVs are exports only).
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

    prefix = args.prefix or "vid"
    region = args.region or "UNKNOWN"
    source = args.source or "YouTube_CC"
    verify_cc = not args.no_cc_check

    # Load pipeline just to get a ready-to-use downloader with correct config
    pipeline = VSRPipeline.from_config(args.config)
    _apply_cookie_overrides(pipeline, args)
    downloader: YouTubeDownloader = pipeline.services.downloader

    # Sursa de adevăr: tabela videos din dataset.db (CSV-urile sunt exporturi).
    db_rows = pipeline.catalog.db.videos.all()
    existing_ids = {str(r["video_id"]) for r in db_rows}
    existing_urls = {str(r.get("youtube_url") or "") for r in db_rows} - {""}

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
    added, failed, skipped, registered = 0, 0, 0, 0

    # Track IDs we imported in this run so duplicate URLs inside the same
    # input list are also caught.
    imported_yt_ids: set = set()

    strict_pairs = bool(getattr(args, "pre_downloaded", False))
    for i, entry in enumerate(urls, 1):
        given_id, url = parse_import_line(entry)
        if url is None or (strict_pairs and given_id is None):
            expected = "'md_001 https://...'" if strict_pairs else "'URL' sau 'id URL'"
            logger.error(f"  Linie nerecunoscută (aștept {expected}): {entry!r}")
            failed += 1
            continue
        logger.info(f"[{i}/{total}] Importing: {url}"
                    + (f" (id dat: {given_id})" if given_id else ""))

        if given_id and given_id in existing_ids:
            logger.warning(f"  {given_id} e deja înregistrat — sărit")
            skipped += 1
            continue

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

        if given_id:
            video_id = given_id
        else:
            video_id = f"{prefix}_{next_num:03d}"
            while video_id in existing_ids:
                next_num += 1
                video_id = f"{prefix}_{next_num:03d}"

        row = {col: "" for col in VIDEOS_MASTER_SCHEMA.keys()}
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

        # ── Pre-downloaded raw on disk? Map it: metadata is already fetched
        # above; verify the FILE (ffprobe: streams + duration vs metadata) so
        # a corrupt copy is caught NOW, not mid-pipeline. No download.
        raw_path = downloader.output_dir / f"{video_id}.mp4"
        if raw_path.exists():
            expected = None
            try:
                expected = float(info_obj.duration) if info_obj and info_obj.duration else None
            except (TypeError, ValueError):
                expected = None
            try:
                properties = downloader._probe_video(raw_path, expected)
                row["status"] = ProcessingStatus.PENDING.value
                added += 1
                registered += 1
                logger.info(f"  {video_id}: raw pe disc, integritate OK "
                            f"[{properties}] — fără descărcare → pending")
            except Exception as e:
                row["status"] = ProcessingStatus.FAILED.value
                row["error_message"] = f"raw file failed integrity check: {e}"[:500]
                failed += 1
                logger.error(f"  {video_id}: raw pe disc dar PICĂ verificarea "
                             f"ffprobe ({e}) — marcat failed, fișierul rămâne")
            _upsert_video_row(pipeline, video_id, row)
            existing_ids.add(video_id)
            existing_urls.add(url)
            imported_yt_ids.add(yt_id)
            continue

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

        # Row-ul intră imediat în DB — UI-ul îl vede în timp real
        _upsert_video_row(pipeline, video_id, row)
        existing_ids.add(video_id)
        existing_urls.add(url)
        imported_yt_ids.add(yt_id)
        if not given_id:
            next_num += 1

    print(f"\n{'='*50}")
    print(f"Bulk import summary")
    print(f"  URLs processed : {total}")
    print(f"  Downloaded     : {added}")
    print(f"  Failed         : {failed}")
    print(f"  Skipped (dup.) : {skipped}")
    print(f"  Pre-downloaded : {registered}  (mapate fără descărcare)")
    print(f"  Catalog        : dataset.db (CSV doar prin export_catalog.py)")



def cmd_init(args):

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
  python backend/orchestrator/cli.py stats
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
    p_batch.add_argument("excel", nargs="?", default="",
                         help="ignorat — selecția vine din dataset.db")
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
    p_stats.add_argument("excel", nargs="?", default="",
                         help="ignorat — statisticile vin din dataset.db")
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
    p_resume_batch.add_argument("excel", nargs="?", default="",
                                help="ignorat — selecția vine din dataset.db")
    p_resume_batch.add_argument("--limit", type=int, help="Max videos to resume")
    p_resume_batch.add_argument(
        "--video-id",
        nargs="+",
        metavar="ID",
        help="Resume only these video IDs (bypasses the status filter)",
    )

    # bulk-import — download/map a list of YouTube URLs into dataset.db
    p_bulk = subs.add_parser(
        "bulk-import",
        help="Download/map YouTube URLs into the catalog (dataset.db)",
    )
    p_bulk.add_argument("excel", nargs="?", default="",
                        help="ignorat — rândurile intră în dataset.db")
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
    p_bulk.add_argument("--pre-downloaded", action="store_true",
                        help="strict: every line must be 'id URL'; raw files "
                             "already in data/raw are mapped, not downloaded")
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
    p_sync.add_argument("excel", nargs="?", default="",
                        help="ignorat — sincronizarea scrie dataset.db")
    p_sync.add_argument(
        "--video-id",
        nargs="+",
        metavar="ID",
        help="Only sync these video IDs (default: all rows missing stats)",
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


if __name__ == "__main__":
    main()
