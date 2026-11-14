"""
Stats service — the dashboard aggregations, ported from the Flask app
with identical response shapes, fed by DB-built DataFrames.

What CHANGED under the hood (invisible to the UI):
    - review state comes from segments.review_status / transcript_edited /
      trimmed (review_status.json is gone);
    - rejected segments stay in the DB but are EXCLUDED from aggregates,
      matching the old behaviour where a reject removed the CSV row;
    - the Conf 1/2/3 distribution is computed from whisper_conf_min /
      whisper_conf with the pipeline's own formula — no more sampling
      thousands of annotation files per request;
    - the vocabulary walks the indexed `words` table in ONE SQL query
      instead of parsing every annotation file (the app's slowest scan),
      with the old evenly-distributed fallback for segments lacking words.
"""

import math
import time as _time

import pandas as pd

from vsr_shared.catalog_db import CatalogDatabase
from api.dataframes import segments_frame, videos_frame

_cache: dict = {}
_CACHE_TTL = 20.0        # seconds — stats change only when a new segment is written
_VOCAB_CACHE_TTL = 3600  # vocabulary is large and slow; rebuild at most once per hour


def _cached(key: str, fn, db: CatalogDatabase):
    now = _time.monotonic()
    entry = _cache.get(key)
    if entry and (now - entry[0]) < _CACHE_TTL:
        return entry[1]
    result = fn(db)
    _cache[key] = (now, result)
    return result


def _conf_level(conf_min, conf_mean) -> int:
    """LRS2 Conf 1/2/3 — same rule as services/segmenter/transcription.py."""
    if conf_min is None or conf_mean is None:
        return 1
    try:
        conf_min, conf_mean = float(conf_min), float(conf_mean)
    except (TypeError, ValueError):
        return 1
    if math.isnan(conf_min) or math.isnan(conf_mean):
        return 1
    if conf_min >= 0.7 and conf_mean >= 0.9:
        return 3
    if conf_min >= 0.5 and conf_mean >= 0.7:
        return 2
    return 1


def _active_segments(db: CatalogDatabase) -> pd.DataFrame:
    """Segments minus the rejected ones (parity with the CSV-removal era)."""
    df = segments_frame(db)
    if df.empty or "review_status" not in df.columns:
        return df
    return df[df["review_status"].fillna("") != "rejected"]


def _clean_word(word: str) -> str:
    return word.strip(".,;:!?\"'()[]").upper()


# --------------------------------------------------------------- /api/stats

def stats_videos(db: CatalogDatabase) -> dict:
    df = videos_frame(db)
    if df.empty:
        return {"total": 0, "by_status": {}}

    by_status = (
        {str(k): int(v) for k, v in df["status"].value_counts().to_dict().items()}
        if "status" in df.columns else {}
    )
    ext_series = pd.to_numeric(
        df.get("total_duration_extracted", pd.Series(dtype=float)), errors="coerce")
    src_series = pd.to_numeric(
        df.get("duration_seconds", pd.Series(dtype=float)), errors="coerce")
    total_dur = float(ext_series.fillna(0).sum())
    total_src = float(src_series.fillna(0).sum())

    def _dist(series):
        series = series.dropna()
        if series.empty:
            return None
        return {"min": round(float(series.min()), 2),
                "mean": round(float(series.mean()), 2),
                "max": round(float(series.max()), 2)}

    def _vc(col):
        if col not in df.columns:
            return {}
        s = df[col].dropna().astype(str).str.strip()
        s = s[s != ""]
        return {str(k): int(v) for k, v in s.value_counts().head(20).to_dict().items()}

    speech_by_region: dict = {}
    if "region" in df.columns and "total_duration_extracted" in df.columns:
        for region, sub in df.groupby(df["region"].fillna("UNKNOWN").astype(str)):
            total = pd.to_numeric(
                sub["total_duration_extracted"], errors="coerce").fillna(0).sum()
            if total > 0:
                speech_by_region[region] = round(float(total), 2)

    top_extracted, top_mined = [], []
    if "video_id" in df.columns:
        columns = [c for c in ("video_id", "title", "region") if c in df.columns]
        aux = df[columns].copy()
        aux["_extracted"] = ext_series
        aux["_source"] = src_series
        for _, r in aux.dropna(subset=["_extracted"]).sort_values(
                "_extracted", ascending=False).head(10).iterrows():
            top_extracted.append({
                "video_id": str(r.get("video_id", "")),
                "title": str(r.get("title", "") or ""),
                "region": str(r.get("region", "") or ""),
                "extracted_s": round(float(r["_extracted"]), 1),
                "source_s": None if pd.isna(r["_source"]) else round(float(r["_source"]), 1),
            })
        mined = aux[(aux["_source"] > 0) & aux["_extracted"].notna()].copy()
        mined["_ratio"] = mined["_extracted"] / mined["_source"]
        for _, r in mined.sort_values("_ratio", ascending=False).head(10).iterrows():
            top_mined.append({
                "video_id": str(r.get("video_id", "")),
                "title": str(r.get("title", "") or ""),
                "region": str(r.get("region", "") or ""),
                "extracted_s": round(float(r["_extracted"]), 1),
                "source_s": round(float(r["_source"]), 1),
                "ratio_pct": round(float(r["_ratio"]) * 100, 1),
            })

    return {
        "total": len(df),
        "by_status": by_status,
        "by_region": _vc("region"),
        "by_source": _vc("source"),
        "by_license": _vc("license"),
        "top_channels": _vc("source_channel"),
        "speech_by_region_s": speech_by_region,
        "total_duration_h": round(total_dur / 3600, 2),
        "total_source_duration_s": round(total_src, 2),
        "total_source_duration_h": round(total_src / 3600, 2),
        "source_duration": _dist(src_series),
        "extracted_duration": _dist(ext_series[ext_series > 0]),
        "top_extracted": top_extracted,
        "top_mined": top_mined,
    }


# Candidate training filter (notes/filtrare_planificare.md, "balanced"),
# applied step by step so the dashboard shows where the hours are lost.
_CONSENSUS_OK = ("all_agree", "dialectal_agree")
_MIN_TRAIN_S, _MAX_TRAIN_S = 1.0, 16.0


def _dissertation_stats(seg_df: pd.DataFrame, durations: pd.Series) -> dict:
    """Training funnel, transcript consensus and digit-label counts."""
    def num(col):
        if col not in seg_df.columns:
            return pd.Series(float("nan"), index=seg_df.index)
        return pd.to_numeric(seg_df[col], errors="coerce")

    def part(mask) -> dict:
        return {"count": int(mask.sum()),
                "duration_s": round(float(durations[mask].fillna(0).sum()), 1)}

    consensus = (seg_df["transcript_consensus"].fillna("").astype(str).str.strip()
                 if "transcript_consensus" in seg_df.columns
                 else pd.Series("", index=seg_df.index))

    # NaN comparisons are False, so a missing score never passes a filter.
    visual = (num("asd_score") >= 2) & (num("mouth_landmark_fail_rate") < 0.5)
    text_ok = visual & (num("whisper_conf") > 0.20) & (num("num_words") >= 4)
    length_ok = text_ok & (durations >= _MIN_TRAIN_S) & (durations <= _MAX_TRAIN_S)
    agreed = length_ok & consensus.isin(_CONSENSUS_OK)
    training_funnel = [
        {"key": "all", "label": "All extracted", **part(durations.notna())},
        {"key": "visual", "label": "Speaker visible (ASD ≥ 2, mouth landmarks OK)", **part(visual)},
        {"key": "text", "label": "Usable transcript (Whisper conf > 0.20, ≥ 4 words)", **part(text_ok)},
        {"key": "length", "label": f"Length {_MIN_TRAIN_S:g}–{_MAX_TRAIN_S:g} s", **part(length_ok)},
        {"key": "agreed", "label": "Whisper variants agree", **part(agreed)},
    ]

    by_consensus = {}
    for name in consensus[consensus != ""].unique():
        by_consensus[str(name)] = part(consensus == name)
    if (consensus == "").any():
        by_consensus["not computed"] = part(consensus == "")
    dialect_wer = num("wer_dialectat_mediu_vs_large").dropna()

    # Digits are not aligned by wav2vec2 and are not what the lips say —
    # these labels need number normalisation before training.
    text = seg_df["text"].fillna("").astype(str)
    tokens = text.str.split()
    digit_share = tokens.map(
        lambda ws: sum(any(c.isdigit() for c in w) for w in ws) / len(ws) if ws else 0.0)
    numbers = {"with_digits": part(digit_share > 0),
               "mostly_digits": part(digit_share >= 0.5)}

    speaker = (seg_df["speaker_id"].fillna("").astype(str).str.strip()
               if "speaker_id" in seg_df.columns else pd.Series("", index=seg_df.index))
    speaker_dur = durations[speaker != ""].groupby(speaker[speaker != ""]).sum()
    speaker_dur = speaker_dur.sort_values(ascending=False)
    total_spk_s = float(speaker_dur.sum()) or 1.0
    speaker_balance = {
        "n_speakers": int(len(speaker_dur)),
        "top1_id": str(speaker_dur.index[0]) if len(speaker_dur) else None,
        "top1_pct": round(100.0 * float(speaker_dur.iloc[:1].sum()) / total_spk_s, 1),
        "top10_pct": round(100.0 * float(speaker_dur.iloc[:10].sum()) / total_spk_s, 1),
        "with_10min": int((speaker_dur >= 600).sum()),
        "under_1min": int((speaker_dur < 60).sum()),
    }

    return {
        "training_funnel": training_funnel,
        "by_consensus": by_consensus,
        "dialect_wer": ({"mean": round(float(dialect_wer.mean()), 4),
                         "median": round(float(dialect_wer.median()), 4),
                         "count": int(len(dialect_wer))}
                        if not dialect_wer.empty else None),
        "numbers": numbers,
        "speaker_balance": speaker_balance,
    }


def stats_segments(db: CatalogDatabase) -> dict:
    full = segments_frame(db)
    if full.empty:
        return {"total": 0}
    review_series = full.get(
        "review_status", pd.Series("", index=full.index)).fillna("")
    rejected_count = int((review_series == "rejected").sum())
    seg_df = full[review_series != "rejected"]
    if seg_df.empty:
        return {"total": 0, "rejected_count": rejected_count}

    n = len(seg_df)
    durations = pd.to_numeric(seg_df["duration"], errors="coerce")
    total_dur_s = float(durations.fillna(0).sum())
    total_words = int(pd.to_numeric(seg_df["num_words"], errors="coerce").fillna(0).sum())
    total_chars = int(pd.to_numeric(seg_df["num_chars"], errors="coerce").fillna(0).sum())

    def _vocab_of(frame) -> set:
        vocab = set()
        for text in frame["text"].dropna().astype(str):
            for word in text.split():
                cleaned = _clean_word(word)
                if cleaned:
                    vocab.add(cleaned)
        return vocab

    unique_words = len(_vocab_of(seg_df))

    approved_mask = seg_df["review_status"].fillna("") == "approved"
    approved_df = seg_df[approved_mask]
    approved = {
        "count": int(len(approved_df)),
        "total_duration_s": round(float(
            pd.to_numeric(approved_df["duration"], errors="coerce").fillna(0).sum()), 2),
        "total_words": int(pd.to_numeric(
            approved_df["num_words"], errors="coerce").fillna(0).sum()),
        "unique_words": len(_vocab_of(approved_df)) if not approved_df.empty else 0,
    }
    edited_count = int((
        (pd.to_numeric(full.get("transcript_edited", 0), errors="coerce").fillna(0) == 1)
        | (pd.to_numeric(full.get("trimmed", 0), errors="coerce").fillna(0) == 1)
    ).sum())

    wer_series = pd.to_numeric(seg_df["wer"], errors="coerce")
    avg_wer = round(float(wer_series.mean()), 4) if wer_series.notna().any() else None
    with_wer_count = int(wer_series.notna().sum())

    duration_stats = None
    duration_bands = []
    d = durations.dropna()
    if not d.empty:
        duration_stats = {
            "min": round(float(d.min()), 2),
            "p25": round(float(d.quantile(0.25)), 2),
            "median": round(float(d.median()), 2),
            "mean": round(float(d.mean()), 2),
            "p75": round(float(d.quantile(0.75)), 2),
            "p95": round(float(d.quantile(0.95)), 2),
            "max": round(float(d.max()), 2),
        }
        for label, lo, hi in [("<1s", None, 1.0), ("1–3s", 1.0, 3.0),
                              ("3–6s", 3.0, 6.0), ("6–10s", 6.0, 10.0),
                              ("10–15s", 10.0, 15.0), (">15s", 15.0, None)]:
            if lo is None:
                mask = d < hi
            elif hi is None:
                mask = d >= lo
            else:
                mask = (d >= lo) & (d < hi)
            count = int(mask.sum())
            duration_bands.append({
                "label": label,
                "count": count,
                "pct": round(100.0 * count / len(d), 1) if len(d) else 0.0,
                "duration_s": round(float(d[mask].sum()), 1),
            })

    words_numeric = pd.to_numeric(seg_df["num_words"], errors="coerce").dropna()
    words_stats = None
    if not words_numeric.empty:
        words_stats = {"min": int(words_numeric.min()),
                       "median": int(words_numeric.median()),
                       "mean": round(float(words_numeric.mean()), 1),
                       "max": int(words_numeric.max())}

    seg_by_region: dict = {}
    if "region" in seg_df.columns:
        for region, sub in seg_df.groupby(seg_df["region"].astype(str)):
            seg_by_region[region] = {
                "count": int(len(sub)),
                "duration_s": round(float(
                    pd.to_numeric(sub["duration"], errors="coerce").fillna(0).sum()), 1),
            }

    # Conf distribution — vectorized (iterrows over thousands of segments is ~100x slower).
    _cm = pd.to_numeric(
        seg_df["whisper_conf_min"] if "whisper_conf_min" in seg_df.columns
        else pd.Series(dtype=float), errors="coerce")
    _ca = pd.to_numeric(
        seg_df["whisper_conf"] if "whisper_conf" in seg_df.columns
        else pd.Series(dtype=float), errors="coerce")
    _both_valid = _cm.notna() & _ca.notna()
    _tier3 = _both_valid & (_cm >= 0.7) & (_ca >= 0.9)
    _tier2 = _both_valid & (~_tier3) & (_cm >= 0.5) & (_ca >= 0.7)
    conf_dist = {
        "3": int(_tier3.sum()),
        "2": int(_tier2.sum()),
        "1": int((_both_valid & ~_tier3 & ~_tier2).sum()),
        "unknown": int((~_both_valid).sum()),
    }

    word_freq: dict = {}
    for text in seg_df["text"].dropna().astype(str):
        for word in text.split():
            cleaned = _clean_word(word)
            if cleaned:
                word_freq[cleaned] = word_freq.get(cleaned, 0) + 1
    rare1 = sum(1 for c in word_freq.values() if c == 1)
    rare2 = sum(1 for c in word_freq.values() if c == 2)

    tiers = None
    training_ready = None
    if "quality_tier" in seg_df.columns:
        tier_series = seg_df["quality_tier"].fillna("").astype(str).str.strip().str.upper()
        tiers = {}
        for tier_name in ("A", "B", "C"):
            mask = tier_series == tier_name
            tiers[tier_name] = {
                "count": int(mask.sum()),
                "duration_s": round(float(durations[mask].fillna(0).sum()), 2),
            }
        ready_mask = approved_mask & tier_series.isin(["A", "B"])
        training_ready = {
            "count": int(ready_mask.sum()),
            "duration_s": round(float(durations[ready_mask].fillna(0).sum()), 2),
        }

    wpm = round((total_words / total_dur_s) * 60.0, 1) if total_dur_s > 0 else 0
    return {
        **_dissertation_stats(seg_df, durations),
        "total": n,
        "total_duration_s": round(total_dur_s, 2),
        "total_duration_h": round(total_dur_s / 3600, 3),
        "avg_duration_s": round(float(durations.mean()), 2) if n else 0,
        "duration_stats": duration_stats,
        "duration_bands": duration_bands,
        "words_stats": words_stats,
        "words_per_minute": wpm,
        "total_words": total_words,
        "unique_words": unique_words,
        "rare_words_1": rare1,
        "rare_words_2": rare2,
        "total_chars": total_chars,
        "avg_words": round(float(words_numeric.mean()), 1) if not words_numeric.empty else 0,
        "avg_asd": round(float(pd.to_numeric(
            seg_df["asd_score"], errors="coerce").mean()), 3),
        "avg_syncnet": round(float(pd.to_numeric(
            seg_df["syncnet_conf"], errors="coerce").mean()), 3),
        "avg_whisper_conf": round(float(pd.to_numeric(
            seg_df["whisper_conf"], errors="coerce").mean()), 3),
        "avg_wer": avg_wer,
        "with_wer_count": with_wer_count,
        "edited_count": edited_count,
        "approved": approved,
        "rejected_count": rejected_count,
        "by_region": seg_by_region,
        "by_conf": conf_dist,
        "tiers": tiers,
        "training_ready": training_ready,
    }


# ------------------------------------------------- /api/stats/distributions

def distributions(db: CatalogDatabase) -> dict:
    df = _active_segments(db)
    if df.empty:
        return {"wer_buckets": [], "duration_buckets": [], "health": {}}
    n = len(df)

    wer_buckets = []
    wer_series = pd.to_numeric(df["wer"], errors="coerce").dropna()
    if not wer_series.empty:
        edges = [i / 20 for i in range(21)]
        for i in range(20):
            lo, hi = edges[i], edges[i + 1]
            mask = (wer_series >= lo) & (
                (wer_series < hi) if i < 19 else (wer_series <= hi))
            wer_buckets.append({"lo_pct": round(lo * 100, 1),
                                "hi_pct": round(hi * 100, 1),
                                "count": int(mask.sum())})

    duration_buckets = []
    d = pd.to_numeric(df["duration"], errors="coerce").dropna()
    if not d.empty:
        for sec in range(1, 15):
            mask = (d >= sec) & ((d < sec + 1) if sec < 14 else (d <= sec + 1))
            duration_buckets.append(
                {"lo_s": sec, "hi_s": sec + 1, "count": int(mask.sum())})

    def _missing(col):
        if col not in df.columns:
            return n
        series = df[col]
        return int(series.isna().sum() + (series.astype(str) == "").sum())

    _cm = pd.to_numeric(
        df["whisper_conf_min"] if "whisper_conf_min" in df.columns
        else pd.Series(dtype=float), errors="coerce")
    _ca = pd.to_numeric(
        df["whisper_conf"] if "whisper_conf" in df.columns
        else pd.Series(dtype=float), errors="coerce")
    _valid = _cm.notna() & _ca.notna()
    scanned = int(_valid.sum())
    _t3 = _valid & (_cm >= 0.7) & (_ca >= 0.9)
    _t2 = _valid & (~_t3) & (_cm >= 0.5) & (_ca >= 0.7)
    conf1_count = int((_valid & ~_t3 & ~_t2).sum())

    def _pct(value, denom):
        return round(100.0 * value / denom, 1) if denom else 0.0

    missing_speaker = _missing("speaker_id")
    missing_wer = int(pd.to_numeric(df["wer"], errors="coerce").isna().sum())
    return {
        "wer_buckets": wer_buckets,
        "duration_buckets": duration_buckets,
        "health": {
            "n_segments": n,
            "missing_speaker_id": {"count": missing_speaker,
                                   "pct": _pct(missing_speaker, n)},
            "missing_wer": {"count": missing_wer, "pct": _pct(missing_wer, n)},
            "conf_1_low": {"count": conf1_count,
                           "annotations_scanned": scanned,
                           "pct_of_scanned": _pct(conf1_count, scanned)},
        },
    }


# --------------------------------------------------------- /api/vocabulary

def vocabulary(db: CatalogDatabase) -> dict:
    """Word table with REAL per-word durations — one aggregation over `words`.

    A two-level group (word, segment_id) collapsed in Python gives the number
    of distinct segments per word without a costly distinct-count; rejected
    segments are excluded inside the aggregation.
    """
    rejected_ids = db.segments.rejected_ids()
    rows = db.segments.word_segment_counts(rejected_ids)

    counts: dict = {}
    for r in rows:
        entry = counts.setdefault(
            r["word"], {"samples": 0, "occurrences": 0, "duration": 0.0})
        entry["samples"] += 1
        entry["occurrences"] += int(r["occ"])
        entry["duration"] += float(r["dur"] or 0.0)

    # Fallback (old semantics): segments with no word rows contribute their
    # duration spread evenly over the transcript words.
    for row in db.segments.without_words(exclude_rejected=True):
        text = (row["text"] or "").strip()
        if not text:
            continue
        words_in_seg = [w for w in (_clean_word(t) for t in text.split()) if w]
        if not words_in_seg:
            continue
        per_word = float(row["duration"] or 0.0) / len(words_in_seg)
        seen = set()
        for word in words_in_seg:
            entry = counts.setdefault(
                word, {"samples": 0, "occurrences": 0, "duration": 0.0})
            entry["occurrences"] += 1
            entry["duration"] += max(0.0, per_word)
            if word not in seen:
                seen.add(word)
                entry["samples"] += 1

    words = [{"word": w, "samples": e["samples"], "occurrences": e["occurrences"],
              "duration": round(e["duration"], 2)} for w, e in counts.items()]
    words.sort(key=lambda x: x["samples"], reverse=True)
    return {"words": words, "total_unique": len(words)}


# ── Cached entry-points called by the router ──────────────────────────────

def get_stats(db: CatalogDatabase) -> dict:
    def _compute(d):
        videos = stats_videos(d)
        payload = {"videos": videos, "segments": stats_segments(d)}
        if "total_duration_h" in videos:
            payload["videos"]["total_duration_s"] = round(
                videos["total_duration_h"] * 3600, 2)
        return payload
    return _cached("stats", _compute, db)


def get_distributions(db: CatalogDatabase) -> dict:
    return _cached("distributions", distributions, db)


def get_vocabulary(db: CatalogDatabase) -> dict:
    now = _time.monotonic()
    entry = _cache.get("vocabulary")
    if entry and (now - entry[0]) < _VOCAB_CACHE_TTL:
        return entry[1]
    result = vocabulary(db)
    _cache["vocabulary"] = (now, result)
    return result
