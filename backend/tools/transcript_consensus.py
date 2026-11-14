"""
Consens 3-way între transcrierile Whisper
==========================================
Perechile WER disponibile în DB:
  wer_standard_vs_dialectat_mediu  = WER(whisper_standard_mediu, whisper_dialectat_mediu)
  wer_mediu_vs_large               = WER(whisper_standard_mediu, whisper_dialectat_large)
                                     [denumire înșelătoare — de fapt std vs large-dialectal]

Perechea lipsă calculată de acest script:
  wer_dialectat_mediu_vs_large     = WER(whisper_dialectat_mediu, whisper_dialectat_large)

Categorie consens (prag T = 0.20):
  all_agree        — toate 3 perechi < T  → toate modelele sunt de acord
  dialectal_agree  — med+large < T, std diferă  → prompt-ul dialectal contează
  std_large_agree  — std+large < T, med diferă
  std_med_agree    — std+med < T, large diferă
  all_disagree     — nicio pereche sub T

Comenzi uzuale
--------------
# Calculează WER lipsă + consens + afișează statistici
docker exec -it vsr-pipeline-worker python /app/backend/tools/transcript_consensus.py

# Recalculează tot
docker exec -it vsr-pipeline-worker python /app/backend/tools/transcript_consensus.py --force

# Doar statistici (nu scrie nimic)
docker exec -it vsr-pipeline-worker python /app/backend/tools/transcript_consensus.py --stats-only
"""

import argparse
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _BACKEND_DIR.parent
sys.path[:0] = [str(_BACKEND_DIR), str(_BACKEND_DIR / "shared")]

from loguru import logger  # noqa: E402
from vsr_shared.catalog_db import CatalogDatabase  # noqa: E402
from vsr_shared.wer_utils import compute_wer  # noqa: E402

AGREE_THRESHOLD = 0.20
BATCH_SIZE      = 500


# ── Consensus logic ───────────────────────────────────────────────────────────

def _consensus(w_sm, w_sl, w_ml, T=AGREE_THRESHOLD):
    """
    w_sm = WER(standard, med-dialectal)
    w_sl = WER(standard, large-dialectal)
    w_ml = WER(med-dialectal, large-dialectal)
    """
    if w_sm is None or w_sl is None or w_ml is None:
        return "insufficient"
    agree_sm = w_sm < T
    agree_sl = w_sl < T
    agree_ml = w_ml < T
    n = agree_sm + agree_sl + agree_ml
    if n == 3:
        return "all_agree"
    if n == 0:
        return "all_disagree"
    if agree_ml:
        return "dialectal_agree"   # med+large agree, std diferă
    if agree_sl:
        return "std_large_agree"   # std+large agree, med diferă
    return "std_med_agree"         # std+med agree, large diferă


# ── Main processing ───────────────────────────────────────────────────────────

def _frame(db: CatalogDatabase, fields):
    import pandas as pd
    rows = db.segments.query(fields=fields, exclude_rejected=True)
    return pd.DataFrame([{f: r.get(f) for f in fields} for r in rows], columns=fields)


def run(force: bool):
    db = CatalogDatabase()
    fields = ["segment_id", "whisper_dialectat_mediu", "whisper_dialectat_large",
              "wer_standard_vs_dialectat_mediu", "wer_mediu_vs_large",
              "wer_dialectat_mediu_vs_large", "transcript_consensus"]
    rows = [r for r in db.segments.query(fields=fields, exclude_rejected=True)
            if force or r.get("wer_dialectat_mediu_vs_large") is None
            or r.get("transcript_consensus") is None]
    rows.sort(key=lambda r: r["segment_id"])

    if not rows:
        logger.info("Nimic de calculat.")
        return

    logger.info(f"{len(rows)} segmente de procesat...")
    pending = {}
    done = 0
    for r in rows:
        # Perechea lipsă: dialectat_mediu vs dialectat_large
        w_ml, _ = compute_wer(r.get("whisper_dialectat_mediu"),
                              r.get("whisper_dialectat_large"))
        label = _consensus(r.get("wer_standard_vs_dialectat_mediu"),
                           r.get("wer_mediu_vs_large"), w_ml)
        pending[r["segment_id"]] = {
            "wer_dialectat_mediu_vs_large": round(w_ml, 4) if w_ml is not None else None,
            "transcript_consensus": label}
        done += 1
        if done % BATCH_SIZE == 0:
            db.segments.set_many(pending)
            pending.clear()
            logger.info(f"  {done}/{len(rows)} gata")

    db.segments.set_many(pending)
    logger.info(f"Gata: {done} segmente actualizate.")


# ── Statistics ────────────────────────────────────────────────────────────────

_ORDER = {"all_agree": 1, "dialectal_agree": 2, "std_large_agree": 3,
          "std_med_agree": 4, "all_disagree": 5}


def stats():
    db = CatalogDatabase()
    df = _frame(db, ["transcript_consensus", "quality_tier", "duration",
                     "wer_standard_vs_dialectat_mediu", "wer_mediu_vs_large",
                     "wer_dialectat_mediu_vs_large"])
    seg_total = len(df)
    h_total = float(df["duration"].fillna(0).sum()) / 3600.0

    print("\n" + "=" * 90)
    print("STATISTICI CONSENS TRANSCRIPT 3-WAY")
    print("  Perechi WER:")
    print("    (1) std  vs med-dialectal  = wer_standard_vs_dialectat_mediu")
    print("    (2) std  vs large-dialectal = wer_mediu_vs_large  [denumire inselatoare]")
    print("    (3) med-dialectal vs large-dialectal = wer_dialectat_mediu_vs_large  [NOU]")
    print(f"  Prag acord: WER < {AGREE_THRESHOLD:.0%}")
    print(f"  Total: {seg_total:,} segmente  /  {h_total:.2f}h")
    print("=" * 90)
    if not seg_total:
        return

    df["cons"] = df["transcript_consensus"].fillna("(null)")
    df["rank"] = df["transcript_consensus"].map(_ORDER).fillna(6)

    # ── 1. Totale per consens ──────────────────────────────────────────────
    print(f"\n[ Totale per categorie consens ]")
    print(f"  {'Consens':<18} {'Segm':>7}  {'Ore':>8}  {'%':>6}  Interpretare")
    print(f"  {'-'*80}")
    interp = {
        "all_agree":       "toate 3 modele produc acelasi text",
        "dialectal_agree": "med+large dialect concord, std difera",
        "std_large_agree": "std+large concord, med-dialect difera",
        "std_med_agree":   "std+med concord, large-dialect difera",
        "all_disagree":    "nicio pereche de acord — risc maxim",
        "insufficient":    "transcript lipsa",
    }
    g = (df.groupby(["rank", "cons"], dropna=False)
           .agg(n=("cons", "size"), h=("duration", "sum")).reset_index()
           .sort_values(["rank", "cons"]))
    for _, row in g.iterrows():
        n, h = int(row["n"]), float(row["h"] or 0) / 3600.0
        print(f"  {row['cons']:<18} {n:>7}  {h:>8.2f}h  {100.0 * n / seg_total:>5.1f}%"
              f"  {interp.get(row['cons'], '')}")
    print(f"  {'TOTAL':<18} {seg_total:>7}  {h_total:>8.2f}h  100.0%")

    # ── 2. Consens × tier ──────────────────────────────────────────────────
    print(f"\n[ Consens x Tier ]")
    print(f"  {'Consens':<18} {'Tier':<5}  {'Segm':>7}  {'Ore':>8}  {'%':>6}")
    print(f"  {'-'*55}")
    df["tier"] = df["quality_tier"].fillna("?").replace("", "?")
    g = (df.groupby(["rank", "cons", "tier"])
           .agg(n=("cons", "size"), h=("duration", "sum")).reset_index()
           .sort_values(["rank", "cons", "tier"]))
    prev = None
    for _, row in g.iterrows():
        if row["cons"] != prev and prev is not None:
            print()
        prev = row["cons"]
        n, h = int(row["n"]), float(row["h"] or 0) / 3600.0
        print(f"  {row['cons']:<18} {row['tier']:<5}  {n:>7}  {h:>8.2f}h"
              f"  {100.0 * n / seg_total:>5.1f}%")

    # ── 3. WER distributions ───────────────────────────────────────────────
    wer_cols = [
        ("wer_standard_vs_dialectat_mediu", "std vs med   "),
        ("wer_mediu_vs_large",              "std vs large "),
        ("wer_dialectat_mediu_vs_large",    "med vs large "),
    ]
    buckets = [
        (0.00, 0.10, "0-10%"),
        (0.10, 0.20, "10-20%"),
        (0.20, 0.30, "20-30%"),
        (0.30, 0.50, "30-50%"),
        (0.50, 1.00, "50-100%"),
        (1.00, 999,  ">100%"),
    ]
    print(f"\n[ Distributie WER (buckets) ]")
    print(f"  {'Bucket':<10}", end="")
    for _, lbl in wer_cols:
        print(f"  {lbl:>16}", end="")
    print()
    print(f"  {'-'*70}")
    for lo, hi, name in buckets:
        print(f"  {name:<10}", end="")
        for col, _ in wer_cols:
            v = df[col].astype(float)
            print(f"  {int(((v >= lo) & (v < hi)).sum()):>16,}", end="")
        print()

    print(f"\n[ Medii WER ]")
    for col, lbl in wer_cols:
        v = df[col].dropna().astype(float)
        if v.empty:
            print(f"  {lbl}: n=0")
            continue
        print(f"  {lbl}: avg={v.mean():.3f}  min={v.min():.3f}  max={v.max():.3f}  n={len(v):,}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Consens 3-way intre transcrieri Whisper")
    p.add_argument("--stats-only", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="Recalculeaza chiar daca coloana e populata")
    args = p.parse_args()

    if not args.stats_only:
        run(force=args.force)

    stats()


if __name__ == "__main__":
    main()
