"""
Whisper Large-v3 — Transcriere cu prompt dialectal moldovenesc
==============================================================
Rescrie transcriptele existente folosind Whisper large-v3 int8 + initial_prompt
dialectal. Rezultatele merg in coloana `whisper_dialectat_large` din DB.

Coloana sursa (referinta):  whisper_standard_mediu
Coloana destinatie:         whisper_dialectat_large  +  metrici conf
WER calculat:               wer_mediu_vs_large

Prompt:  config/dialect_prompt.txt  (editeaza liber, fara modificari de cod)

Comenzi uzuale
--------------
# 1. Valideaza prompt-ul pe 20 segmente Tier A (nu scrie in DB)
docker exec -it vsr-pipeline-worker python /app/backend/tools/whisper_dialectat_large.py --validate

# 2. Ruleaza pe tot Tier A (recomandat pentru referinta WER)
docker exec -it vsr-pipeline-worker python /app/backend/tools/whisper_dialectat_large.py --tier A

# 3. Extinde la Tier B
docker exec -it vsr-pipeline-worker python /app/backend/tools/whisper_dialectat_large.py --tier B

# Test rapid pe 10 segmente
docker exec -it vsr-pipeline-worker python /app/backend/tools/whisper_dialectat_large.py --limit 10

# Refaceaza segmentele deja procesate
docker exec -it vsr-pipeline-worker python /app/backend/tools/whisper_dialectat_large.py --tier A --force
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _BACKEND_DIR.parent
sys.path[:0] = [str(_BACKEND_DIR), str(_BACKEND_DIR / "shared")]

from loguru import logger  # noqa: E402
from vsr_shared.catalog_db import CatalogDatabase  # noqa: E402
from vsr_shared.text_normalization import clean_word  # noqa: E402
from vsr_shared.wer_utils import compute_wer  # noqa: E402

# ── Constante ────────────────────────────────────────────────────────────────
_AUDIO_DIR   = _PROJECT_ROOT / "data" / "processed"
_PROMPT_FILE = _PROJECT_ROOT / "config" / "dialect_prompt.txt"

MODEL_NAME   = "large-v3"
TARGET_COL   = "whisper_dialectat_large"
CONF_COL     = "whisper_dialectat_large_conf"
CONF_MIN_COL = "whisper_dialectat_large_conf_min"
CONF_P25_COL = "whisper_dialectat_large_conf_p25"
WER_COL      = "wer_mediu_vs_large"


# ── Audio ─────────────────────────────────────────────────────────────────────

def _ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def _to_wav(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    r = subprocess.run(
        [_ffmpeg(), "-y", "-i", str(src), "-ar", "16000", "-ac", "1", str(dst)],
        capture_output=True,
    )
    return r.returncode == 0 and dst.exists() and dst.stat().st_size > 0


def _audio(video_id: str, segment_id: str) -> Path:
    return _AUDIO_DIR / video_id / "audio" / f"{segment_id}.wav"


# ── Transcriber ───────────────────────────────────────────────────────────────

class _Transcriber:
    def __init__(self, device: str, prompt: str):
        from faster_whisper import WhisperModel
        logger.info(f"Incarc Whisper {MODEL_NAME} ({device}, int8)...")
        self._m = WhisperModel(MODEL_NAME, device=device, compute_type="int8")
        self._prompt = prompt
        logger.info("Model incarcat.")

    def transcribe(self, wav: Path) -> tuple:
        """Returneaza (text, avg_conf, min_conf, p25_conf)."""
        segs, _ = self._m.transcribe(
            str(wav), language="ro", beam_size=5, initial_prompt=self._prompt,
            word_timestamps=True,
        )
        words, confs = [], []
        for s in segs:
            for w in (s.words or []):
                c = clean_word(w.word)
                if c:
                    words.append(c)
                    confs.append(w.probability)

        text = " ".join(words)
        if not confs:
            return text, 0.0, 0.0, 0.0

        confs_s = sorted(confs)
        avg = sum(confs) / len(confs)
        mn  = confs_s[0]
        p25 = confs_s[max(0, len(confs_s) // 4 - 1)]
        return text, round(avg, 4), round(mn, 4), round(p25, 4)


# ── DB ────────────────────────────────────────────────────────────────────────

def _load_prompt() -> str:
    if not _PROMPT_FILE.exists():
        logger.error(f"Fisier prompt negasit: {_PROMPT_FILE}")
        sys.exit(1)
    p = _PROMPT_FILE.read_text(encoding="utf-8").strip()
    logger.info(f"Prompt: {p}")
    return p


def _query(db: CatalogDatabase, tier, video_ids, limit, force) -> list:
    rows = db.segments.query(
        fields=["segment_id", "video_id", "quality_tier", "whisper_standard_mediu", TARGET_COL],
        exclude_rejected=True)
    if tier:
        rows = [r for r in rows if r.get("quality_tier") == tier]
    if video_ids:
        wanted = set(video_ids)
        rows = [r for r in rows if r.get("video_id") in wanted]
    if not force:
        rows = [r for r in rows if not r.get(TARGET_COL)]
    rows.sort(key=lambda r: (r.get("quality_tier") or "", r["segment_id"]))
    if limit:
        rows = rows[:int(limit)]
    return [{"sid": r["segment_id"], "vid": r["video_id"],
             "ref": r.get("whisper_standard_mediu") or ""} for r in rows]


def _write(db, sid, text, conf, conf_min, conf_p25, wer_val):
    db.segments.update_fields(sid, {
        TARGET_COL: text, CONF_COL: conf, CONF_MIN_COL: conf_min,
        CONF_P25_COL: conf_p25,
        WER_COL: round(wer_val, 4) if wer_val is not None else None})


# ── Moduri ────────────────────────────────────────────────────────────────────

def validate(args):
    """Afiseaza comparatie pe N segmente fara a scrie in DB."""
    prompt = _load_prompt()
    db = CatalogDatabase()
    rows = _query(db, tier="A", video_ids=None,
                  limit=args.validate_n, force=True)

    if not rows:
        logger.warning("Nu s-au gasit segmente Tier A.")
        return

    device = _device(args.device)
    tr = _Transcriber(device, prompt)

    print(f"\n{'='*100}")
    print(f"VALIDATE — Whisper {MODEL_NAME} + prompt dialectal  ({len(rows)} segmente)")
    print(f"{'='*100}")
    print(f"  {'SEGMENT_ID':<32} {'WER':>5}  {'STANDARD':<38}  {'DIALECTAL'}")
    print(f"  {'-'*95}")

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "s.wav"
        for r in rows:
            ok = _to_wav(_audio(r["vid"], r["sid"]), wav)
            if not ok:
                print(f"  {r['sid']:<32}         [audio lipsa]")
                continue
            text_d, *_ = tr.transcribe(wav)
            wer_v, _ = compute_wer(r["ref"], text_d)
            ws = f"{wer_v:.2f}" if wer_v is not None else "  — "
            print(f"  {r['sid']:<32} {ws:>5}  {r['ref'][:36]:<38}  {text_d[:36]}")

    print(f"\nPrompt folosit:\n  {prompt}\n")
    print("Daca rezultatele sunt satisfacatoare, ruleaza fara --validate.")


def run(args):
    """Proceseaza segmentele si scrie rezultatele in DB."""
    prompt = _load_prompt()
    db = CatalogDatabase()
    rows = _query(db, tier=args.tier, video_ids=args.video_id,
                  limit=args.limit, force=args.force)

    if not rows:
        logger.info("Nimic de procesat.")
        return

    logger.info(f"{len(rows)} segmente de procesat cu Whisper {MODEL_NAME}...")
    device = _device(args.device)
    tr = _Transcriber(device, prompt)

    done = skipped = 0
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "s.wav"
        for r in rows:
            if not _to_wav(_audio(r["vid"], r["sid"]), wav):
                skipped += 1
                continue
            try:
                text_d, conf, conf_min, conf_p25 = tr.transcribe(wav)
            except Exception as e:
                logger.warning(f"  {r['sid']}: {e}")
                skipped += 1
                continue

            wer_v, _ = compute_wer(r["ref"], text_d)
            _write(db, r["sid"], text_d, conf, conf_min, conf_p25, wer_v)
            done += 1

            if done % 50 == 0:
                logger.info(f"  {done}/{len(rows)} gata | {skipped} sarite")

    logger.info(f"Gata: {done} procesate, {skipped} sarite.")


# ── Main ──────────────────────────────────────────────────────────────────────

def _device(val: str) -> str:
    if val != "auto":
        return val
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def main():
    p = argparse.ArgumentParser(
        description=f"Whisper {MODEL_NAME} + prompt dialectal → {TARGET_COL}"
    )
    p.add_argument("--validate", action="store_true",
                   help="Preview pe N segmente (nu scrie in DB)")
    p.add_argument("--validate-n", type=int, default=20, metavar="N")
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")
    p.add_argument("--tier", choices=["A", "B", "C"])
    p.add_argument("--video-id", nargs="*")
    p.add_argument("--limit", type=int)
    p.add_argument("--force", action="store_true",
                   help="Reprocseaza chiar daca coloana e populata")
    args = p.parse_args()

    if args.validate:
        validate(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
