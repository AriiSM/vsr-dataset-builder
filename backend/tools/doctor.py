"""
Doctor — the machine's green/red readiness report. Run it BEFORE any
pipeline run; the same checks serve as the Docker healthcheck later.

Checks (each an independent line — a missing lib is a finding, not a crash):
    system    python version, ffmpeg/ffprobe, free disk on data/
    gpu       torch import, CUDA available, VRAM total
    packages  every ML dependency the worker needs (importable or not)
    models    the config/models.yaml manifest (present + hash, via fetch_models)
    catalog   dataset.db opens, schema version, WAL mode
    config    config.yaml parses; pyannote token present when diarization on
    api       frontend build present (dist/)

Usage (from the repo root):
    python backend/tools/doctor.py
    python backend/tools/doctor.py --quick     # skip model hash verification
Exit code: 0 = ready, 1 = at least one RED finding.
"""

import argparse
import importlib
import shutil
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _BACKEND_DIR.parent
sys.path[:0] = [str(_BACKEND_DIR), str(_BACKEND_DIR / "shared"),
                str(_BACKEND_DIR / "tools")]

import yaml  # noqa: E402

GREEN, RED, YELLOW = "✓", "✗", "⚠"


class Report:
    def __init__(self):
        self.reds = 0

    def line(self, ok, label, detail="", warn=False):
        if ok:
            print(f"  {GREEN} {label}" + (f" — {detail}" if detail else ""))
        elif warn:
            print(f"  {YELLOW} {label}" + (f" — {detail}" if detail else ""))
        else:
            self.reds += 1
            print(f"  {RED} {label}" + (f" — {detail}" if detail else ""))


def _ffmpeg_pair():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe(), "imageio-ffmpeg (bundled)"
    except Exception:
        pass
    found = shutil.which("ffmpeg")
    return found, "PATH" if found else None


def check_system(report: Report, data_dir: Path):
    print("SISTEM")
    version = sys.version_info
    report.line(version >= (3, 9), "python",
                f"{version.major}.{version.minor}.{version.micro}")

    ffmpeg, source = _ffmpeg_pair()
    report.line(bool(ffmpeg), "ffmpeg", source or "nu e pe PATH")
    ffprobe = shutil.which("ffprobe") or (
        ffmpeg and shutil.which("ffprobe", path=str(Path(ffmpeg).parent)))
    report.line(bool(ffprobe), "ffprobe",
                "" if ffprobe else "nu e pe PATH (verificarea descărcărilor pică)")

    try:
        usage = shutil.disk_usage(data_dir if data_dir.exists() else _PROJECT_ROOT)
        free_gb = usage.free / 1e9
        report.line(free_gb > 50, f"disc liber pe data/ ({free_gb:.0f} GB)",
                    "" if free_gb > 50 else "sub 50 GB — reprocesarea cere sute",
                    warn=free_gb > 10)
    except OSError:
        report.line(False, "disc liber", "necitibil")


def check_gpu(report: Report):
    print("GPU")
    try:
        import torch
    except ImportError:
        report.line(False, "torch", "neinstalat (mașină de dezvoltare? — "
                    "workerul cere requirements.txt complet)", warn=True)
        return
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        report.line(True, "CUDA", f"{name} · {vram:.1f} GB VRAM")
        report.line(vram >= 3.5, "VRAM suficient",
                    "" if vram >= 3.5 else "sub 4 GB — config-ul actual cere ~4")
    else:
        report.line(False, "CUDA", "torch instalat, dar niciun GPU vizibil")


_WORKER_PACKAGES = [
    "torch", "torchaudio", "cv2", "whisperx", "faster_whisper", "mediapipe",
    "insightface", "onnxruntime", "sklearn", "yt_dlp", "pandas", "loguru",
]
_API_PACKAGES = ["fastapi", "uvicorn", "pandas", "yaml"]


def check_packages(report: Report):
    print("PACHETE (worker)")
    for package in _WORKER_PACKAGES:
        try:
            importlib.import_module(package)
            report.line(True, package)
        except Exception as e:
            report.line(False, package, str(e)[:60], warn=True)
    print("PACHETE (api)")
    # Avertisment, nu blocant: în containerul de worker API-ul lipsește prin
    # design (are containerul lui); blocant e doar pe o mașină all-in-one.
    for package in _API_PACKAGES:
        try:
            importlib.import_module(package)
            report.line(True, package)
        except Exception as e:
            report.line(False, package,
                        "lipsă aici — OK dacă API-ul rulează în containerul lui",
                        warn=True)


def check_models(report: Report, models_dir: Path, quick: bool):
    print("MODELE (config/models.yaml)")
    if quick:
        print("  – sărit (--quick)")
        return
    try:
        from fetch_models import load_manifest, _verify, _target
    except ImportError as e:
        report.line(False, "manifest", f"fetch_models necitibil: {e}")
        return
    for name, entry in load_manifest().items():
        required = entry.get("required", False)
        if entry["kind"] == "hub":
            cache_root = models_dir / entry["path"]
            has_cache = cache_root.exists() and any(cache_root.rglob("*"))
            report.line(has_cache, f"{name} (cache {entry['path']}/)",
                        "" if has_cache else "gol — se descarcă la fetch/prima rulare",
                        warn=not required)
            continue
        state = _verify(_target(models_dir, entry), entry)
        report.line(state == "ok", f"{name}",
                    {"ok": "prezent + hash corect",
                     "unpinned": "prezent, hash nefixat (--pin)",
                     "mismatch": "HASH DIFERIT — fișier corupt/înlocuit",
                     "missing": "lipsă — rulează fetch_models.py"}[state],
                    warn=(state == "unpinned") or (state == "missing" and not required))


def check_catalog(report: Report, catalog_dir: Path):
    print("CATALOG")
    db_path = catalog_dir / "dataset.db"
    if not db_path.exists():
        report.line(False, "dataset.db", f"lipsă la {db_path} — "
                    "rulează `python backend/orchestrator/cli.py init ./data`",
                    warn=True)
        return
    try:
        from vsr_shared.catalog_db import CatalogDatabase, SCHEMA_VERSION
        db = CatalogDatabase(db_path)
        version = db.connection.execute("PRAGMA user_version").fetchone()[0]
        journal = db.connection.execute("PRAGMA journal_mode").fetchone()[0]
        overview = dict(db.connection.execute(
            "SELECT * FROM dataset_overview").fetchone() or {})
        db.close()
        report.line(version == SCHEMA_VERSION, "schema",
                    f"v{version} (așteptat v{SCHEMA_VERSION})")
        report.line(journal.lower() == "wal", "journal", journal)
        report.line(True, "conținut",
                    f"{overview.get('num_segments', 0)} segmente · "
                    f"{overview.get('num_videos', 0)} video-uri")
    except Exception as e:
        report.line(False, "dataset.db", f"nu se deschide: {e}")


def check_config(report: Report, config_path: Path):
    print("CONFIG")
    if not config_path.exists():
        report.line(False, "config.yaml", f"lipsă la {config_path}")
        return None
    try:
        cfg = yaml.safe_load(config_path.read_text())
        report.line(True, "config.yaml", "parsează")
    except yaml.YAMLError as e:
        report.line(False, "config.yaml", f"YAML invalid: {e}")
        return None
    diarization = (cfg.get("segmentation", {}) or {}).get("diarization", {}) or {}
    if diarization.get("enabled"):
        token = (diarization.get("hf_token") or "").strip()
        report.line(bool(token), "pyannote hf_token",
                    "setat" if token else "GOL — diarizarea va eșua zgomotos")
    return cfg


def check_frontend(report: Report):
    print("FRONTEND")
    dist = _PROJECT_ROOT / "frontend" / "dist" / "index.html"
    report.line(dist.exists(), "build React (frontend/dist)",
                "" if dist.exists() else "lipsă — rulează `npm run build` în frontend/",
                warn=True)


def main():
    parser = argparse.ArgumentParser(description="VSR machine readiness report")
    parser.add_argument("--config", type=Path,
                        default=_PROJECT_ROOT / "config" / "config.yaml")
    parser.add_argument("--quick", action="store_true",
                        help="skip model hash verification (slow on big files)")
    args = parser.parse_args()

    report = Report()
    cfg = None
    print(f"VSR doctor — {_PROJECT_ROOT}")

    cfg = check_config(report, args.config) or {}
    paths = cfg.get("paths", {})
    base_dir = Path(paths.get("base_dir", "./data"))
    if not base_dir.is_absolute():
        base_dir = _PROJECT_ROOT / base_dir
    models_dir = Path(paths.get("models_dir", "./models"))
    if not models_dir.is_absolute():
        models_dir = _PROJECT_ROOT / models_dir
    catalog_dir = Path(paths.get("catalog_dir", base_dir / "catalog"))
    if not catalog_dir.is_absolute():
        catalog_dir = _PROJECT_ROOT / catalog_dir

    check_system(report, base_dir)
    check_gpu(report)
    check_packages(report)
    check_models(report, models_dir, args.quick)
    check_catalog(report, catalog_dir)
    check_frontend(report)

    from vsr_shared.model_env import apply_model_env
    env = apply_model_env(models_dir)
    print("CACHE-URI (redirecționate sub models/)")
    for key, value in env.items():
        print(f"  · {key} = {value}")

    print("REZULTAT:", f"{GREEN} PREGĂTIT" if report.reds == 0
          else f"{RED} {report.reds} problemă(e) blocante")
    sys.exit(0 if report.reds == 0 else 1)


if __name__ == "__main__":
    main()
