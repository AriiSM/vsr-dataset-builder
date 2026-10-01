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
    print("SYSTEM")
    version = sys.version_info
    report.line(version >= (3, 9), "python",
                f"{version.major}.{version.minor}.{version.micro}")

    ffmpeg, source = _ffmpeg_pair()
    report.line(bool(ffmpeg), "ffmpeg", source or "not on PATH")
    ffprobe = shutil.which("ffprobe") or (
        ffmpeg and shutil.which("ffprobe", path=str(Path(ffmpeg).parent)))
    report.line(bool(ffprobe), "ffprobe",
                "" if ffprobe else "not on PATH (download verification will fail)")

    try:
        usage = shutil.disk_usage(data_dir if data_dir.exists() else _PROJECT_ROOT)
        free_gb = usage.free / 1e9
        report.line(free_gb > 50, f"free disk on data/ ({free_gb:.0f} GB)",
                    "" if free_gb > 50 else "under 50 GB — reprocessing needs hundreds",
                    warn=free_gb > 10)
    except OSError:
        report.line(False, "free disk", "unreadable")


def check_gpu(report: Report):
    print("GPU")
    try:
        import torch
    except ImportError:
        report.line(False, "torch", "not installed (dev machine? — "
                    "worker requires the full requirements.txt)", warn=True)
        return
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        report.line(True, "CUDA", f"{name} · {vram:.1f} GB VRAM")
        report.line(vram >= 3.5, "sufficient VRAM",
                    "" if vram >= 3.5 else "under 4 GB — current config needs ~4")
    else:
        report.line(False, "CUDA", "torch installed but no GPU visible")


_WORKER_PACKAGES = [
    "torch", "torchaudio", "cv2", "whisperx", "faster_whisper", "mediapipe",
    "insightface", "onnxruntime", "sklearn", "yt_dlp", "pandas", "loguru",
]
_API_PACKAGES = ["fastapi", "uvicorn", "pandas", "yaml"]


def check_packages(report: Report):
    print("PACKAGES (worker)")
    for package in _WORKER_PACKAGES:
        try:
            importlib.import_module(package)
            report.line(True, package)
        except Exception as e:
            report.line(False, package, str(e)[:60], warn=True)
    print("PACKAGES (api)")
    # Warning, not blocking: the worker container intentionally has no API
    # packages (it has its own container); only blocking on an all-in-one machine.
    for package in _API_PACKAGES:
        try:
            importlib.import_module(package)
            report.line(True, package)
        except Exception as e:
            report.line(False, package,
                        "missing here — OK if the API runs in its own container",
                        warn=True)


def check_models(report: Report, models_dir: Path, quick: bool):
    print("MODELS (config/models.yaml)")
    if quick:
        print("  – skipped (--quick)")
        return
    try:
        from fetch_models import load_manifest, _verify, _target
    except ImportError as e:
        report.line(False, "manifest", f"fetch_models unreadable: {e}")
        return
    for name, entry in load_manifest().items():
        required = entry.get("required", False)
        if entry["kind"] == "hub":
            cache_root = models_dir / entry["path"]
            has_cache = cache_root.exists() and any(cache_root.rglob("*"))
            report.line(has_cache, f"{name} (cache {entry['path']}/)",
                        "" if has_cache else "empty — will download on fetch/first run",
                        warn=not required)
            continue
        state = _verify(_target(models_dir, entry), entry)
        report.line(state == "ok", f"{name}",
                    {"ok": "present + hash correct",
                     "unpinned": "present, hash not pinned (--pin)",
                     "mismatch": "HASH MISMATCH — file corrupted or replaced",
                     "missing": "missing — run fetch_models.py"}[state],
                    warn=(state == "unpinned") or (state == "missing" and not required))


def check_catalog(report: Report, catalog_dir: Path):
    print("CATALOG")
    db_path = catalog_dir / "dataset.db"
    if not db_path.exists():
        report.line(False, "dataset.db", f"missing at {db_path} — "
                    "run `python backend/orchestrator/cli.py init ./data`",
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
                    f"v{version} (expected v{SCHEMA_VERSION})")
        report.line(journal.lower() in ("wal", "delete"), "journal",
                    journal + ("" if journal.lower() == "wal"
                               else " (bind-mount fallback Windows — OK)"))
        report.line(True, "contents",
                    f"{overview.get('num_segments', 0)} segments · "
                    f"{overview.get('num_videos', 0)} videos")
    except Exception as e:
        report.line(False, "dataset.db", f"cannot open: {e}")


def check_config(report: Report, config_path: Path):
    print("CONFIG")
    if not config_path.exists():
        report.line(False, "config.yaml", f"missing at {config_path}")
        return None
    try:
        cfg = yaml.safe_load(config_path.read_text())
        report.line(True, "config.yaml", "parses ok")
    except yaml.YAMLError as e:
        report.line(False, "config.yaml", f"invalid YAML: {e}")
        return None
    diarization = (cfg.get("segmentation", {}) or {}).get("diarization", {}) or {}
    if diarization.get("enabled"):
        token = (diarization.get("hf_token") or "").strip()
        report.line(bool(token), "pyannote hf_token",
                    "set" if token else "EMPTY — diarization will fail loudly")
    return cfg


def check_frontend(report: Report):
    print("FRONTEND")
    dist = _PROJECT_ROOT / "frontend" / "dist" / "index.html"
    report.line(dist.exists(), "React build (frontend/dist)",
                "" if dist.exists() else "missing — run `npm run build` in frontend/",
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
    print("CACHES (redirected under models/)")
    for key, value in env.items():
        print(f"  · {key} = {value}")

    print("RESULT:", f"{GREEN} READY" if report.reds == 0
          else f"{RED} {report.reds} blocking issue(s)")
    sys.exit(0 if report.reds == 0 else 1)


if __name__ == "__main__":
    main()
