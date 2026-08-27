"""
Fetch + verify every model weight from the config/models.yaml manifest.

Idempotent: present-and-verified files are skipped. Direct/archive entries
download with plain urllib (no ML libs needed); hub entries pre-download
through their library and run only where it is installed (the GPU machine).

Usage (from the repo root):
    python backend/tools/fetch_models.py              # fetch what's missing
    python backend/tools/fetch_models.py --check      # verify only, no downloads
    python backend/tools/fetch_models.py --only talknet_asd
    python backend/tools/fetch_models.py --pin        # record sha256 of present
                                                      #   files into the manifest
Exit code: 0 = everything required present+verified, 1 otherwise.
"""

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _BACKEND_DIR.parent
sys.path[:0] = [str(_BACKEND_DIR), str(_BACKEND_DIR / "shared")]

import yaml  # noqa: E402

from vsr_shared.model_env import apply_model_env  # noqa: E402

MANIFEST_PATH = _PROJECT_ROOT / "config" / "models.yaml"


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_of_tree(path: Path) -> str:
    """Deterministic hash of a directory (sorted relative paths + contents)."""
    digest = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(file.relative_to(path)).encode())
        digest.update(sha256_of(file).encode())
    return digest.hexdigest()


def load_manifest() -> dict:
    return yaml.safe_load(MANIFEST_PATH.read_text())["models"]


def _target(models_dir: Path, entry: dict) -> Path:
    return models_dir / entry["path"]


def _verify(target: Path, entry: dict) -> str:
    """'ok' | 'unpinned' | 'mismatch' | 'missing'."""
    if not target.exists():
        return "missing"
    pinned = entry.get("sha256")
    if not pinned:
        return "unpinned"
    actual = sha256_of_tree(target) if target.is_dir() else sha256_of(target)
    return "ok" if actual == pinned else "mismatch"


def _download(url: str, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    print(f"    descarc {url}")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {done / total * 100:5.1f}% "
                      f"({done >> 20}/{total >> 20} MB)", end="")
    print()
    tmp.replace(destination)


def _fetch_direct(models_dir: Path, name: str, entry: dict) -> bool:
    target = _target(models_dir, entry)
    if entry.get("url") is None:
        print(f"  ✗ {name}: fără URL — copiază manual fișierul la {target}"
              f" (sau urcă-l în repo-ul tău HF și pune link-ul în models.yaml)")
        return False
    if entry["kind"] == "archive":
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "archive.zip"
            _download(entry["url"], archive)
            extract_to = models_dir / entry.get("extract_to", entry["path"])
            if extract_to.exists():
                shutil.rmtree(extract_to)
            extract_to.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(extract_to)
            # A zip that wraps everything in one folder gets flattened.
            children = list(extract_to.iterdir())
            if len(children) == 1 and children[0].is_dir():
                for item in children[0].iterdir():
                    shutil.move(str(item), extract_to)
                children[0].rmdir()
    else:
        _download(entry["url"], target)
    return True


def _fetch_hub(name: str, entry: dict) -> bool:
    hub = entry.get("hub")
    try:
        if hub == "faster_whisper":
            try:
                from faster_whisper.utils import download_model
            except ImportError:
                print(f"  – {name}: faster-whisper neinstalat aici — "
                      "rulează pe mașina de procesare")
                return False
            print(f"    pre-descarc whisper {entry['model_name']} …")
            download_model(entry["model_name"])
        elif hub == "torch_hub":
            try:
                import torch
            except ImportError:
                print(f"  – {name}: torch neinstalat aici — "
                      "rulează pe mașina de procesare")
                return False
            print(f"    pre-descarc {entry['repo']} …")
            torch.hub.load(entry["repo"], "silero_vad", trust_repo=True)
        elif hub == "pyannote":
            print(f"  – {name}: gated pe HF (token) — se descarcă la prima"
                  " rulare a pipeline-ului; doctor verifică token-ul")
            return True
        else:
            print(f"  ✗ {name}: hub necunoscut '{hub}'")
            return False
        return True
    except Exception as e:
        print(f"  ✗ {name}: pre-download eșuat: {e}")
        return False


def run(models_dir: Path, check_only: bool, pin: bool, only: str = None) -> int:
    manifest = load_manifest()
    apply_model_env(models_dir)

    problems = 0
    pinned_updates = {}
    for name, entry in manifest.items():
        if only and only != name:
            continue
        required = entry.get("required", False)
        marker = "!" if required else "·"

        if entry["kind"] == "hub":
            if check_only or pin:
                print(f"  – {name} [{marker}] hub — verificat de doctor prin cache")
            else:
                _fetch_hub(name, entry)
            continue

        target = _target(models_dir, entry)
        state = _verify(target, entry)

        if state == "ok":
            print(f"  ✓ {name}: prezent + hash corect")
            continue
        if state == "unpinned":
            actual = (sha256_of_tree(target) if target.is_dir()
                      else sha256_of(target))
            if pin:
                pinned_updates[name] = actual
                print(f"  ✓ {name}: prezent — hash fixat {actual[:16]}…")
            else:
                print(f"  ⚠ {name}: prezent, dar hash NEFIXAT"
                      f" (rulează --pin ca să-l înregistrezi)")
            continue
        if state == "mismatch":
            problems += 1
            print(f"  ✗ {name}: HASH DIFERIT de manifest — fișier corupt sau"
                  " înlocuit; șterge-l și re-rulează fetch")
            continue

        # missing
        if check_only or pin:
            problems += required
            print(f"  {'✗' if required else '–'} {name}: lipsă ({target})")
            continue
        if _fetch_direct(models_dir, name, entry):
            state = _verify(target, entry)
            if state == "mismatch":
                problems += 1
                print(f"  ✗ {name}: hash diferit DUPĂ download")
            else:
                print(f"  ✓ {name}: descărcat"
                      + (" (hash nefixat — rulează --pin)" if state == "unpinned" else ""))
        else:
            problems += required

    if pinned_updates:
        text = MANIFEST_PATH.read_text()
        manifest_full = yaml.safe_load(text)
        for name, digest in pinned_updates.items():
            manifest_full["models"][name]["sha256"] = digest
        MANIFEST_PATH.write_text(
            yaml.safe_dump(manifest_full, sort_keys=False, allow_unicode=True))
        print(f"manifest actualizat: {len(pinned_updates)} hash(uri) fixate")

    return 0 if problems == 0 else 1


def main():
    parser = argparse.ArgumentParser(description="Fetch + verify model weights")
    parser.add_argument("--models-dir", type=Path,
                        default=_PROJECT_ROOT / "models")
    parser.add_argument("--check", action="store_true",
                        help="verify only — no downloads")
    parser.add_argument("--pin", action="store_true",
                        help="record sha256 of present files into models.yaml")
    parser.add_argument("--only", help="one manifest entry by name")
    args = parser.parse_args()
    sys.exit(run(args.models_dir, args.check, args.pin, args.only))


if __name__ == "__main__":
    main()
