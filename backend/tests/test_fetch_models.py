"""Unit tests for fetch_models + model_env + doctor smoke.

The download machinery is exercised with file:// URLs and a synthetic
manifest — no network, no big files.

Run from the repo root:
    python backend/tests/test_fetch_models.py
"""

import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_BACKEND_DIR), str(_BACKEND_DIR / "shared"),
                str(_BACKEND_DIR / "tools")]

import yaml  # noqa: E402

import fetch_models  # noqa: E402
from vsr_shared.model_env import apply_model_env  # noqa: E402


class ShaAndVerifyTests(unittest.TestCase):
    def test_file_hash_and_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "w.pth"
            f.write_bytes(b"weights")
            digest = fetch_models.sha256_of(f)
            self.assertEqual(len(digest), 64)

            self.assertEqual(fetch_models._verify(f, {"sha256": digest}), "ok")
            self.assertEqual(fetch_models._verify(f, {"sha256": None}), "unpinned")
            self.assertEqual(fetch_models._verify(f, {"sha256": "0" * 64}),
                             "mismatch")
            self.assertEqual(
                fetch_models._verify(Path(tmp) / "nope", {"sha256": digest}),
                "missing")

    def test_tree_hash_is_order_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "model"
            (root / "sub").mkdir(parents=True)
            (root / "a.onnx").write_bytes(b"aaa")
            (root / "sub" / "b.onnx").write_bytes(b"bbb")
            first = fetch_models.sha256_of_tree(root)
            second = fetch_models.sha256_of_tree(root)
            self.assertEqual(first, second)
            (root / "a.onnx").write_bytes(b"changed")
            self.assertNotEqual(first, fetch_models.sha256_of_tree(root))


class RunFlowTests(unittest.TestCase):
    def _write_manifest(self, tmp: Path, models: dict) -> Path:
        manifest = tmp / "models.yaml"
        manifest.write_text(yaml.safe_dump({"models": models}))
        return manifest

    def test_direct_download_verify_and_pin(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "src.bin"
            source.write_bytes(b"talknet-weights")
            models_dir = tmp / "models"

            manifest = self._write_manifest(tmp, {
                "talknet_asd": {
                    "path": "talknet_asd.pth", "kind": "direct",
                    "url": source.as_uri(), "sha256": None, "required": True,
                }})
            original = fetch_models.MANIFEST_PATH
            fetch_models.MANIFEST_PATH = manifest
            try:
                # 1. fetch: downloads via file:// URL
                self.assertEqual(fetch_models.run(models_dir, False, False), 0)
                self.assertEqual(
                    (models_dir / "talknet_asd.pth").read_bytes(),
                    b"talknet-weights")
                # 2. pin: records the hash into the manifest
                self.assertEqual(fetch_models.run(models_dir, False, True), 0)
                pinned = yaml.safe_load(manifest.read_text())
                digest = pinned["models"]["talknet_asd"]["sha256"]
                self.assertEqual(len(digest), 64)
                # 3. check: passes with the pin; fails after corruption
                self.assertEqual(fetch_models.run(models_dir, True, False), 0)
                (models_dir / "talknet_asd.pth").write_bytes(b"corrupt")
                self.assertEqual(fetch_models.run(models_dir, True, False), 1)
            finally:
                fetch_models.MANIFEST_PATH = original

    def test_archive_extraction_flattens_wrapper_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            archive_src = tmp / "buffalo.zip"
            with zipfile.ZipFile(archive_src, "w") as zf:
                zf.writestr("buffalo_l/det.onnx", b"det")
                zf.writestr("buffalo_l/rec.onnx", b"rec")
            models_dir = tmp / "models"
            manifest = self._write_manifest(tmp, {
                "buffalo": {
                    "path": "insightface/models/buffalo_l", "kind": "archive",
                    "url": archive_src.as_uri(),
                    "extract_to": "insightface/models/buffalo_l",
                    "sha256": None, "required": True,
                }})
            original = fetch_models.MANIFEST_PATH
            fetch_models.MANIFEST_PATH = manifest
            try:
                self.assertEqual(fetch_models.run(models_dir, False, False), 0)
                target = models_dir / "insightface" / "models" / "buffalo_l"
                # wrapper dir flattened: files sit directly in the target
                self.assertTrue((target / "det.onnx").exists())
                self.assertTrue((target / "rec.onnx").exists())
            finally:
                fetch_models.MANIFEST_PATH = original

    def test_manual_without_url_fails_check_when_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            manifest = self._write_manifest(tmp, {
                "syncnet": {"path": "syncnet.pth", "kind": "manual",
                            "url": None, "sha256": None, "required": True}})
            original = fetch_models.MANIFEST_PATH
            fetch_models.MANIFEST_PATH = manifest
            try:
                self.assertEqual(fetch_models.run(tmp / "m", True, False), 1)
            finally:
                fetch_models.MANIFEST_PATH = original


class ModelEnvTests(unittest.TestCase):
    def test_redirects_without_overriding_user_env(self):
        import os
        with tempfile.TemporaryDirectory() as tmp:
            saved = {k: os.environ.pop(k, None)
                     for k in ("TORCH_HOME", "HF_HOME", "INSIGHTFACE_HOME")}
            try:
                os.environ["HF_HOME"] = "/custom/hf"
                applied = apply_model_env(Path(tmp))
                self.assertEqual(applied["HF_HOME"], "/custom/hf")     # respectat
                self.assertTrue(applied["TORCH_HOME"].endswith("/torch"))
            finally:
                for key, value in saved.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value


class DoctorSmokeTests(unittest.TestCase):
    def test_doctor_reports_instead_of_crashing(self):
        result = subprocess.run(
            [sys.executable,
             str(_BACKEND_DIR / "tools" / "doctor.py"), "--quick"],
            capture_output=True, text=True, timeout=60,
        )
        self.assertIn("REZULTAT:", result.stdout)
        self.assertIn(result.returncode, (0, 1))


class RealManifestTests(unittest.TestCase):
    def test_repo_manifest_parses_and_has_required_entries(self):
        manifest = fetch_models.load_manifest()
        for required in ("talknet_asd", "syncnet_v2", "insightface_buffalo_l",
                         "whisper_medium", "silero_vad"):
            self.assertIn(required, manifest)
            self.assertIn("kind", manifest[required])


if __name__ == "__main__":
    unittest.main(verbosity=2)
