"""
API settings — paths resolved once, from config.yaml + environment.

The API is deliberately torch-free: it needs only WHERE things are
(catalog db, processed media, logs, frontend build), never how the
pipeline computes them. Environment overrides (VSR_*) exist for Docker,
where the volumes mount at fixed locations.
"""

import os
from functools import lru_cache
from pathlib import Path

import yaml

_BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = _BACKEND_DIR.parent


class Settings:
    def __init__(self, config_path: Path = None):
        self.config_path = Path(
            config_path
            or os.environ.get("VSR_CONFIG", PROJECT_ROOT / "config" / "config.yaml")
        )
        cfg = {}
        if self.config_path.exists():
            cfg = yaml.safe_load(self.config_path.read_text()) or {}
        paths = cfg.get("paths", {})

        base_dir = self._resolve(paths.get("base_dir", "./data"))
        self.catalog_dir = Path(os.environ.get(
            "VSR_CATALOG_DIR",
            self._resolve(paths.get("catalog_dir", base_dir / "catalog"))))
        self.processed_dir = Path(os.environ.get(
            "VSR_PROCESSED_DIR",
            self._resolve(paths.get("processed_dir", base_dir / "processed"))))
        self.logs_dir = Path(os.environ.get("VSR_LOGS_DIR", base_dir / "logs"))
        self.cache_dir = Path(os.environ.get("VSR_CACHE_DIR", base_dir / "cache"))
        self.frontend_dist = Path(os.environ.get(
            "VSR_FRONTEND_DIST", PROJECT_ROOT / "frontend" / "dist"))
        self.git_sha = os.environ.get("GIT_SHA", "")

    @staticmethod
    def _resolve(path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def db_path(self) -> Path:
        return self.catalog_dir / "dataset.db"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
