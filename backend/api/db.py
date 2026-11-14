"""
Database access for the API.

MongoDB's client is thread-safe and pools connections itself, so every
request can share one CatalogDatabase per catalog location (recreated only
when the settings point somewhere else, e.g. tests).
"""

import threading

from api.settings import get_settings
from vsr_shared.catalog_db import CatalogDatabase

_db: CatalogDatabase | None = None
_db_key: str | None = None
_guard = threading.Lock()


def get_db():
    """FastAPI dependency (generator): the shared catalog handle."""
    global _db, _db_key
    key = str(get_settings().db_path)
    with _guard:
        if _db is None or _db_key != key:
            _db = CatalogDatabase(get_settings().db_path)
            _db_key = key
        db = _db
    yield db
