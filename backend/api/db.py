"""
Database access for the API — a small per-process connection pool.

FastAPI resolves a sync dependency and runs the endpoint body in threadpool
threads that are NOT guaranteed to be the same thread, so thread-local
connections break (sqlite3.ProgrammingError: objects created in a thread...).

Instead, connections are opened with check_same_thread=False and handed out
through a pool: each request borrows one for its whole duration and returns
it afterwards, so a connection is never used by two requests at once — which
is the guarantee that makes cross-thread use safe. WAL keeps concurrent
readers happy; the rare API writes ride busy_timeout.
"""

import queue
import threading

from api.settings import get_settings
from vsr_shared.catalog_db import CatalogDatabase

_pool: "queue.SimpleQueue[CatalogDatabase]" = queue.SimpleQueue()
_pool_db_path: str | None = None
_pool_guard = threading.Lock()


def get_db():
    """FastAPI dependency (generator): borrow a catalog connection for the
    duration of one request, then return it to the pool."""
    global _pool_db_path

    settings_path = str(get_settings().db_path)
    with _pool_guard:
        # Settings changed (tests point at a different db): drop stale pool.
        if _pool_db_path != settings_path:
            while True:
                try:
                    _pool.get_nowait().close()
                except queue.Empty:
                    break
            _pool_db_path = settings_path

    try:
        db = _pool.get_nowait()
    except queue.Empty:
        db = CatalogDatabase(get_settings().db_path, check_same_thread=False)

    try:
        yield db
    finally:
        # A failed request may leave a transaction open; never return a
        # dirty connection to the pool (rollback is a no-op when clean).
        try:
            db.connection.rollback()
            _pool.put(db)
        except Exception:
            db.close()
