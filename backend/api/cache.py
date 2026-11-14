"""
Response cache for the heavy read-only endpoints (segment list, stats,
distributions, vocabulary).

These endpoints read the WHOLE catalog (60k segments / 780k words), so an
index cannot speed them up; what costs is the aggregation, the pandas work
and the JSON encoding of ~15 MB. The data only changes when the pipeline or a
reviewer writes, so each payload is computed once and kept as ready-to-send
JSON bytes, keyed by the catalog data version (catalog_db bumps it on every
write).

Staleness policy (stale-while-revalidate):
  - version unchanged                 → cached bytes (milliseconds)
  - version changed, cache younger
    than VSR_API_CACHE_STALE_S (120 s) → cached bytes now + refresh in background
                                        (the next request gets fresh data)
  - otherwise                          → recompute before answering
A background warmer computes everything at API startup and refreshes stale
payloads every VSR_API_CACHE_WARM_S (60 s) — while the pipeline writes a clip
every few seconds this bounds the recompute load instead of chasing every write.
Tests set VSR_API_CACHE_STALE_S=0 (always fresh).
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict

from fastapi import Response
from fastapi.encoders import jsonable_encoder

logger = logging.getLogger("vsr.api.cache")

_STALE_S = float(os.environ.get("VSR_API_CACHE_STALE_S", "120"))
_WARM_EVERY_S = float(os.environ.get("VSR_API_CACHE_WARM_S", "60"))


@dataclass
class _Entry:
    version: int
    body: bytes
    computed_at: float


_builders: Dict[str, Callable] = {}
_entries: Dict[str, _Entry] = {}
_locks: Dict[str, threading.Lock] = {}
_refreshing: set = set()
_guard = threading.Lock()


def register(name: str):
    """Decorator: fn(db) -> JSON-able payload, served through the cache."""
    def deco(fn):
        _builders[name] = fn
        _locks[name] = threading.Lock()
        return fn
    return deco


def _encode(payload) -> bytes:
    # same encoding as FastAPI's default JSONResponse
    return json.dumps(jsonable_encoder(payload), ensure_ascii=False,
                      allow_nan=False, indent=None, separators=(",", ":")).encode("utf-8")


def _key(name: str, db) -> str:
    return f"{db.db_name}:{name}"      # one cache per catalog database


def _refresh(name: str, db) -> _Entry:
    with _locks[name]:
        version = db.data_version()
        entry = _entries.get(_key(name, db))
        if entry is not None and entry.version == version:
            return entry               # someone else refreshed meanwhile
        started = time.time()
        entry = _Entry(version, _encode(_builders[name](db)), time.time())
        _entries[_key(name, db)] = entry
        logger.info(f"cache {name}: rebuilt in {time.time() - started:.1f}s "
                    f"({len(entry.body) / 1e6:.1f} MB, v{version})")
        return entry


def _refresh_in_background(name: str, db):
    key = _key(name, db)
    with _guard:
        if key in _refreshing:
            return
        _refreshing.add(key)

    def run():
        try:
            _refresh(name, db)
        except Exception as e:  # never kill the request path
            logger.warning(f"cache {name}: background refresh failed: {e}")
        finally:
            with _guard:
                _refreshing.discard(key)
    threading.Thread(target=run, daemon=True, name=f"cache-{name}").start()


def cached_response(name: str, db) -> Response:
    entry = _entries.get(_key(name, db))
    version = db.data_version()
    if entry is None:
        entry = _refresh(name, db)
    elif entry.version != version:
        if time.time() - entry.computed_at <= _STALE_S:
            _refresh_in_background(name, db)
        else:
            entry = _refresh(name, db)
    return Response(content=entry.body, media_type="application/json")


def start_warmer(get_db: Callable):
    """Compute every payload once now, then keep them fresh."""
    def loop():
        while True:
            try:
                db = next(get_db())
                version = db.data_version()
                for name in list(_builders):
                    entry = _entries.get(_key(name, db))
                    if entry is None or entry.version != version:
                        _refresh(name, db)
            except Exception as e:
                logger.warning(f"cache warmer: {e}")
            time.sleep(_WARM_EVERY_S)
    threading.Thread(target=loop, daemon=True, name="cache-warmer").start()


def clear():
    """Drop every cached payload (tests)."""
    _entries.clear()
