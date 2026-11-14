"""
VSR API — the FastAPI application.

The command brain, never the muscle: torch-free, GPU-free. Serves the
React build + JSON over the MongoDB catalog; the worker (backend/worker) does the
heavy lifting through the jobs table.

Run (works from ANY directory):
    python backend/run_api.py             # launcher, recommended
or (CWD-sensitive — repo root only):
    uvicorn api.main:app --app-dir backend --port 8000
Docs: http://localhost:8000/docs
"""

import os
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_BACKEND_DIR), str(_BACKEND_DIR / "shared")]

from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.gzip import GZipMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from api.settings import get_settings  # noqa: E402
from api.routers import (  # noqa: E402
    jobs,
    review,
    segments,
    speakers,
    stats,
    videos,
)

# Number of connections to keep warm — Dashboard loads 5 requests in parallel.


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Open the catalog once at startup (creates indexes/views if missing).
    from api.db import get_db
    from api import cache
    next(get_db())
    if os.environ.get("VSR_API_CACHE_WARM", "1") != "0":
        cache.start_warmer(get_db)   # precompute heavy payloads in background
    yield


app = FastAPI(
    title="Romanian VSR Dataset API",
    description="Catalog, review and job control for the VSR pipeline.",
    version="1.0",
    lifespan=lifespan,
)

# ~15 MB of JSON for the segment list → ~2-3 MB on the wire
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=5)

app.include_router(jobs.router)
app.include_router(review.router)
app.include_router(stats.router)
app.include_router(videos.router)
app.include_router(segments.router)
app.include_router(speakers.router)


def _catalog_location() -> str:
    from api.db import get_db
    try:
        return next(get_db()).location
    except Exception as e:
        return f"unreachable: {e}"


def _catalog_reachable() -> bool:
    from api.db import get_db
    try:
        return next(get_db()).ping()
    except Exception:
        return False


@app.get("/api/health")
def health():
    settings = get_settings()
    return {
        "ok": True,
        "git_sha": settings.git_sha,
        "catalog_db": _catalog_location(),
        "catalog_db_exists": _catalog_reachable(),
    }


# ── React build (same-origin, zero CORS) ─────────────────────────────────
_dist = get_settings().frontend_dist
if _dist.exists():
    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(_dist / "index.html")
else:
    @app.get("/", include_in_schema=False)
    def index_missing():
        return JSONResponse(
            {"error": "frontend build not found — run `npm run build` in frontend/"},
            status_code=503,
        )


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
