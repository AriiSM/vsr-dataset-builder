"""
Stats routes — dashboard aggregates, distributions, vocabulary.

Response shapes match the Flask endpoints 1:1 (the React Stats tab is the
contract); the data now comes from dataset.db through stats_service.
"""

from fastapi import APIRouter, Depends

from api.db import get_db
from api import stats_service
from vsr_shared.catalog_db import CatalogDatabase

router = APIRouter(prefix="/api", tags=["stats"])


@router.get("/stats")
def stats(db: CatalogDatabase = Depends(get_db)):
    videos = stats_service.stats_videos(db)
    payload = {
        "videos": videos,
        "segments": stats_service.stats_segments(db),
    }
    if "total_duration_h" in videos:
        payload["videos"]["total_duration_s"] = round(
            videos["total_duration_h"] * 3600, 2)
    return payload


@router.get("/stats/distributions")
def stats_distributions(db: CatalogDatabase = Depends(get_db)):
    return stats_service.distributions(db)


@router.get("/vocabulary")
def vocabulary(db: CatalogDatabase = Depends(get_db)):
    return stats_service.vocabulary(db)
