"""
Stats routes — dashboard aggregates, distributions, vocabulary.

Response shapes match the Flask endpoints 1:1 (the React Stats tab is the
contract); the data comes from the MongoDB catalog through stats_service, served
from api.cache (recomputed only when the catalog changes).
"""

from fastapi import APIRouter, Depends

from api.cache import cached_response, register
from api.db import get_db
from api import stats_service
from vsr_shared.catalog_db import CatalogDatabase

router = APIRouter(prefix="/api", tags=["stats"])


register("stats")(stats_service.get_stats)
register("distributions")(stats_service.get_distributions)
register("vocabulary")(stats_service.get_vocabulary)


@router.get("/stats")
def stats(db: CatalogDatabase = Depends(get_db)):
    return cached_response("stats", db)


@router.get("/stats/distributions")
def stats_distributions(db: CatalogDatabase = Depends(get_db)):
    return cached_response("distributions", db)


@router.get("/vocabulary")
def vocabulary(db: CatalogDatabase = Depends(get_db)):
    return cached_response("vocabulary", db)
