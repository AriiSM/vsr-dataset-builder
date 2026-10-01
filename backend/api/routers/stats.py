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
    return stats_service.get_stats(db)


@router.get("/stats/distributions")
def stats_distributions(db: CatalogDatabase = Depends(get_db)):
    return stats_service.get_distributions(db)


@router.get("/vocabulary")
def vocabulary(db: CatalogDatabase = Depends(get_db)):
    return stats_service.get_vocabulary(db)
