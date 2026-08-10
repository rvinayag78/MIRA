from fastapi import APIRouter

from app.config import get_settings
from app.schemas import HealthOut

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut)
async def health() -> HealthOut:
    settings = get_settings()
    return HealthOut(status="ok", rerank_enabled=settings.rerank_enabled)
