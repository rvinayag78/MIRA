from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.db.pool import close_pool, init_pool
from app.queue import close_queue, init_queue
from app.routes import api_router


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    settings.audio_dir.mkdir(parents=True, exist_ok=True)
    settings.tts_dir.mkdir(parents=True, exist_ok=True)
    await init_pool()
    try:
        await init_queue()
    except Exception:
        # Allow API to boot without Redis for /health; enqueue will fail clearly
        pass
    yield
    await close_queue()
    await close_pool()


app = FastAPI(title="Ovyu Demo Agent API", version="0.1.0", lifespan=lifespan)
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router)
