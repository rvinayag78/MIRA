from fastapi import APIRouter

from app.routes import agents, chat, health, memories

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(agents.router)
api_router.include_router(memories.router)
api_router.include_router(chat.router)
