from __future__ import annotations

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from app.config import get_settings

_redis: ArqRedis | None = None


async def init_queue() -> ArqRedis:
    global _redis
    if _redis is None:
        _redis = await create_pool(RedisSettings.from_dsn(get_settings().redis_url))
    return _redis


async def close_queue() -> None:
    global _redis
    if _redis is not None:
        await _redis.close(close_connection_pool=True)
        _redis = None


def get_queue() -> ArqRedis:
    if _redis is None:
        raise RuntimeError("Queue not initialized")
    return _redis


async def enqueue_ingest(memory_id: str) -> None:
    redis = get_queue()
    await redis.enqueue_job("ingest_memory", memory_id)


async def enqueue_clone(agent_id: str) -> None:
    redis = get_queue()
    await redis.enqueue_job("clone_agent_voice", agent_id)
