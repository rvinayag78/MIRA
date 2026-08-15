from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator
from uuid import UUID

import asyncpg

from app.config import get_settings

_pool: asyncpg.Pool | None = None


async def init_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        settings = get_settings()
        kwargs: dict = {
            "dsn": settings.asyncpg_dsn,
            "min_size": 1,
            "max_size": 10,
            "command_timeout": 60,
        }
        if settings.asyncpg_ssl is not None:
            kwargs["ssl"] = settings.asyncpg_ssl
        _pool = await asyncpg.create_pool(**kwargs)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database pool not initialized")
    return _pool


@asynccontextmanager
async def agent_connection(agent_id: UUID | None = None) -> AsyncIterator[asyncpg.Connection]:
    """Acquire a connection and optionally set RLS agent context."""
    pool = get_pool()
    async with pool.acquire() as conn:
        # Bypass RLS for demo bootstrap when no agent set; when set, enforce.
        if agent_id is not None:
            await conn.execute(
                "SELECT set_config('app.current_agent_id', $1, true)",
                str(agent_id),
            )
            # Force RLS even for table owner in demo
            await conn.execute("SET LOCAL row_security = on")
        yield conn


@asynccontextmanager
async def admin_connection() -> AsyncIterator[asyncpg.Connection]:
    """Connection without RLS agent context (for agent creation / token lookup)."""
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute("SET LOCAL row_security = off")
        yield conn
