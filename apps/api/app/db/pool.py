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
        # Session-level GUC so it survives asyncpg autocommit; always restore
        # before returning the connection to the pool.
        previous = await conn.fetchval("SELECT current_setting('app.current_agent_id', true)")
        try:
            if agent_id is not None:
                await conn.execute(
                    "SELECT set_config('app.current_agent_id', $1, false)",
                    str(agent_id),
                )
            yield conn
        finally:
            await conn.execute(
                "SELECT set_config('app.current_agent_id', $1, false)",
                previous or "",
            )


@asynccontextmanager
async def admin_connection() -> AsyncIterator[asyncpg.Connection]:
    """Connection without RLS agent context (for agent creation / token lookup)."""
    pool = get_pool()
    async with pool.acquire() as conn:
        previous = await conn.fetchval("SELECT current_setting('app.current_agent_id', true)")
        rls_disabled = False
        try:
            await conn.execute("SELECT set_config('app.current_agent_id', '', false)")
            try:
                await conn.execute("SET row_security = off")
                rls_disabled = True
            except asyncpg.PostgresError:
                # Non-superusers cannot disable RLS; empty current_agent_id is the admin path.
                pass
            yield conn
        finally:
            await conn.execute(
                "SELECT set_config('app.current_agent_id', $1, false)",
                previous or "",
            )
            if rls_disabled:
                try:
                    await conn.execute("SET row_security = on")
                except asyncpg.PostgresError:
                    pass
