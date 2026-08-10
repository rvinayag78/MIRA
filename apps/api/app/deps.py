from __future__ import annotations

from uuid import UUID

from fastapi import Header, HTTPException, Query

from app.db import queries
from app.db.pool import admin_connection


async def require_agent_token(
    agent_id: UUID,
    x_agent_token: str | None = Header(default=None, alias="X-Agent-Token"),
    token: str | None = Query(default=None),
) -> UUID:
    share = x_agent_token or token
    if not share:
        raise HTTPException(status_code=401, detail="Missing agent token")
    async with admin_connection() as conn:
        agent = await queries.get_agent_by_token(conn, agent_id, share)
    if agent is None:
        raise HTTPException(status_code=403, detail="Invalid agent token")
    return agent["id"]
