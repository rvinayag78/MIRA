from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from app import queue
from app.db import queries
from app.db.pool import admin_connection, agent_connection
from app.deps import require_agent_token
from app.schemas import AgentCreate, AgentOut

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("", response_model=AgentOut)
async def create_agent(body: AgentCreate) -> AgentOut:
    async with admin_connection() as conn:
        created = await queries.create_agent(conn, body.display_name.strip())
    agent_id = created["id"]
    return AgentOut(
        id=agent_id,
        display_name=created["display_name"],
        elevenlabs_voice_id=created["elevenlabs_voice_id"],
        status=created["status"],
        created_at=created["created_at"],
        share_token=created["share_token"],
        keeper_url=f"/k/{agent_id}?token={created['share_token']}",
        indexed_memories=0,
    )


@router.get("/{agent_id}", response_model=AgentOut)
async def get_agent(
    agent_id: UUID,
    _: UUID = Depends(require_agent_token),
) -> AgentOut:
    async with agent_connection(agent_id) as conn:
        agent = await queries.get_agent(conn, agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")
        count = await queries.count_indexed_memories(conn, agent_id)
    # Need token again for keeper_url — omit full token on GET; client already has it
    return AgentOut(
        id=agent["id"],
        display_name=agent["display_name"],
        elevenlabs_voice_id=agent["elevenlabs_voice_id"],
        status=agent["status"],
        created_at=agent["created_at"],
        indexed_memories=count,
        keeper_url=f"/k/{agent_id}",
    )


@router.post("/{agent_id}/clone-voice", response_model=AgentOut)
async def clone_voice(
    agent_id: UUID,
    _: UUID = Depends(require_agent_token),
) -> AgentOut:
    async with agent_connection(agent_id) as conn:
        agent = await queries.get_agent(conn, agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")
        count = await queries.count_indexed_memories(conn, agent_id)
        if count < 1:
            raise HTTPException(status_code=400, detail="Need at least one indexed memory")
        await queries.set_agent_status(conn, agent_id, "cloning")

    await queue.enqueue_clone(str(agent_id))

    async with agent_connection(agent_id) as conn:
        agent = await queries.get_agent(conn, agent_id)
        count = await queries.count_indexed_memories(conn, agent_id)
    assert agent is not None
    return AgentOut(
        id=agent["id"],
        display_name=agent["display_name"],
        elevenlabs_voice_id=agent["elevenlabs_voice_id"],
        status=agent["status"],
        created_at=agent["created_at"],
        indexed_memories=count,
        keeper_url=f"/k/{agent_id}",
    )
