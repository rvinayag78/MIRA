from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from app import queue
from app.db import queries
from app.db.pool import admin_connection, agent_connection
from app.deps import require_agent_token
from app.schemas import AgentCreate, AgentOut

router = APIRouter(prefix="/agents", tags=["agents"])


def _agent_out(agent, *, share_token: str | None = None, ready: dict | None = None) -> AgentOut:
    ready = ready or {}
    agent_id = agent["id"]
    return AgentOut(
        id=agent_id,
        display_name=agent["display_name"],
        elevenlabs_voice_id=agent["elevenlabs_voice_id"],
        status=agent["status"],
        created_at=agent["created_at"],
        share_token=share_token,
        keeper_url=(
            f"/k/{agent_id}?token={share_token}" if share_token else f"/k/{agent_id}"
        ),
        indexed_memories=ready.get("text_indexed", 0) + ready.get("voice_indexed", 0),
        text_indexed=ready.get("text_indexed", 0),
        voice_indexed=ready.get("voice_indexed", 0),
        text_required=ready.get("text_required", 3),
        voice_required=ready.get("voice_required", 3),
        ready_for_keeper=ready.get("ready_for_keeper", False),
    )


@router.post("", response_model=AgentOut)
async def create_agent(body: AgentCreate) -> AgentOut:
    async with admin_connection() as conn:
        created = await queries.create_agent(conn, body.display_name.strip())
        ready = await queries.readiness(conn, created["id"])
    return _agent_out(created, share_token=created["share_token"], ready=ready)


@router.get("/{agent_id}", response_model=AgentOut)
async def get_agent(
    agent_id: UUID,
    _: UUID = Depends(require_agent_token),
) -> AgentOut:
    async with agent_connection(agent_id) as conn:
        agent = await queries.get_agent(conn, agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")
        ready = await queries.readiness(conn, agent_id)
    return _agent_out(agent, ready=ready)


@router.post("/{agent_id}/clone-voice", response_model=AgentOut)
async def clone_voice(
    agent_id: UUID,
    _: UUID = Depends(require_agent_token),
) -> AgentOut:
    async with agent_connection(agent_id) as conn:
        agent = await queries.get_agent(conn, agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")
        voice_n = await queries.count_indexed_by_kind(conn, agent_id, "voice")
        if voice_n < 1:
            raise HTTPException(status_code=400, detail="Need at least one indexed voice memory")
        await queries.set_agent_status(conn, agent_id, "cloning")

    await queue.enqueue_clone(str(agent_id))

    async with agent_connection(agent_id) as conn:
        agent = await queries.get_agent(conn, agent_id)
        ready = await queries.readiness(conn, agent_id)
    assert agent is not None
    return _agent_out(agent, ready=ready)
