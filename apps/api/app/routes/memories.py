from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app import queue
from app.config import get_settings
from app.db import queries
from app.db.pool import admin_connection, agent_connection
from app.deps import require_agent_token
from app.schemas import MemoryOut, TextMemoryCreate

router = APIRouter(tags=["memories"])


def _row_to_memory(row) -> MemoryOut:
    return MemoryOut(
        id=row["id"],
        agent_id=row["agent_id"],
        kind=row["kind"] if "kind" in row.keys() else "voice",
        text_content=row["text_content"] if "text_content" in row.keys() else None,
        audio_uri=row["audio_uri"],
        duration_ms=row["duration_ms"],
        status=row["status"],
        error_message=row["error_message"],
        assemblyai_transcript_id=row["assemblyai_transcript_id"],
        created_at=row["created_at"],
    )


@router.post("/memories/text", response_model=MemoryOut)
async def create_text_memory(body: TextMemoryCreate) -> MemoryOut:
    async with admin_connection() as conn:
        agent = await queries.get_agent_by_token(conn, body.agent_id, body.token)
    if agent is None:
        raise HTTPException(status_code=403, detail="Invalid agent token")

    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    async with agent_connection(body.agent_id) as conn:
        total = await queries.count_active_by_kind(conn, body.agent_id, "text")
        if total >= queries.REQUIRED_TEXT_MEMORIES:
            raise HTTPException(
                status_code=400,
                detail=f"Already have {queries.REQUIRED_TEXT_MEMORIES} text memories",
            )
        row = await queries.create_memory(
            conn,
            body.agent_id,
            kind="text",
            text_content=text,
        )

    await queue.enqueue_ingest(str(row["id"]))
    return _row_to_memory(row)


@router.post("/memories", response_model=MemoryOut)
async def upload_memory(
    agent_id: UUID = Form(...),
    token: str = Form(...),
    duration_ms: int | None = Form(default=None),
    file: UploadFile = File(...),
) -> MemoryOut:
    async with admin_connection() as conn:
        agent = await queries.get_agent_by_token(conn, agent_id, token)
    if agent is None:
        raise HTTPException(status_code=403, detail="Invalid agent token")

    async with agent_connection(agent_id) as conn:
        total = await queries.count_active_by_kind(conn, agent_id, "voice")
        if total >= queries.REQUIRED_VOICE_MEMORIES:
            raise HTTPException(
                status_code=400,
                detail=f"Already have {queries.REQUIRED_VOICE_MEMORIES} voice memories",
            )

    settings = get_settings()
    settings.audio_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "recording.webm").suffix or ".webm"
    filename = f"{uuid4()}{suffix}"
    dest = settings.audio_dir / filename
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty audio file")
    dest.write_bytes(content)

    async with agent_connection(agent_id) as conn:
        row = await queries.create_memory(
            conn,
            agent_id,
            kind="voice",
            audio_uri=str(dest),
            duration_ms=duration_ms,
        )

    await queue.enqueue_ingest(str(row["id"]))
    return _row_to_memory(row)


@router.get("/memories", response_model=list[MemoryOut])
async def list_memories(
    agent_id: UUID,
    _: UUID = Depends(require_agent_token),
) -> list[MemoryOut]:
    async with agent_connection(agent_id) as conn:
        rows = await queries.list_memories(conn, agent_id)
    return [_row_to_memory(r) for r in rows]
