from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import get_settings
from app.db import queries
from app.db.pool import admin_connection, agent_connection
from app.schemas import ChatRequest, ChatResponse, CitationOut
from app.services import elevenlabs, generate
from app.services.retrieve import hybrid_retrieve

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest) -> ChatResponse:
    async with admin_connection() as conn:
        agent = await queries.get_agent_by_token(conn, body.agent_id, body.token)
    if agent is None:
        raise HTTPException(status_code=403, detail="Invalid agent token")

    agent_id = agent["id"]

    async with agent_connection(agent_id) as conn:
        session_id = body.session_id or await queries.create_session(conn, agent_id)
        await queries.insert_message(
            conn,
            session_id=session_id,
            agent_id=agent_id,
            role="user",
            content=body.message,
        )

        chunks = await hybrid_retrieve(conn, agent_id, body.message)
        result = await generate.answer_question(body.message, chunks)

        if result.citations and not generate.check_citations_valid(result, chunks):
            result = generate.GroundedAnswer(
                answer=generate.REFUSAL,
                citations=[],
                confidence=0.0,
                refused=True,
                intent=result.intent,
            )

        audio_url = None
        audio_uri = None
        if body.speak and not result.refused and agent["elevenlabs_voice_id"]:
            try:
                path = await elevenlabs.text_to_speech(
                    agent["elevenlabs_voice_id"], result.answer
                )
                audio_uri = str(path)
                audio_url = f"/tts/{path.name}"
            except elevenlabs.ElevenLabsError:
                audio_url = None

        await queries.insert_message(
            conn,
            session_id=session_id,
            agent_id=agent_id,
            role="assistant",
            content=result.answer,
            citations=[c.__dict__ for c in result.citations],
            audio_uri=audio_uri,
        )

    return ChatResponse(
        session_id=session_id,
        answer=result.answer,
        citations=[CitationOut(chunk_id=c.chunk_id, quote=c.quote) for c in result.citations],
        confidence=result.confidence,
        refused=result.refused,
        intent=result.intent,
        audio_url=audio_url,
    )


@router.get("/tts/{filename}")
async def get_tts(filename: str) -> FileResponse:
    settings = get_settings()
    # prevent path traversal
    safe = Path(filename).name
    path = settings.tts_dir / safe
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(path, media_type="audio/mpeg")
