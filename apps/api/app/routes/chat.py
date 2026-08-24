from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import get_settings
from app.db import queries
from app.db.pool import admin_connection, agent_connection
from app.schemas import ChatRequest, ChatResponse, CitationOut
from app.services import elevenlabs, generate
from app.services.evidence import build_evidence_package
from app.services.retrieve import retrieve_substance
from app.services.voyage import VoyageError

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest) -> ChatResponse:
    async with admin_connection() as conn:
        agent = await queries.get_agent_by_token(conn, body.agent_id, body.token)
    if agent is None:
        raise HTTPException(status_code=403, detail="Invalid agent token")

    agent_id = agent["id"]

    async with agent_connection(agent_id) as conn:
        ready = await queries.readiness(conn, agent_id)
        if not ready["ready_for_keeper"]:
            raise HTTPException(
                status_code=403,
                detail=(
                    f"Keeper chat unlocks after {ready['text_required']} text and "
                    f"{ready['voice_required']} voice memories "
                    f"(have {ready['text_indexed']} text, {ready['voice_indexed']} voice)."
                ),
            )
        session_id = body.session_id or await queries.create_session(conn, agent_id)
        await queries.insert_message(
            conn,
            session_id=session_id,
            agent_id=agent_id,
            role="user",
            content=body.message,
        )
        rows = await queries.list_session_messages(conn, session_id, limit=9)
        history = [{"role": r["role"], "content": r["content"]} for r in rows]
        if history and history[-1]["role"] == "user":
            history = history[:-1]

    try:
        retrieve_q = generate.expand_retrieval_query(body.message, history)
        semantic_q = generate.broaden_retrieval_query(retrieve_q)
        async with agent_connection(agent_id) as conn:
            retrieval = await retrieve_substance(
                conn,
                agent_id,
                body.message,
                expanded_query=retrieve_q,
                semantic_query=semantic_q,
            )
        chunks = retrieval.chunks
        evidence = build_evidence_package(
            body.message,
            chunks,
            retrieval.facts,
            query=retrieval.query,
        )
        result = await generate.answer_question(
            body.message,
            chunks,
            maker_name=agent["display_name"] or "the maker",
            history=history,
            facts=retrieval.facts,
            evidence=evidence,
        )
    except VoyageError as exc:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=502, detail=str(exc)[:800]) from exc
    except Exception as exc:
        logger.exception("Chat generation failed")
        raise HTTPException(status_code=502, detail=str(exc)[:800]) from exc

    # Soft uncertainty answers may have empty citations; only hard-fail invented cites
    if result.citations and not generate.check_citations_valid(result, chunks):
        result = generate.GroundedAnswer(
            answer=generate.DEFAULT_UNCERTAINTY,
            citations=[],
            confidence=0.0,
            refused=True,
            intent=result.intent,
            coverage=evidence.coverage,
            uncertainty=True,
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

    # Persist citations with provenance for later "where did this come from?"
    citation_payload = [c.to_dict() for c in result.citations]
    if result.provenance and not citation_payload:
        citation_payload = [p.to_dict() for p in result.provenance]

    async with agent_connection(agent_id) as conn:
        await queries.insert_message(
            conn,
            session_id=session_id,
            agent_id=agent_id,
            role="assistant",
            content=result.answer,
            citations=citation_payload,
            audio_uri=audio_uri,
        )

    return ChatResponse(
        session_id=session_id,
        answer=result.answer,
        citations=[
            CitationOut(
                chunk_id=c.chunk_id,
                quote=c.quote,
                memory_id=c.memory_id,
                support_level=c.support_level,
            )
            for c in result.citations
        ],
        confidence=result.confidence,
        refused=result.refused,
        intent=result.intent,
        audio_url=audio_url,
        coverage=result.coverage,
        uncertainty=result.uncertainty,
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
