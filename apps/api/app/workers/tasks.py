from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from uuid import UUID

from app.config import get_settings
from app.db import queries
from app.db.pool import admin_connection, init_pool
from app.services import assemblyai, elevenlabs, extract, voyage
from app.services.chunking import TextChunk, chunk_utterances
from app.services.extract import fact_status_for_draft
from app.services.memory_types import KnowledgeFactDraft

logger = logging.getLogger(__name__)


async def _embed_and_store(
    *,
    agent_id: UUID,
    memory_id: UUID,
    texts: list[str],
    document_context: str,
    meta: list[tuple[str | None, float | None, float | None]] | None = None,
) -> list[UUID]:
    embeddings = await voyage.embed_documents(texts, document_context=document_context or None)
    if len(embeddings) < len(texts):
        rest = await voyage.embed_documents(texts[len(embeddings) :])
        embeddings.extend(rest)
    embeddings = embeddings[: len(texts)]

    # Structured extraction + fact embeds happen BEFORE any DB writes so a
    # mid-loop Voyage failure cannot leave searchable knowledge_facts while
    # ingest later marks the memory status=error (asyncpg autocommits otherwise).
    extraction = await extract.extract_memory_structure(
        transcript=document_context or " ".join(texts),
        chunk_texts=texts,
    )

    prepared_facts: list[tuple[KnowledgeFactDraft, list[float]]] = []
    for draft in extraction.fact_drafts:
        if not draft.source_chunk_indices:
            continue
        if not any(0 <= i < len(texts) for i in draft.source_chunk_indices):
            continue
        try:
            fact_embs = await voyage.embed_documents([draft.statement])
        except Exception:  # noqa: BLE001
            logger.exception("Fact embed failed; skipping fact")
            continue
        if not fact_embs:
            continue
        prepared_facts.append((draft, fact_embs[0]))

    chunk_ids: list[UUID] = []
    async with admin_connection() as conn:
        async with conn.transaction():
            await queries.delete_knowledge_facts_for_memory(conn, memory_id)
            await queries.delete_chunks_for_memory(conn, memory_id)
            await queries.update_memory_transcripts(
                conn,
                memory_id,
                raw_transcript=document_context or None,
                cleaned_transcript=extraction.cleaned_transcript,
            )
            for i, (text, emb) in enumerate(zip(texts, embeddings)):
                speaker, ts_start, ts_end = (None, None, None)
                if meta and i < len(meta):
                    speaker, ts_start, ts_end = meta[i]
                chunk_meta = (
                    extraction.chunk_metas[i].to_dict()
                    if i < len(extraction.chunk_metas)
                    else {}
                )
                salience = (
                    extraction.chunk_saliences[i]
                    if i < len(extraction.chunk_saliences)
                    else 0.5
                )
                cid = await queries.insert_chunk(
                    conn,
                    agent_id=agent_id,
                    memory_id=memory_id,
                    text=text,
                    speaker=speaker,
                    ts_start=ts_start,
                    ts_end=ts_end,
                    embedding=emb,
                    meta=chunk_meta,
                    salience=salience,
                )
                chunk_ids.append(cid)

            for draft, fact_emb in prepared_facts:
                src_chunks = [
                    chunk_ids[i]
                    for i in draft.source_chunk_indices
                    if 0 <= i < len(chunk_ids)
                ]
                if not src_chunks:
                    continue
                status = fact_status_for_draft(draft, evidence_count=1)
                await queries.upsert_knowledge_fact(
                    conn,
                    agent_id=agent_id,
                    statement=draft.statement,
                    category=draft.category,
                    people=draft.people,
                    places=draft.places,
                    topics=draft.topics,
                    confidence=draft.confidence,
                    salience=draft.salience,
                    status=status,
                    supporting_chunk_ids=src_chunks,
                    supporting_memory_ids=[memory_id],
                    embedding=fact_emb,
                )

            await queries.update_memory_status(conn, memory_id, "indexed")
    return chunk_ids


async def ingest_memory(ctx: dict, memory_id: str) -> None:
    """Ingest text or voice memory → chunk → embed → extract → upsert chunks/facts."""
    await init_pool()
    mid = UUID(memory_id)
    settings = get_settings()

    async with admin_connection() as conn:
        memory = await queries.get_memory(conn, mid)
        if memory is None:
            logger.error("Memory %s not found", memory_id)
            return
        agent_id: UUID = memory["agent_id"]
        kind = memory["kind"] or "voice"

    try:
        if kind == "text":
            text = (memory["text_content"] or "").strip()
            if not text:
                raise RuntimeError("Empty text memory")
            async with admin_connection() as conn:
                await queries.update_memory_status(conn, mid, "embedding")
            await _embed_and_store(
                agent_id=agent_id,
                memory_id=mid,
                texts=[text],
                document_context=text,
                meta=[(None, None, None)],
            )
        else:
            audio_uri = memory["audio_uri"]
            if not audio_uri:
                raise RuntimeError("Voice memory missing audio")
            async with admin_connection() as conn:
                await queries.update_memory_status(conn, mid, "transcribing")

            audio_path = Path(audio_uri)
            if not audio_path.is_absolute():
                audio_path = settings.audio_dir / audio_path.name

            transcript = await asyncio.to_thread(assemblyai.transcribe_file, audio_path)
            async with admin_connection() as conn:
                await queries.update_memory_status(
                    conn, mid, "embedding", transcript_id=transcript.transcript_id
                )

            chunks = chunk_utterances(transcript.utterances)
            if not chunks and transcript.text:
                chunks = [
                    TextChunk(text=transcript.text, speaker=None, ts_start=None, ts_end=None)
                ]
            if not chunks:
                raise RuntimeError("No transcript text")

            await _embed_and_store(
                agent_id=agent_id,
                memory_id=mid,
                texts=[c.text for c in chunks],
                document_context=transcript.text or "",
                meta=[(c.speaker, c.ts_start, c.ts_end) for c in chunks],
            )

            # Auto-clone once we have enough voice samples
            async with admin_connection() as conn:
                voice_n = await queries.count_indexed_by_kind(conn, agent_id, "voice")
                agent = await queries.get_agent(conn, agent_id)
            if (
                voice_n >= queries.REQUIRED_VOICE_MEMORIES
                and agent
                and not agent["elevenlabs_voice_id"]
            ):
                await clone_agent_voice(ctx, str(agent_id))

        logger.info("Indexed memory %s", memory_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ingest failed for %s", memory_id)
        async with admin_connection() as conn:
            await queries.update_memory_status(
                conn, mid, "error", error_message=str(exc)[:1000]
            )


async def clone_agent_voice(ctx: dict, agent_id: str) -> None:
    await init_pool()
    aid = UUID(agent_id)
    settings = get_settings()

    async with admin_connection() as conn:
        agent = await queries.get_agent(conn, aid)
        if agent is None:
            return
        await queries.set_agent_status(conn, aid, "cloning")
        paths = await queries.list_memory_audio_paths(conn, aid)
        display_name = agent["display_name"]

    audio_paths: list[Path] = []
    for uri in paths:
        p = Path(uri)
        if not p.is_absolute():
            p = settings.audio_dir / p.name
        if p.exists():
            audio_paths.append(p)

    if not audio_paths:
        async with admin_connection() as conn:
            await queries.set_agent_status(conn, aid, "error")
        return

    try:
        voice_id = await elevenlabs.clone_voice(f"mira-{display_name}", audio_paths)
    except Exception:  # noqa: BLE001
        logger.exception("Voice clone failed for %s", agent_id)
        async with admin_connection() as conn:
            await queries.set_agent_status(conn, aid, "error")
        return

    async with admin_connection() as conn:
        await queries.set_agent_voice(conn, aid, voice_id, status="ready")
    logger.info("Cloned voice for agent %s -> %s", agent_id, voice_id)
