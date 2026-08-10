from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from uuid import UUID

from app.config import get_settings
from app.db import queries
from app.db.pool import admin_connection, init_pool
from app.services import assemblyai, elevenlabs, voyage
from app.services.chunking import chunk_utterances

logger = logging.getLogger(__name__)


async def ingest_memory(ctx: dict, memory_id: str) -> None:
    """Transcribe → chunk → embed → upsert chunks."""
    await init_pool()
    mid = UUID(memory_id)
    settings = get_settings()

    async with admin_connection() as conn:
        memory = await queries.get_memory(conn, mid)
        if memory is None:
            logger.error("Memory %s not found", memory_id)
            return
        agent_id: UUID = memory["agent_id"]
        audio_uri: str = memory["audio_uri"]
        await queries.update_memory_status(conn, mid, "transcribing")

    audio_path = Path(audio_uri)
    if not audio_path.is_absolute():
        audio_path = settings.audio_dir / audio_path.name

    try:
        transcript = await asyncio.to_thread(assemblyai.transcribe_file, audio_path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Transcription failed for %s", memory_id)
        async with admin_connection() as conn:
            await queries.update_memory_status(
                conn, mid, "error", error_message=str(exc)[:1000]
            )
        return

    async with admin_connection() as conn:
        await queries.update_memory_status(
            conn, mid, "embedding", transcript_id=transcript.transcript_id
        )

    chunks = chunk_utterances(transcript.utterances)
    if not chunks and transcript.text:
        from app.services.chunking import TextChunk

        chunks = [TextChunk(text=transcript.text, speaker=None, ts_start=None, ts_end=None)]

    try:
        texts = [c.text for c in chunks]
        embeddings = await voyage.embed_documents(
            texts, document_context=transcript.text or None
        )
        if len(embeddings) != len(chunks):
            # pad/truncate safely
            if len(embeddings) < len(chunks):
                # embed remainder without context
                rest = await voyage.embed_documents(texts[len(embeddings) :])
                embeddings.extend(rest)
            embeddings = embeddings[: len(chunks)]
    except Exception as exc:  # noqa: BLE001
        logger.exception("Embedding failed for %s", memory_id)
        async with admin_connection() as conn:
            await queries.update_memory_status(
                conn, mid, "error", error_message=str(exc)[:1000]
            )
        return

    async with admin_connection() as conn:
        await queries.delete_chunks_for_memory(conn, mid)
        for chunk, emb in zip(chunks, embeddings, strict=False):
            await queries.insert_chunk(
                conn,
                agent_id=agent_id,
                memory_id=mid,
                text=chunk.text,
                speaker=chunk.speaker,
                ts_start=chunk.ts_start,
                ts_end=chunk.ts_end,
                embedding=emb,
            )
        await queries.update_memory_status(conn, mid, "indexed")

    logger.info("Indexed memory %s (%d chunks)", memory_id, len(chunks))


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
        voice_id = await elevenlabs.clone_voice(f"ovyu-{display_name}", audio_paths)
    except Exception:  # noqa: BLE001
        logger.exception("Voice clone failed for %s", agent_id)
        async with admin_connection() as conn:
            await queries.set_agent_status(conn, aid, "error")
        return

    async with admin_connection() as conn:
        await queries.set_agent_voice(conn, aid, voice_id, status="ready")
    logger.info("Cloned voice for agent %s -> %s", agent_id, voice_id)
