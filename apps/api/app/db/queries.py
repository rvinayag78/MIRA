from __future__ import annotations

import hashlib
import json
import secrets
from typing import Any
from uuid import UUID

import asyncpg


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_share_token() -> str:
    return secrets.token_urlsafe(24)


async def create_agent(conn: asyncpg.Connection, display_name: str) -> dict[str, Any]:
    token = new_share_token()
    token_hash = hash_token(token)
    row = await conn.fetchrow(
        """
        INSERT INTO agents (display_name, share_token_hash)
        VALUES ($1, $2)
        RETURNING id, display_name, elevenlabs_voice_id, status, created_at
        """,
        display_name,
        token_hash,
    )
    assert row is not None
    return {
        "id": row["id"],
        "display_name": row["display_name"],
        "elevenlabs_voice_id": row["elevenlabs_voice_id"],
        "status": row["status"],
        "created_at": row["created_at"],
        "share_token": token,
    }


async def get_agent_by_token(conn: asyncpg.Connection, agent_id: UUID, token: str) -> asyncpg.Record | None:
    return await conn.fetchrow(
        """
        SELECT id, display_name, elevenlabs_voice_id, status, created_at
        FROM agents
        WHERE id = $1 AND share_token_hash = $2
        """,
        agent_id,
        hash_token(token),
    )


async def get_agent(conn: asyncpg.Connection, agent_id: UUID) -> asyncpg.Record | None:
    return await conn.fetchrow(
        """
        SELECT id, display_name, elevenlabs_voice_id, status, created_at
        FROM agents WHERE id = $1
        """,
        agent_id,
    )


async def set_agent_voice(
    conn: asyncpg.Connection, agent_id: UUID, voice_id: str, status: str = "ready"
) -> None:
    await conn.execute(
        """
        UPDATE agents
        SET elevenlabs_voice_id = $2, status = $3
        WHERE id = $1
        """,
        agent_id,
        voice_id,
        status,
    )


async def set_agent_status(conn: asyncpg.Connection, agent_id: UUID, status: str) -> None:
    await conn.execute("UPDATE agents SET status = $2 WHERE id = $1", agent_id, status)


REQUIRED_TEXT_MEMORIES = 3
REQUIRED_VOICE_MEMORIES = 3


async def create_memory(
    conn: asyncpg.Connection,
    agent_id: UUID,
    *,
    kind: str,
    audio_uri: str | None = None,
    text_content: str | None = None,
    duration_ms: int | None = None,
) -> asyncpg.Record:
    row = await conn.fetchrow(
        """
        INSERT INTO memories (agent_id, kind, audio_uri, text_content, duration_ms, status)
        VALUES ($1, $2, $3, $4, $5, 'pending')
        RETURNING id, agent_id, kind, text_content, audio_uri, duration_ms, status,
                  error_message, created_at, assemblyai_transcript_id
        """,
        agent_id,
        kind,
        audio_uri,
        text_content,
        duration_ms,
    )
    assert row is not None
    return row


async def list_memories(conn: asyncpg.Connection, agent_id: UUID) -> list[asyncpg.Record]:
    return await conn.fetch(
        """
        SELECT id, agent_id, kind, text_content, audio_uri, duration_ms, status, error_message,
               assemblyai_transcript_id, created_at
        FROM memories
        WHERE agent_id = $1
        ORDER BY created_at DESC
        """,
        agent_id,
    )


async def get_memory(conn: asyncpg.Connection, memory_id: UUID) -> asyncpg.Record | None:
    return await conn.fetchrow(
        """
        SELECT id, agent_id, kind, text_content, audio_uri, duration_ms, status, error_message,
               assemblyai_transcript_id, created_at
        FROM memories WHERE id = $1
        """,
        memory_id,
    )


async def update_memory_status(
    conn: asyncpg.Connection,
    memory_id: UUID,
    status: str,
    *,
    transcript_id: str | None = None,
    error_message: str | None = None,
) -> None:
    await conn.execute(
        """
        UPDATE memories
        SET status = $2,
            assemblyai_transcript_id = COALESCE($3, assemblyai_transcript_id),
            error_message = $4
        WHERE id = $1
        """,
        memory_id,
        status,
        transcript_id,
        error_message,
    )


async def delete_memory(conn: asyncpg.Connection, memory_id: UUID) -> None:
    """Remove a memory row (and cascaded chunks). Used to roll back failed enqueue."""
    await conn.execute("DELETE FROM memories WHERE id = $1", memory_id)


async def delete_chunks_for_memory(conn: asyncpg.Connection, memory_id: UUID) -> None:
    await conn.execute("DELETE FROM chunks WHERE memory_id = $1", memory_id)


async def insert_chunk(
    conn: asyncpg.Connection,
    *,
    agent_id: UUID,
    memory_id: UUID,
    text: str,
    speaker: str | None,
    ts_start: float | None,
    ts_end: float | None,
    embedding: list[float],
) -> UUID:
    # asyncpg needs vector as string for pgvector
    emb_str = "[" + ",".join(str(float(x)) for x in embedding) + "]"
    row = await conn.fetchrow(
        """
        INSERT INTO chunks (agent_id, memory_id, text, speaker, ts_start, ts_end, embedding)
        VALUES ($1, $2, $3, $4, $5, $6, $7::vector)
        RETURNING id
        """,
        agent_id,
        memory_id,
        text,
        speaker,
        ts_start,
        ts_end,
        emb_str,
    )
    assert row is not None
    return row["id"]


async def count_indexed_memories(conn: asyncpg.Connection, agent_id: UUID) -> int:
    return int(
        await conn.fetchval(
            "SELECT COUNT(*) FROM memories WHERE agent_id = $1 AND status = 'indexed'",
            agent_id,
        )
    )


async def count_indexed_by_kind(conn: asyncpg.Connection, agent_id: UUID, kind: str) -> int:
    return int(
        await conn.fetchval(
            """
            SELECT COUNT(*) FROM memories
            WHERE agent_id = $1 AND kind = $2 AND status = 'indexed'
            """,
            agent_id,
            kind,
        )
    )


async def count_by_kind(conn: asyncpg.Connection, agent_id: UUID, kind: str) -> int:
    return int(
        await conn.fetchval(
            "SELECT COUNT(*) FROM memories WHERE agent_id = $1 AND kind = $2",
            agent_id,
            kind,
        )
    )


async def readiness(conn: asyncpg.Connection, agent_id: UUID) -> dict[str, Any]:
    text_n = await count_indexed_by_kind(conn, agent_id, "text")
    voice_n = await count_indexed_by_kind(conn, agent_id, "voice")
    return {
        "text_indexed": text_n,
        "voice_indexed": voice_n,
        "text_required": REQUIRED_TEXT_MEMORIES,
        "voice_required": REQUIRED_VOICE_MEMORIES,
        "ready_for_keeper": text_n >= REQUIRED_TEXT_MEMORIES
        and voice_n >= REQUIRED_VOICE_MEMORIES,
    }


async def list_memory_audio_paths(conn: asyncpg.Connection, agent_id: UUID) -> list[str]:
    rows = await conn.fetch(
        """
        SELECT audio_uri FROM memories
        WHERE agent_id = $1 AND kind = 'voice' AND status = 'indexed' AND audio_uri IS NOT NULL
        ORDER BY created_at ASC
        """,
        agent_id,
    )
    return [r["audio_uri"] for r in rows]


async def dense_search(
    conn: asyncpg.Connection, agent_id: UUID, embedding: list[float], limit: int
) -> list[asyncpg.Record]:
    emb_str = "[" + ",".join(str(float(x)) for x in embedding) + "]"
    return await conn.fetch(
        """
        SELECT id, memory_id, text, speaker, ts_start, ts_end,
               1 - (embedding <=> $2::vector) AS score
        FROM chunks
        WHERE agent_id = $1 AND embedding IS NOT NULL
        ORDER BY embedding <=> $2::vector
        LIMIT $3
        """,
        agent_id,
        emb_str,
        limit,
    )


async def sparse_search(
    conn: asyncpg.Connection, agent_id: UUID, query: str, limit: int
) -> list[asyncpg.Record]:
    return await conn.fetch(
        """
        SELECT id, memory_id, text, speaker, ts_start, ts_end,
               ts_rank_cd(tsv, plainto_tsquery('english', $2)) AS score
        FROM chunks
        WHERE agent_id = $1 AND tsv @@ plainto_tsquery('english', $2)
        ORDER BY score DESC
        LIMIT $3
        """,
        agent_id,
        query,
        limit,
    )


async def get_chunks_by_ids(
    conn: asyncpg.Connection, agent_id: UUID, chunk_ids: list[UUID]
) -> list[asyncpg.Record]:
    if not chunk_ids:
        return []
    return await conn.fetch(
        """
        SELECT id, memory_id, text, speaker, ts_start, ts_end
        FROM chunks
        WHERE agent_id = $1 AND id = ANY($2::uuid[])
        """,
        agent_id,
        chunk_ids,
    )


async def create_session(conn: asyncpg.Connection, agent_id: UUID) -> UUID:
    row = await conn.fetchrow(
        "INSERT INTO chat_sessions (agent_id) VALUES ($1) RETURNING id",
        agent_id,
    )
    assert row is not None
    return row["id"]


async def get_owned_session(
    conn: asyncpg.Connection, session_id: UUID, agent_id: UUID
) -> UUID | None:
    """Return session_id only if it belongs to agent_id; else None."""
    row = await conn.fetchrow(
        """
        SELECT id FROM chat_sessions
        WHERE id = $1 AND agent_id = $2
        """,
        session_id,
        agent_id,
    )
    return row["id"] if row is not None else None


async def resolve_session(
    conn: asyncpg.Connection, agent_id: UUID, session_id: UUID | None
) -> UUID:
    """Use an owned session_id or create a new one. Never attach to another agent."""
    if session_id is not None:
        owned = await get_owned_session(conn, session_id, agent_id)
        if owned is not None:
            return owned
    return await create_session(conn, agent_id)


async def insert_message(
    conn: asyncpg.Connection,
    *,
    session_id: UUID,
    agent_id: UUID,
    role: str,
    content: str,
    citations: list[dict[str, Any]] | None = None,
    audio_uri: str | None = None,
) -> UUID:
    row = await conn.fetchrow(
        """
        INSERT INTO messages (session_id, agent_id, role, content, citations, audio_uri)
        VALUES ($1, $2, $3, $4, $5::jsonb, $6)
        RETURNING id
        """,
        session_id,
        agent_id,
        role,
        content,
        json.dumps(citations) if citations is not None else None,
        audio_uri,
    )
    assert row is not None
    return row["id"]


async def load_all_chunks_for_agent(conn: asyncpg.Connection, agent_id: UUID) -> list[asyncpg.Record]:
    return await conn.fetch(
        """
        SELECT id, memory_id, text, speaker, ts_start, ts_end
        FROM chunks WHERE agent_id = $1
        ORDER BY created_at ASC
        """,
        agent_id,
    )
