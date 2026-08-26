from __future__ import annotations

import hashlib
import json
import re
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
        SELECT id, agent_id, kind, text_content, raw_transcript, cleaned_transcript,
               audio_uri, duration_ms, status, error_message,
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


async def update_memory_transcripts(
    conn: asyncpg.Connection,
    memory_id: UUID,
    *,
    raw_transcript: str | None,
    cleaned_transcript: str | None,
) -> None:
    await conn.execute(
        """
        UPDATE memories
        SET raw_transcript = COALESCE($2, raw_transcript),
            cleaned_transcript = COALESCE($3, cleaned_transcript)
        WHERE id = $1
        """,
        memory_id,
        raw_transcript,
        cleaned_transcript,
    )


async def delete_chunks_for_memory(conn: asyncpg.Connection, memory_id: UUID) -> None:
    await conn.execute("DELETE FROM chunks WHERE memory_id = $1", memory_id)


async def delete_knowledge_facts_for_memory(
    conn: asyncpg.Connection, memory_id: UUID
) -> None:
    """Remove facts whose only supporting memory is this one; else drop the link."""
    rows = await conn.fetch(
        """
        SELECT id, supporting_memory_ids, supporting_chunk_ids, evidence_count
        FROM knowledge_facts
        WHERE $1 = ANY(supporting_memory_ids)
        """,
        memory_id,
    )
    for row in rows:
        mem_ids = [m for m in (row["supporting_memory_ids"] or []) if m != memory_id]
        if not mem_ids:
            await conn.execute("DELETE FROM knowledge_facts WHERE id = $1", row["id"])
            continue
        chunk_ids = list(row["supporting_chunk_ids"] or [])
        await conn.execute(
            """
            UPDATE knowledge_facts
            SET supporting_memory_ids = $2,
                supporting_chunk_ids = $3,
                evidence_count = GREATEST(1, $4),
                updated_at = now()
            WHERE id = $1
            """,
            row["id"],
            mem_ids,
            chunk_ids,
            len(mem_ids),
        )


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
    meta: dict[str, Any] | None = None,
    salience: float = 0.5,
) -> UUID:
    # asyncpg needs vector as string for pgvector
    emb_str = "[" + ",".join(str(float(x)) for x in embedding) + "]"
    row = await conn.fetchrow(
        """
        INSERT INTO chunks (
            agent_id, memory_id, text, speaker, ts_start, ts_end,
            embedding, meta, salience
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7::vector, $8::jsonb, $9)
        RETURNING id
        """,
        agent_id,
        memory_id,
        text,
        speaker,
        ts_start,
        ts_end,
        emb_str,
        json.dumps(meta or {}),
        max(0.0, min(1.0, float(salience))),
    )
    assert row is not None
    return row["id"]


def _emb_str(embedding: list[float]) -> str:
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"


async def upsert_knowledge_fact(
    conn: asyncpg.Connection,
    *,
    agent_id: UUID,
    statement: str,
    category: str,
    people: list[str],
    places: list[str],
    topics: list[str],
    confidence: float,
    salience: float,
    status: str,
    supporting_chunk_ids: list[UUID],
    supporting_memory_ids: list[UUID],
    embedding: list[float],
    conflict_note: str | None = None,
) -> UUID:
    """Merge into a similar existing fact when cosine similarity is high.

    Distinct entities (different people/places) or conflicting statements must
    NOT merge: templated relationship facts are near neighbors in embed space
    and would otherwise collapse into one established claim that drops names.
    """
    emb = _emb_str(embedding)
    similar = await conn.fetchrow(
        """
        SELECT id, statement, people, places, supporting_chunk_ids, supporting_memory_ids,
               evidence_count, confidence, status, conflict_note
        FROM knowledge_facts
        WHERE agent_id = $1 AND embedding IS NOT NULL
        ORDER BY embedding <=> $2::vector
        LIMIT 1
        """,
        agent_id,
        emb,
    )
    merge = False
    if similar is not None:
        dist = await conn.fetchval(
            """
            SELECT embedding <=> $2::vector
            FROM knowledge_facts WHERE id = $1
            """,
            similar["id"],
            emb,
        )
        merge = should_merge_knowledge_facts(
            distance=float(dist) if dist is not None else 1.0,
            existing_statement=similar["statement"] or "",
            new_statement=statement,
            existing_people=list(similar["people"] or []),
            new_people=people,
            existing_places=list(similar["places"] or []),
            new_places=places,
        )

    if merge and similar is not None:
        chunk_ids = list(dict.fromkeys([*(similar["supporting_chunk_ids"] or []), *supporting_chunk_ids]))
        mem_ids = list(dict.fromkeys([*(similar["supporting_memory_ids"] or []), *supporting_memory_ids]))
        evidence_count = len(mem_ids)
        new_conf = max(float(similar["confidence"] or 0), confidence)
        new_status = status
        note = conflict_note or similar["conflict_note"]
        if evidence_count >= 2 and new_conf >= 0.55:
            new_status = "established" if similar["status"] != "disputed" else "disputed"

        await conn.execute(
            """
            UPDATE knowledge_facts
            SET supporting_chunk_ids = $2,
                supporting_memory_ids = $3,
                evidence_count = $4,
                confidence = $5,
                status = $6,
                conflict_note = $7,
                salience = GREATEST(salience, $8),
                updated_at = now()
            WHERE id = $1
            """,
            similar["id"],
            chunk_ids,
            mem_ids,
            evidence_count,
            new_conf,
            new_status,
            note,
            salience,
        )
        return similar["id"]

    row = await conn.fetchrow(
        """
        INSERT INTO knowledge_facts (
            agent_id, statement, category, people, places, topics,
            confidence, evidence_count, supporting_chunk_ids, supporting_memory_ids,
            status, salience, conflict_note, embedding
        )
        VALUES (
            $1, $2, $3, $4, $5, $6,
            $7, $8, $9, $10,
            $11, $12, $13, $14::vector
        )
        RETURNING id
        """,
        agent_id,
        statement,
        category,
        people,
        places,
        topics,
        confidence,
        max(1, len(set(supporting_memory_ids))),
        supporting_chunk_ids,
        supporting_memory_ids,
        status,
        salience,
        conflict_note,
        emb,
    )
    assert row is not None
    return row["id"]


_NAME_STOPWORDS = {
    "has",
    "have",
    "had",
    "the",
    "a",
    "an",
    "my",
    "our",
    "his",
    "her",
    "their",
    "named",
    "called",
    "sister",
    "brother",
    "mother",
    "father",
    "mom",
    "dad",
    "friend",
    "wife",
    "husband",
    "partner",
    "connected",
    "place",
    "lived",
    "from",
    "in",
    "on",
    "at",
    "to",
    "and",
    "or",
    "of",
    "is",
    "was",
    "were",
    "with",
}


def _norm_entity_set(values: list[str]) -> set[str]:
    return {v.strip().lower() for v in values if v and str(v).strip()}


def _entity_sets_conflict(existing: list[str], new: list[str]) -> bool:
    """True when both sides name entities and the sets disagree."""
    a = _norm_entity_set(existing)
    b = _norm_entity_set(new)
    if not a or not b:
        return False
    return a != b


def _proper_names(text: str) -> set[str]:
    names = {m.group(0).lower() for m in re.finditer(r"\b[A-Z][a-z]{2,}\b", text or "")}
    return {n for n in names if n not in _NAME_STOPWORDS}


def _statements_conflict(a: str, b: str) -> bool:
    """Detect same-template facts that disagree on year or named entity."""
    years_a = set(re.findall(r"\b(?:19|20)\d{2}\b", a))
    years_b = set(re.findall(r"\b(?:19|20)\d{2}\b", b))
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    overlap = len(ta & tb)
    if years_a and years_b and years_a.isdisjoint(years_b) and overlap >= 3:
        return True
    names_a = _proper_names(a)
    names_b = _proper_names(b)
    if names_a and names_b and names_a != names_b and overlap >= 3:
        return True
    return False


def should_merge_knowledge_facts(
    *,
    distance: float,
    existing_statement: str,
    new_statement: str,
    existing_people: list[str],
    new_people: list[str],
    existing_places: list[str],
    new_places: list[str],
    max_distance: float = 0.25,
) -> bool:
    """Return True only when the new fact is the same claim as the nearest neighbor.

    Near-neighbor relationship templates (sister Lena vs sister Laura) often sit
    well under the cosine distance threshold; merging them would keep one name,
    attach the other memory as evidence, and promote a false established fact.
    """
    if distance > max_distance:
        return False
    if _entity_sets_conflict(existing_people, new_people):
        return False
    if _entity_sets_conflict(existing_places, new_places):
        return False
    if _statements_conflict(existing_statement, new_statement):
        return False
    return True


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
            """
            SELECT COUNT(*) FROM memories
            WHERE agent_id = $1 AND kind = $2 AND status <> 'error'
            """,
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
        SELECT id, memory_id, text, speaker, ts_start, ts_end, meta, salience,
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
        SELECT id, memory_id, text, speaker, ts_start, ts_end, meta, salience,
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


async def dense_search_facts(
    conn: asyncpg.Connection, agent_id: UUID, embedding: list[float], limit: int
) -> list[asyncpg.Record]:
    emb_str = "[" + ",".join(str(float(x)) for x in embedding) + "]"
    return await conn.fetch(
        """
        SELECT id, statement, category, confidence, status, people, places, topics,
               supporting_chunk_ids, supporting_memory_ids, conflict_note, salience,
               1 - (embedding <=> $2::vector) AS score
        FROM knowledge_facts
        WHERE agent_id = $1 AND embedding IS NOT NULL
          AND status IN ('established', 'candidate', 'disputed')
        ORDER BY embedding <=> $2::vector
        LIMIT $3
        """,
        agent_id,
        emb_str,
        limit,
    )


async def sparse_search_facts(
    conn: asyncpg.Connection, agent_id: UUID, query: str, limit: int
) -> list[asyncpg.Record]:
    return await conn.fetch(
        """
        SELECT id, statement, category, confidence, status, people, places, topics,
               supporting_chunk_ids, supporting_memory_ids, conflict_note, salience,
               ts_rank_cd(tsv, plainto_tsquery('english', $2)) AS score
        FROM knowledge_facts
        WHERE agent_id = $1 AND tsv @@ plainto_tsquery('english', $2)
          AND status IN ('established', 'candidate', 'disputed')
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
        SELECT id, memory_id, text, speaker, ts_start, ts_end, meta, salience
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


async def list_session_messages(
    conn: asyncpg.Connection, session_id: UUID, *, limit: int = 8
) -> list[asyncpg.Record]:
    rows = await conn.fetch(
        """
        SELECT role, content
        FROM messages
        WHERE session_id = $1
        ORDER BY created_at DESC
        LIMIT $2
        """,
        session_id,
        limit,
    )
    return list(reversed(rows))


async def load_all_chunks_for_agent(conn: asyncpg.Connection, agent_id: UUID) -> list[asyncpg.Record]:
    return await conn.fetch(
        """
        SELECT id, memory_id, text, speaker, ts_start, ts_end, meta, salience
        FROM chunks WHERE agent_id = $1
        ORDER BY created_at ASC
        """,
        agent_id,
    )


async def list_chunks_for_memory(
    conn: asyncpg.Connection, memory_id: UUID
) -> list[asyncpg.Record]:
    return await conn.fetch(
        """
        SELECT id, memory_id, text, speaker, ts_start, ts_end, meta, salience
        FROM chunks WHERE memory_id = $1
        ORDER BY created_at ASC
        """,
        memory_id,
    )
