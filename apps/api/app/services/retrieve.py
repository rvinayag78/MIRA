from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import asyncpg

from app.config import get_settings
from app.db import queries
from app.services import voyage


@dataclass
class RetrievedChunk:
    id: UUID
    memory_id: UUID
    text: str
    speaker: str | None
    ts_start: float | None
    ts_end: float | None
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "memory_id": str(self.memory_id),
            "text": self.text,
            "speaker": self.speaker,
            "ts_start": self.ts_start,
            "ts_end": self.ts_end,
            "score": self.score,
        }


def reciprocal_rank_fusion(
    rankings: list[list[UUID]],
    *,
    k: int = 60,
) -> list[tuple[UUID, float]]:
    scores: dict[UUID, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


_SMALL_CORPUS = 40


def _row_to_chunk(row: asyncpg.Record, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        id=row["id"],
        memory_id=row["memory_id"],
        text=row["text"],
        speaker=row["speaker"],
        ts_start=row["ts_start"],
        ts_end=row["ts_end"],
        score=score,
    )


async def hybrid_retrieve(
    conn: asyncpg.Connection,
    agent_id: UUID,
    query: str,
    *,
    semantic_query: str | None = None,
    rerank_enabled: bool | None = None,
) -> list[RetrievedChunk]:
    settings = get_settings()
    use_rerank = settings.rerank_enabled if rerank_enabled is None else rerank_enabled

    query_emb = await voyage.embed_query(semantic_query or query)
    dense_rows = await queries.dense_search(conn, agent_id, query_emb, settings.dense_top_k)
    sparse_rows = await queries.sparse_search(conn, agent_id, query, settings.sparse_top_k)

    dense_ids = [r["id"] for r in dense_rows]
    sparse_ids = [r["id"] for r in sparse_rows]
    fused = reciprocal_rank_fusion([dense_ids, sparse_ids], k=settings.rrf_k)

    by_id: dict[UUID, asyncpg.Record] = {r["id"]: r for r in dense_rows}
    for r in sparse_rows:
        by_id.setdefault(r["id"], r)

    candidates: list[RetrievedChunk] = []
    for doc_id, score in fused[: max(settings.final_top_k * 3, settings.final_top_k)]:
        row = by_id.get(doc_id)
        if row is None:
            continue
        candidates.append(_row_to_chunk(row, float(score)))

    if use_rerank and candidates:
        docs = [c.text for c in candidates]
        try:
            ranked = await voyage.rerank(query, docs, top_k=settings.final_top_k)
            reranked: list[RetrievedChunk] = []
            for item in ranked:
                idx = item["index"]
                if idx is None or idx >= len(candidates):
                    continue
                c = candidates[idx]
                c.score = float(item["relevance_score"])
                reranked.append(c)
            if reranked:
                return reranked[: settings.final_top_k]
        except voyage.VoyageError:
            # Fail open to RRF ordering
            pass

    return candidates[: settings.final_top_k]


async def retrieve_for_chat(
    conn: asyncpg.Connection,
    agent_id: UUID,
    query: str,
    *,
    semantic_query: str | None = None,
) -> list[RetrievedChunk]:
    """Rank by hybrid search, then include the rest of a small memory set.

    Open questions like "what was dad like?" can match a related story even when
    the query does not name that place.
    """
    ranked = await hybrid_retrieve(conn, agent_id, query, semantic_query=semantic_query)
    n = await conn.fetchval("SELECT COUNT(*) FROM chunks WHERE agent_id = $1", agent_id)
    if n is None or int(n) > _SMALL_CORPUS:
        return ranked
    seen = {c.id for c in ranked}
    extras: list[RetrievedChunk] = []
    for row in await queries.load_all_chunks_for_agent(conn, agent_id):
        if row["id"] in seen:
            continue
        extras.append(_row_to_chunk(row, 0.0))
    return ranked + extras
