"""Hybrid retrieval with entity boosting, diversity, and semantic facts.

Pipeline:
  query understanding → dense+sparse RRF → optional Voyage rerank →
  entity/salience boost → per-memory diversity → knowledge facts → evidence package
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import asyncpg

from app.config import get_settings
from app.db import queries
from app.services import voyage
from app.services.memory_types import (
    ChunkMeta,
    QueryUnderstanding,
    RetrievedFact,
)


@dataclass
class RetrievedChunk:
    id: UUID
    memory_id: UUID
    text: str
    speaker: str | None
    ts_start: float | None
    ts_end: float | None
    score: float
    meta: dict[str, Any] = field(default_factory=dict)
    salience: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "memory_id": str(self.memory_id),
            "text": self.text,
            "speaker": self.speaker,
            "ts_start": self.ts_start,
            "ts_end": self.ts_end,
            "score": self.score,
            "meta": self.meta,
            "salience": self.salience,
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

_STORY_CUES = re.compile(
    r"\b(story|stories|memory|memories|remember|tell me about|what was .+ like)\b",
    re.I,
)
_PERSON_CUES = [
    (re.compile(r"\b(dad|daddy|father|papa)\b", re.I), "dad"),
    (re.compile(r"\b(mom|mum|mama|mother)\b", re.I), "mom"),
    (re.compile(r"\b(sister)\b", re.I), "sister"),
    (re.compile(r"\b(brother)\b", re.I), "brother"),
]
_PLACE_CUES = re.compile(
    r"\b(?:in|at|from|to|near)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
)
_YEAR_CUES = re.compile(r"\b((?:19|20)\d{2})s?\b")


def understand_query(
    question: str,
    *,
    expanded: str | None = None,
    semantic: str | None = None,
) -> QueryUnderstanding:
    """Lightweight query understanding (no LLM). Enough for entity boosting."""
    q = question.strip()
    people: list[str] = []
    for pat, label in _PERSON_CUES:
        if pat.search(q):
            people.append(label)
    # Named people: capitalized tokens after family words
    for m in re.finditer(
        r"\b(?:sister|brother|friend|wife|husband|partner)\s+([A-Z][a-z]+)\b", q
    ):
        people.append(m.group(1))
    for m in re.finditer(r"\b([A-Z][a-z]{2,})\b", q):
        name = m.group(1)
        if name.lower() not in {
            "what",
            "where",
            "when",
            "who",
            "how",
            "why",
            "tell",
            "about",
            "your",
            "did",
            "was",
            "were",
            "the",
        }:
            # Keep only if question also has lowercase context (avoid over-matching)
            if name.lower() in q.lower():
                people.append(name)

    places = [m.group(1) for m in _PLACE_CUES.finditer(q)]
    time_hints = _YEAR_CUES.findall(q)
    topics: list[str] = []
    for cue, topic in (
        ("home", "home"),
        ("childhood", "childhood"),
        ("work", "work"),
        ("job", "work"),
        ("school", "school"),
        ("family", "family"),
    ):
        if cue in q.lower():
            topics.append(topic)

    # Dedupe preserving order
    def uniq(xs: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in xs:
            k = x.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(x)
        return out

    return QueryUnderstanding(
        raw=q,
        expanded=(expanded or q).strip(),
        semantic=(semantic or expanded or q).strip(),
        people=uniq(people),
        places=uniq(places),
        topics=uniq(topics),
        time_hints=uniq(time_hints),
        wants_story=bool(_STORY_CUES.search(q)),
    )


def _row_to_chunk(row: asyncpg.Record, score: float) -> RetrievedChunk:
    meta = row["meta"] if "meta" in row.keys() else {}
    if isinstance(meta, str):
        import json

        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            meta = {}
    if not isinstance(meta, dict):
        meta = {}
    salience = float(row["salience"]) if "salience" in row.keys() and row["salience"] is not None else 0.5
    return RetrievedChunk(
        id=row["id"],
        memory_id=row["memory_id"],
        text=row["text"],
        speaker=row["speaker"],
        ts_start=row["ts_start"],
        ts_end=row["ts_end"],
        score=score,
        meta=meta,
        salience=salience,
    )


def _row_to_fact(row: asyncpg.Record, score: float) -> RetrievedFact:
    return RetrievedFact(
        id=row["id"],
        statement=row["statement"],
        category=row["category"],
        confidence=float(row["confidence"] or 0.5),
        status=row["status"],
        people=list(row["people"] or []),
        places=list(row["places"] or []),
        topics=list(row["topics"] or []),
        supporting_chunk_ids=list(row["supporting_chunk_ids"] or []),
        supporting_memory_ids=list(row["supporting_memory_ids"] or []),
        conflict_note=row["conflict_note"],
        score=score,
    )


def boost_chunk_score(chunk: RetrievedChunk, query: QueryUnderstanding) -> float:
    """Re-score with entity overlap, salience, and temporal hints."""
    score = float(chunk.score)
    meta = ChunkMeta.from_dict(chunk.meta)
    entities = query.entity_terms()
    if entities:
        overlap = entities & meta.entity_names()
        # Also match against raw text
        text_l = chunk.text.lower()
        for term in entities:
            if term in text_l:
                overlap.add(term)
        score += 0.08 * len(overlap)
    if query.time_hints:
        for hint in query.time_hints:
            if hint in chunk.text:
                score += 0.06
            for tp in meta.time_periods:
                if hint in (tp.text or "") or (
                    tp.approx_year and hint.startswith(str(tp.approx_year)[:3])
                ):
                    score += 0.05
    score += 0.05 * float(chunk.salience or 0.5)
    if query.wants_story and meta.events:
        score += 0.04
    return score


def diversify_chunks(
    chunks: list[RetrievedChunk],
    *,
    final_k: int,
    max_per_memory: int = 2,
) -> list[RetrievedChunk]:
    """Keep complementary evidence across memories (avoid one-memory domination)."""
    ordered = sorted(chunks, key=lambda c: c.score, reverse=True)
    picked: list[RetrievedChunk] = []
    per_memory: dict[UUID, int] = {}
    for c in ordered:
        n = per_memory.get(c.memory_id, 0)
        if n >= max_per_memory:
            continue
        picked.append(c)
        per_memory[c.memory_id] = n + 1
        if len(picked) >= final_k:
            break
    # If we undershot, fill with remaining best regardless of memory cap
    if len(picked) < final_k:
        seen = {c.id for c in picked}
        for c in ordered:
            if c.id in seen:
                continue
            picked.append(c)
            if len(picked) >= final_k:
                break
    return picked


async def hybrid_retrieve(
    conn: asyncpg.Connection,
    agent_id: UUID,
    query: str,
    *,
    semantic_query: str | None = None,
    rerank_enabled: bool | None = None,
    query_understanding: QueryUnderstanding | None = None,
) -> list[RetrievedChunk]:
    settings = get_settings()
    use_rerank = settings.rerank_enabled if rerank_enabled is None else rerank_enabled
    qu = query_understanding or understand_query(
        query, expanded=query, semantic=semantic_query
    )

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
    pool_n = max(settings.final_top_k * 4, settings.final_top_k)
    for doc_id, score in fused[:pool_n]:
        row = by_id.get(doc_id)
        if row is None:
            continue
        candidates.append(_row_to_chunk(row, float(score)))

    if use_rerank and candidates:
        docs = [c.text for c in candidates]
        try:
            ranked = await voyage.rerank(query, docs, top_k=min(len(candidates), pool_n))
            reranked: list[RetrievedChunk] = []
            for item in ranked:
                idx = item["index"]
                if idx is None or idx >= len(candidates):
                    continue
                c = candidates[idx]
                c.score = float(item["relevance_score"])
                reranked.append(c)
            if reranked:
                candidates = reranked
        except voyage.VoyageError:
            pass

    for c in candidates:
        c.score = boost_chunk_score(c, qu)

    return diversify_chunks(
        candidates,
        final_k=settings.final_top_k,
        max_per_memory=settings.retrieve_max_per_memory,
    )


async def retrieve_facts(
    conn: asyncpg.Connection,
    agent_id: UUID,
    query: str,
    *,
    semantic_query: str | None = None,
    limit: int = 6,
) -> list[RetrievedFact]:
    query_emb = await voyage.embed_query(semantic_query or query)
    dense = await queries.dense_search_facts(conn, agent_id, query_emb, limit)
    sparse = await queries.sparse_search_facts(conn, agent_id, query, limit)
    by_id: dict[UUID, RetrievedFact] = {}
    for r in dense:
        by_id[r["id"]] = _row_to_fact(r, float(r["score"] or 0))
    for r in sparse:
        existing = by_id.get(r["id"])
        score = float(r["score"] or 0)
        if existing is None or score > existing.score:
            by_id[r["id"]] = _row_to_fact(r, score)
    facts = list(by_id.values())
    # Boost established / high confidence
    for f in facts:
        if f.status == "established":
            f.score += 0.1
        elif f.status == "disputed":
            f.score += 0.05  # keep for uncertainty framing
        f.score += 0.05 * f.confidence
    facts.sort(key=lambda f: f.score, reverse=True)
    return facts[:limit]


async def retrieve_for_chat(
    conn: asyncpg.Connection,
    agent_id: UUID,
    query: str,
    *,
    semantic_query: str | None = None,
    query_understanding: QueryUnderstanding | None = None,
) -> list[RetrievedChunk]:
    """Rank by hybrid search, then include the rest of a small memory set.

    Open questions like "what was dad like?" can match a related story even when
    the query does not name that place.
    """
    ranked = await hybrid_retrieve(
        conn,
        agent_id,
        query,
        semantic_query=semantic_query,
        query_understanding=query_understanding,
    )
    n = await conn.fetchval("SELECT COUNT(*) FROM chunks WHERE agent_id = $1", agent_id)
    if n is None or int(n) > _SMALL_CORPUS:
        return ranked
    seen = {c.id for c in ranked}
    extras: list[RetrievedChunk] = []
    for row in await queries.load_all_chunks_for_agent(conn, agent_id):
        if row["id"] in seen:
            continue
        extras.append(_row_to_chunk(row, 0.0))
    # Still diversify the pad so one memory does not flood context
    combined = ranked + extras
    settings = get_settings()
    # For small corpora keep broad coverage but cap total context size
    return diversify_chunks(
        combined,
        final_k=min(len(combined), max(settings.final_top_k, 12)),
        max_per_memory=settings.retrieve_max_per_memory,
    )


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    facts: list[RetrievedFact]
    query: QueryUnderstanding


async def retrieve_substance(
    conn: asyncpg.Connection,
    agent_id: UUID,
    question: str,
    *,
    expanded_query: str,
    semantic_query: str,
) -> RetrievalResult:
    qu = understand_query(
        question, expanded=expanded_query, semantic=semantic_query
    )
    chunks = await retrieve_for_chat(
        conn,
        agent_id,
        expanded_query,
        semantic_query=semantic_query,
        query_understanding=qu,
    )
    facts = await retrieve_facts(
        conn, agent_id, expanded_query, semantic_query=semantic_query
    )
    return RetrievalResult(chunks=chunks, facts=facts, query=qu)
