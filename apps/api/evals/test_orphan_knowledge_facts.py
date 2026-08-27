"""_embed_and_store must not leave orphan knowledge_facts when a mid-write fails.

Distinct from chunk-only transaction coverage: substance ingest upserts searchable
facts (and may merge into other memories' established rows) under asyncpg
autocommit unless the whole DB phase is one transaction with Voyage outside it.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.db import queries
from app.services.memory_types import KnowledgeFactDraft, MemoryExtraction
from app.workers import tasks


def _extraction_with_facts() -> MemoryExtraction:
    return MemoryExtraction(
        cleaned_transcript="My sister Lena lives nearby.",
        chunk_metas=[],
        chunk_saliences=[0.7],
        fact_drafts=[
            KnowledgeFactDraft(
                statement="Has a sister named Lena.",
                category="relationship",
                people=["Lena"],
                confidence=0.9,
                salience=0.8,
                source_chunk_indices=[0],
                explicitly_stated=True,
            ),
            KnowledgeFactDraft(
                statement="Lena lives nearby.",
                category="attribute",
                people=["Lena"],
                confidence=0.85,
                salience=0.6,
                source_chunk_indices=[0],
                explicitly_stated=True,
            ),
        ],
    )


@pytest.mark.asyncio
async def test_embed_and_store_uses_transaction_for_facts_and_chunks():
    agent_id = uuid4()
    memory_id = uuid4()
    texts = ["My sister Lena lives nearby."]
    embeddings = [[0.1, 0.2]]

    conn = MagicMock()
    txn = MagicMock()
    txn.__aenter__ = AsyncMock(return_value=None)
    txn.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=txn)

    admin_cm = MagicMock()
    admin_cm.__aenter__ = AsyncMock(return_value=conn)
    admin_cm.__aexit__ = AsyncMock(return_value=False)

    chunk_id = uuid4()

    with (
        patch.object(tasks, "admin_connection", return_value=admin_cm),
        patch.object(
            tasks.voyage,
            "embed_documents",
            new=AsyncMock(side_effect=[embeddings, [[0.3]], [[0.4]]]),
        ),
        patch.object(
            tasks.extract,
            "extract_memory_structure",
            new=AsyncMock(return_value=_extraction_with_facts()),
        ),
        patch.object(queries, "delete_knowledge_facts_for_memory", new=AsyncMock()) as del_facts,
        patch.object(queries, "delete_chunks_for_memory", new=AsyncMock()) as del_chunks,
        patch.object(queries, "update_memory_transcripts", new=AsyncMock()),
        patch.object(queries, "insert_chunk", new=AsyncMock(return_value=chunk_id)) as insert,
        patch.object(queries, "upsert_knowledge_fact", new=AsyncMock()) as upsert,
        patch.object(queries, "update_memory_status", new=AsyncMock()) as status,
    ):
        await tasks._embed_and_store(
            agent_id=agent_id,
            memory_id=memory_id,
            texts=texts,
            document_context=texts[0],
            meta=[(None, None, None)],
        )

    conn.transaction.assert_called_once_with()
    txn.__aenter__.assert_awaited()
    del_facts.assert_awaited_once_with(conn, memory_id)
    del_chunks.assert_awaited_once_with(conn, memory_id)
    assert insert.await_count == 1
    assert upsert.await_count == 2
    status.assert_awaited_once_with(conn, memory_id, "indexed")


@pytest.mark.asyncio
async def test_fact_embeds_happen_before_db_transaction():
    """Voyage must not run inside the connection; otherwise autocommit orphans facts."""
    agent_id = uuid4()
    memory_id = uuid4()
    texts = ["My sister Lena lives nearby."]
    embeddings = [[0.1, 0.2]]

    order: list[str] = []

    async def track_embed(*_a, **_k):
        order.append("embed")
        if len([x for x in order if x == "embed"]) == 1:
            return embeddings
        return [[0.5]]

    conn = MagicMock()
    txn = MagicMock()

    async def txn_enter(*_a, **_k):
        order.append("txn")
        return None

    txn.__aenter__ = AsyncMock(side_effect=txn_enter)
    txn.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=txn)

    admin_cm = MagicMock()
    admin_cm.__aenter__ = AsyncMock(return_value=conn)
    admin_cm.__aexit__ = AsyncMock(return_value=False)

    with (
        patch.object(tasks, "admin_connection", return_value=admin_cm),
        patch.object(tasks.voyage, "embed_documents", new=AsyncMock(side_effect=track_embed)),
        patch.object(
            tasks.extract,
            "extract_memory_structure",
            new=AsyncMock(return_value=_extraction_with_facts()),
        ),
        patch.object(queries, "delete_knowledge_facts_for_memory", new=AsyncMock()),
        patch.object(queries, "delete_chunks_for_memory", new=AsyncMock()),
        patch.object(queries, "update_memory_transcripts", new=AsyncMock()),
        patch.object(queries, "insert_chunk", new=AsyncMock(return_value=uuid4())),
        patch.object(queries, "upsert_knowledge_fact", new=AsyncMock()),
        patch.object(queries, "update_memory_status", new=AsyncMock()),
    ):
        await tasks._embed_and_store(
            agent_id=agent_id,
            memory_id=memory_id,
            texts=texts,
            document_context=texts[0],
            meta=[(None, None, None)],
        )

    assert "txn" in order
    embed_indexes = [i for i, x in enumerate(order) if x == "embed"]
    txn_index = order.index("txn")
    assert embed_indexes
    assert max(embed_indexes) < txn_index


@pytest.mark.asyncio
async def test_mid_upsert_failure_rolls_back_via_transaction():
    agent_id = uuid4()
    memory_id = uuid4()
    texts = ["My sister Lena lives nearby."]
    embeddings = [[0.1, 0.2]]

    conn = MagicMock()
    txn = MagicMock()
    txn.__aenter__ = AsyncMock(return_value=None)
    txn.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=txn)

    admin_cm = MagicMock()
    admin_cm.__aenter__ = AsyncMock(return_value=conn)
    admin_cm.__aexit__ = AsyncMock(return_value=False)

    with (
        patch.object(tasks, "admin_connection", return_value=admin_cm),
        patch.object(
            tasks.voyage,
            "embed_documents",
            new=AsyncMock(side_effect=[embeddings, [[0.3]], [[0.4]]]),
        ),
        patch.object(
            tasks.extract,
            "extract_memory_structure",
            new=AsyncMock(return_value=_extraction_with_facts()),
        ),
        patch.object(queries, "delete_knowledge_facts_for_memory", new=AsyncMock()),
        patch.object(queries, "delete_chunks_for_memory", new=AsyncMock()),
        patch.object(queries, "update_memory_transcripts", new=AsyncMock()),
        patch.object(queries, "insert_chunk", new=AsyncMock(return_value=uuid4())),
        patch.object(
            queries,
            "upsert_knowledge_fact",
            new=AsyncMock(side_effect=[uuid4(), RuntimeError("db write failed")]),
        ),
        patch.object(queries, "update_memory_status", new=AsyncMock()) as status,
    ):
        with pytest.raises(RuntimeError, match="db write failed"):
            await tasks._embed_and_store(
                agent_id=agent_id,
                memory_id=memory_id,
                texts=texts,
                document_context=texts[0],
                meta=[(None, None, None)],
            )

    # Transaction context still closed (rollback path); status never committed as indexed.
    txn.__aexit__.assert_awaited()
    status.assert_not_awaited()
