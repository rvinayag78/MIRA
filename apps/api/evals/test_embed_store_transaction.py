"""_embed_and_store must not leave orphan chunks when a mid-write fails."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call, patch
from uuid import uuid4

import pytest

from app.db import queries
from app.workers import tasks


@pytest.mark.asyncio
async def test_embed_and_store_uses_a_transaction():
    agent_id = uuid4()
    memory_id = uuid4()
    texts = ["chunk a", "chunk b"]
    embeddings = [[0.1, 0.2], [0.3, 0.4]]

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
            new=AsyncMock(return_value=embeddings),
        ),
        patch.object(queries, "delete_chunks_for_memory", new=AsyncMock()) as delete,
        patch.object(queries, "insert_chunk", new=AsyncMock()) as insert,
        patch.object(queries, "update_memory_status", new=AsyncMock()) as status,
    ):
        await tasks._embed_and_store(
            agent_id=agent_id,
            memory_id=memory_id,
            texts=texts,
            document_context="full doc",
            meta=[(None, None, None), (None, None, None)],
        )

    conn.transaction.assert_called_once_with()
    txn.__aenter__.assert_awaited()
    delete.assert_awaited_once_with(conn, memory_id)
    assert insert.await_count == 2
    status.assert_awaited_once_with(conn, memory_id, "indexed")
    # delete → inserts → status all happen under the same connection/transaction
    assert delete.await_args == call(conn, memory_id)


@pytest.mark.asyncio
async def test_embed_and_store_aborts_before_indexed_when_insert_fails():
    agent_id = uuid4()
    memory_id = uuid4()
    texts = ["chunk a", "chunk b"]
    embeddings = [[0.1], [0.2]]

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
            new=AsyncMock(return_value=embeddings),
        ),
        patch.object(queries, "delete_chunks_for_memory", new=AsyncMock()),
        patch.object(
            queries,
            "insert_chunk",
            new=AsyncMock(side_effect=[uuid4(), RuntimeError("db write failed")]),
        ),
        patch.object(queries, "update_memory_status", new=AsyncMock()) as status,
    ):
        with pytest.raises(RuntimeError, match="db write failed"):
            await tasks._embed_and_store(
                agent_id=agent_id,
                memory_id=memory_id,
                texts=texts,
                document_context="full doc",
            )

    status.assert_not_awaited()
    # Transaction context exits after the failure so the driver can roll back.
    txn.__aexit__.assert_awaited()
