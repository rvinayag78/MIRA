"""Unit tests for enqueue-or-rollback and session ownership helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.db import queries
from app.routes.memories import MAX_AUDIO_BYTES, enqueue_ingest_or_rollback


@pytest.mark.asyncio
async def test_enqueue_success_does_not_delete():
    mid = uuid4()
    with (
        patch("app.routes.memories.queue.enqueue_ingest", new_callable=AsyncMock) as enq,
        patch("app.routes.memories.admin_connection") as admin_cm,
        patch("app.routes.memories.queries.delete_memory", new_callable=AsyncMock) as dele,
    ):
        await enqueue_ingest_or_rollback(mid)
        enq.assert_awaited_once_with(str(mid))
        admin_cm.assert_not_called()
        dele.assert_not_called()


@pytest.mark.asyncio
async def test_enqueue_failure_deletes_memory_and_audio(tmp_path: Path):
    mid = uuid4()
    audio = tmp_path / "clip.webm"
    audio.write_bytes(b"fake-audio")

    conn = MagicMock()
    admin_cm = MagicMock()
    admin_cm.__aenter__ = AsyncMock(return_value=conn)
    admin_cm.__aexit__ = AsyncMock(return_value=False)

    with (
        patch(
            "app.routes.memories.queue.enqueue_ingest",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Queue not initialized"),
        ),
        patch("app.routes.memories.admin_connection", return_value=admin_cm),
        patch("app.routes.memories.queries.delete_memory", new_callable=AsyncMock) as dele,
    ):
        with pytest.raises(HTTPException) as excinfo:
            await enqueue_ingest_or_rollback(mid, audio_path=audio)
        assert excinfo.value.status_code == 503
        dele.assert_awaited_once_with(conn, mid)
        assert not audio.exists()


@pytest.mark.asyncio
async def test_resolve_session_rejects_foreign_session():
    agent_a = uuid4()
    foreign = uuid4()
    created = uuid4()
    conn = AsyncMock()

    with (
        patch(
            "app.db.queries.get_owned_session",
            new_callable=AsyncMock,
            return_value=None,
        ) as owned,
        patch(
            "app.db.queries.create_session",
            new_callable=AsyncMock,
            return_value=created,
        ) as create,
    ):
        sid = await queries.resolve_session(conn, agent_a, foreign)
        assert sid == created
        owned.assert_awaited_once_with(conn, foreign, agent_a)
        create.assert_awaited_once_with(conn, agent_a)


@pytest.mark.asyncio
async def test_resolve_session_keeps_owned_session():
    agent_a = uuid4()
    owned_id = uuid4()
    conn = AsyncMock()

    with (
        patch(
            "app.db.queries.get_owned_session",
            new_callable=AsyncMock,
            return_value=owned_id,
        ),
        patch("app.db.queries.create_session", new_callable=AsyncMock) as create,
    ):
        sid = await queries.resolve_session(conn, agent_a, owned_id)
        assert sid == owned_id
        create.assert_not_called()


def test_max_audio_bytes_is_bounded():
    assert MAX_AUDIO_BYTES <= 25 * 1024 * 1024
    assert MAX_AUDIO_BYTES > 0
