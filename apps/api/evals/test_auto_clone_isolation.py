"""Auto-clone failures must not mark an already-indexed memory as error."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.db import queries
from app.workers import tasks


@pytest.mark.asyncio
async def test_maybe_auto_clone_swallows_clone_errors():
    agent_id = uuid4()
    memory_id = uuid4()
    agent = {
        "id": agent_id,
        "display_name": "Maker",
        "elevenlabs_voice_id": None,
        "status": "ready",
        "created_at": None,
    }

    conn = MagicMock()
    admin_cm = MagicMock()
    admin_cm.__aenter__ = AsyncMock(return_value=conn)
    admin_cm.__aexit__ = AsyncMock(return_value=False)

    with (
        patch.object(tasks, "admin_connection", return_value=admin_cm),
        patch.object(
            queries,
            "count_indexed_by_kind",
            new=AsyncMock(return_value=queries.REQUIRED_VOICE_MEMORIES),
        ),
        patch.object(queries, "get_agent", new=AsyncMock(return_value=agent)),
        patch.object(
            tasks,
            "clone_agent_voice",
            new=AsyncMock(side_effect=RuntimeError("elevenlabs/db blew up")),
        ) as clone,
    ):
        # Must not raise — ingest would otherwise mark the memory as error.
        await tasks._maybe_auto_clone_voice({}, agent_id, memory_id=memory_id)
        clone.assert_awaited_once()


@pytest.mark.asyncio
async def test_maybe_auto_clone_skips_when_under_threshold():
    agent_id = uuid4()
    memory_id = uuid4()

    conn = MagicMock()
    admin_cm = MagicMock()
    admin_cm.__aenter__ = AsyncMock(return_value=conn)
    admin_cm.__aexit__ = AsyncMock(return_value=False)

    with (
        patch.object(tasks, "admin_connection", return_value=admin_cm),
        patch.object(
            queries,
            "count_indexed_by_kind",
            new=AsyncMock(return_value=queries.REQUIRED_VOICE_MEMORIES - 1),
        ),
        patch.object(queries, "get_agent", new=AsyncMock(return_value={"elevenlabs_voice_id": None})),
        patch.object(tasks, "clone_agent_voice", new=AsyncMock()) as clone,
    ):
        await tasks._maybe_auto_clone_voice({}, agent_id, memory_id=memory_id)
        clone.assert_not_awaited()


@pytest.mark.asyncio
async def test_ingest_leaves_memory_indexed_when_auto_clone_raises():
    """Full ingest path: successful embed must stay indexed if auto-clone fails."""
    memory_id = uuid4()
    agent_id = uuid4()
    mid = str(memory_id)

    memory = {
        "id": memory_id,
        "agent_id": agent_id,
        "kind": "voice",
        "text_content": None,
        "audio_uri": "/tmp/fake.webm",
        "duration_ms": 1000,
        "status": "pending",
        "error_message": None,
        "assemblyai_transcript_id": None,
        "created_at": None,
    }

    status_updates: list[str] = []

    async def fake_update_status(_conn, _mid, status, **kwargs):
        status_updates.append(status)

    conn = MagicMock()
    admin_cm = MagicMock()
    admin_cm.__aenter__ = AsyncMock(return_value=conn)
    admin_cm.__aexit__ = AsyncMock(return_value=False)

    utterance = MagicMock(text="hello from the porch", speaker="A", start_ms=0, end_ms=500)
    transcript = MagicMock(
        transcript_id="tx1",
        text="hello from the porch",
        utterances=[utterance],
    )

    with (
        patch.object(tasks, "init_pool", new=AsyncMock()),
        patch.object(tasks, "admin_connection", return_value=admin_cm),
        patch.object(queries, "get_memory", new=AsyncMock(return_value=memory)),
        patch.object(queries, "update_memory_status", new=AsyncMock(side_effect=fake_update_status)),
        patch.object(queries, "delete_chunks_for_memory", new=AsyncMock()),
        patch.object(queries, "insert_chunk", new=AsyncMock(return_value=uuid4())),
        patch.object(
            queries,
            "count_indexed_by_kind",
            new=AsyncMock(return_value=queries.REQUIRED_VOICE_MEMORIES),
        ),
        patch.object(
            queries,
            "get_agent",
            new=AsyncMock(
                return_value={
                    "id": agent_id,
                    "display_name": "Maker",
                    "elevenlabs_voice_id": None,
                    "status": "ready",
                    "created_at": None,
                }
            ),
        ),
        patch.object(tasks.assemblyai, "transcribe_file", return_value=transcript),
        patch.object(
            tasks.voyage,
            "embed_documents",
            new=AsyncMock(return_value=[[0.1] * 8]),
        ),
        patch.object(
            tasks,
            "clone_agent_voice",
            new=AsyncMock(side_effect=RuntimeError("clone failed after index")),
        ),
        patch.object(tasks.asyncio, "to_thread", new=AsyncMock(return_value=transcript)),
    ):
        await tasks.ingest_memory({}, mid)

    assert "indexed" in status_updates
    assert "error" not in status_updates
