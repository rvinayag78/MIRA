"""Regression: contextualized embeddings must keep every chunk vector."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.voyage import VoyageError, _parse_contextualized, embed_documents


def test_parse_contextualized_keeps_all_chunk_embeddings():
    data = {
        "data": [
            {
                "index": 0,
                "data": [
                    {"index": 0, "embedding": [0.1, 0.2]},
                    {"index": 1, "embedding": [0.3, 0.4]},
                    {"index": 2, "embedding": [0.5, 0.6]},
                ],
            }
        ]
    }
    out = _parse_contextualized(data)
    assert out == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]


def test_parse_contextualized_empty_raises():
    with pytest.raises(VoyageError):
        _parse_contextualized({"data": []})


@pytest.mark.asyncio
async def test_embed_documents_sends_chunks_as_one_document():
    texts = ["chunk a", "chunk b", "chunk c"]
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "data": [
            {
                "index": 0,
                "data": [
                    {"index": 0, "embedding": [1.0]},
                    {"index": 1, "embedding": [2.0]},
                    {"index": 2, "embedding": [3.0]},
                ],
            }
        ]
    }

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=fake_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("app.services.voyage.get_settings") as gs,
        patch("app.services.voyage.httpx.AsyncClient", return_value=mock_client),
    ):
        gs.return_value = MagicMock(
            voyage_api_key="test-key",
            voyage_embed_model="voyage-context-4",
            voyage_embed_dim=1024,
        )
        vectors = await embed_documents(texts, document_context="full transcript unused")

    assert vectors == [[1.0], [2.0], [3.0]]
    kwargs = mock_client.post.call_args.kwargs
    assert kwargs["json"]["inputs"] == [texts]
    # Must NOT send [[full, chunk], ...] which stored the wrong vector per chunk
    assert kwargs["json"]["inputs"] != [["full transcript unused", t] for t in texts]


@pytest.mark.asyncio
async def test_embed_documents_falls_back_when_count_mismatches():
    texts = ["chunk a", "chunk b"]
    ctx_resp = MagicMock()
    ctx_resp.status_code = 200
    # Bug-shaped response: only one embedding for two chunks
    ctx_resp.json.return_value = {
        "data": [{"index": 0, "data": [{"index": 0, "embedding": [9.0]}]}]
    }

    std_resp = MagicMock()
    std_resp.status_code = 200
    std_resp.json.return_value = {
        "data": [
            {"index": 0, "embedding": [1.0]},
            {"index": 1, "embedding": [2.0]},
        ]
    }

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=[ctx_resp, std_resp])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("app.services.voyage.get_settings") as gs,
        patch("app.services.voyage.httpx.AsyncClient", return_value=mock_client),
    ):
        gs.return_value = MagicMock(
            voyage_api_key="test-key",
            voyage_embed_model="voyage-context-4",
            voyage_embed_dim=1024,
        )
        vectors = await embed_documents(texts, document_context="doc")

    assert vectors == [[1.0], [2.0]]
    assert mock_client.post.call_count == 2
