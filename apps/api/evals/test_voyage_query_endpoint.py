"""Regression: voyage-context-* queries must use contextualizedembeddings."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.voyage import VoyageError, _parse_contextualized, embed_documents, embed_query


def test_parse_contextualized_query_shape():
    data = {
        "data": [
            {
                "index": 0,
                "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}],
            }
        ]
    }
    assert _parse_contextualized(data) == [[0.1, 0.2, 0.3]]


@pytest.mark.asyncio
async def test_embed_query_uses_contextualized_endpoint():
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "data": [
            {
                "index": 0,
                "data": [{"index": 0, "embedding": [9.0, 8.0]}],
            }
        ]
    }
    fake_client = MagicMock()
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=False)
    fake_client.post = AsyncMock(return_value=fake_resp)

    with (
        patch("app.services.voyage.get_settings") as gs,
        patch("app.services.voyage.httpx.AsyncClient", return_value=fake_client),
    ):
        gs.return_value.voyage_api_key = "test-key"
        gs.return_value.voyage_embed_model = "voyage-context-4"
        gs.return_value.voyage_embed_dim = 1024
        out = await embed_query("Where was the childhood home?")

    assert out == [9.0, 8.0]
    fake_client.post.assert_awaited_once()
    args, kwargs = fake_client.post.await_args
    assert args[0].endswith("/contextualizedembeddings")
    assert kwargs["json"]["inputs"] == [["Where was the childhood home?"]]
    assert kwargs["json"]["input_type"] == "query"
    assert kwargs["json"]["model"] == "voyage-context-4"


@pytest.mark.asyncio
async def test_embed_query_never_calls_standard_embeddings():
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "data": [{"index": 0, "data": [{"index": 0, "embedding": [1.0]}]}]
    }
    fake_client = MagicMock()
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=False)
    fake_client.post = AsyncMock(return_value=fake_resp)

    with (
        patch("app.services.voyage.get_settings") as gs,
        patch("app.services.voyage.httpx.AsyncClient", return_value=fake_client),
    ):
        gs.return_value.voyage_api_key = "test-key"
        gs.return_value.voyage_embed_model = "voyage-context-4"
        gs.return_value.voyage_embed_dim = 1024
        await embed_query("hello")

    urls = [c.args[0] for c in fake_client.post.await_args_list]
    assert all("/contextualizedembeddings" in u for u in urls)
    assert not any(u.rstrip("/").endswith("/embeddings") for u in urls)


@pytest.mark.asyncio
async def test_embed_documents_fallback_stays_on_contextualized():
    """When the primary contextualized call fails, do not hit /embeddings."""
    fail_resp = MagicMock()
    fail_resp.status_code = 400
    fail_resp.text = "bad request"

    ok_resp = MagicMock()
    ok_resp.status_code = 200
    ok_resp.json.return_value = {
        "data": [
            {"index": 0, "data": [{"index": 0, "embedding": [1.0]}]},
            {"index": 1, "data": [{"index": 0, "embedding": [2.0]}]},
        ]
    }

    fake_client = MagicMock()
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=False)
    fake_client.post = AsyncMock(side_effect=[fail_resp, ok_resp])

    with (
        patch("app.services.voyage.get_settings") as gs,
        patch("app.services.voyage.httpx.AsyncClient", return_value=fake_client),
    ):
        gs.return_value.voyage_api_key = "test-key"
        gs.return_value.voyage_embed_model = "voyage-context-4"
        gs.return_value.voyage_embed_dim = 1024
        out = await embed_documents(["a", "b"], document_context="full doc")

    assert out == [[1.0], [2.0]]
    assert fake_client.post.await_count == 2
    for call in fake_client.post.await_args_list:
        assert call.args[0].endswith("/contextualizedembeddings")
    # Fallback sends one chunk per inner list
    assert fake_client.post.await_args_list[1].kwargs["json"]["inputs"] == [["a"], ["b"]]


@pytest.mark.asyncio
async def test_embed_query_error_surfaces():
    fake_resp = MagicMock()
    fake_resp.status_code = 400
    fake_resp.text = "model not found"
    fake_client = MagicMock()
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=False)
    fake_client.post = AsyncMock(return_value=fake_resp)

    with (
        patch("app.services.voyage.get_settings") as gs,
        patch("app.services.voyage.httpx.AsyncClient", return_value=fake_client),
    ):
        gs.return_value.voyage_api_key = "test-key"
        gs.return_value.voyage_embed_model = "voyage-context-4"
        gs.return_value.voyage_embed_dim = 1024
        with pytest.raises(VoyageError, match="contextualized embed error"):
            await embed_query("hello")
