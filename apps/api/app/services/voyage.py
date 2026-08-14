from __future__ import annotations

from typing import Any

import httpx

from app.config import get_settings

VOYAGE_BASE = "https://api.voyageai.com/v1"


class VoyageError(RuntimeError):
    pass


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


async def _contextualized_embed(
    inputs: list[list[str]],
    *,
    input_type: str,
    timeout: float = 120.0,
) -> list[list[float]]:
    """Call /contextualizedembeddings (required for voyage-context-* models)."""
    settings = get_settings()
    if not settings.voyage_api_key:
        raise VoyageError("VOYAGE_API_KEY is not set")

    payload: dict[str, Any] = {
        "model": settings.voyage_embed_model,
        "inputs": inputs,
        "input_type": input_type,
        "output_dimension": settings.voyage_embed_dim,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            f"{VOYAGE_BASE}/contextualizedembeddings",
            headers=_headers(settings.voyage_api_key),
            json=payload,
        )
        if resp.status_code >= 400:
            raise VoyageError(
                f"Voyage contextualized embed error {resp.status_code}: {resp.text}"
            )
        return _parse_contextualized(resp.json())


async def embed_documents(texts: list[str], *, document_context: str | None = None) -> list[list[float]]:
    """
    Embed with voyage-context-4 at 1024d via the contextualized embeddings API.

    voyage-context-* models are not valid on /v1/embeddings; queries and documents
    must both use /v1/contextualizedembeddings so vectors share one space.
    """
    if not texts:
        return []
    settings = get_settings()
    if not settings.voyage_api_key:
        raise VoyageError("VOYAGE_API_KEY is not set")

    # Prefer contextualized embeddings API when a document context is provided.
    # (Input shape for multi-chunk docs is addressed separately; keep call site stable.)
    if document_context is not None:
        try:
            return await _contextualized_embed(
                [[document_context, t] for t in texts],
                input_type="document",
            )
        except VoyageError:
            # Fall through to per-chunk contextualized embeds (still same model/endpoint).
            pass

    # Independent chunk embeds on the contextualized endpoint (not /embeddings).
    # Always document input_type here — queries go through embed_query().
    return await _contextualized_embed([[t] for t in texts], input_type="document")


async def embed_query(query: str) -> list[float]:
    """
    Embed a retrieval query with voyage-context-4.

    Voyage requires context models to use /contextualizedembeddings with one
    query per inner list: inputs=[[query]], input_type=\"query\".
    """
    vectors = await _contextualized_embed([[query]], input_type="query", timeout=60.0)
    if not vectors:
        raise VoyageError("Voyage query embed returned no vectors")
    return vectors[0]


def _parse_contextualized(data: dict[str, Any]) -> list[list[float]]:
    out: list[list[float]] = []
    for item in data.get("data", []):
        if "embedding" in item:
            out.append(item["embedding"])
        elif "data" in item:
            # nested: list of chunk embeddings per document
            nested = item["data"]
            if nested and "embedding" in nested[0]:
                # we sent one chunk per input pair; take first
                out.append(nested[0]["embedding"])
            elif nested and "embeddings" in nested[0]:
                out.append(nested[0]["embeddings"][0])
        elif "embeddings" in item:
            emb = item["embeddings"]
            out.append(emb[0] if isinstance(emb[0], list) else emb)
    if not out:
        raise VoyageError(f"Unexpected Voyage contextualized response: {data}")
    return out


async def rerank(query: str, documents: list[str], *, top_k: int) -> list[dict[str, Any]]:
    """Voyage rerank-2.5. Returns list of {index, relevance_score} sorted by score desc."""
    settings = get_settings()
    if not settings.voyage_api_key:
        raise VoyageError("VOYAGE_API_KEY is not set")
    if not documents:
        return []

    payload = {
        "model": settings.voyage_rerank_model,
        "query": query,
        "documents": documents,
        "top_k": min(top_k, len(documents)),
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{VOYAGE_BASE}/rerank",
            headers=_headers(settings.voyage_api_key),
            json=payload,
        )
        if resp.status_code >= 400:
            raise VoyageError(f"Voyage rerank error {resp.status_code}: {resp.text}")
        data = resp.json()
        results = data.get("data") or data.get("results") or []
        return [
            {
                "index": r.get("index", r.get("document_index")),
                "relevance_score": r.get("relevance_score", r.get("score", 0.0)),
            }
            for r in results
        ]
