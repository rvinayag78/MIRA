from __future__ import annotations

from typing import Any

import httpx

from app.config import get_settings

VOYAGE_BASE = "https://api.voyageai.com/v1"


class VoyageError(RuntimeError):
    pass


def _is_context_model(model: str) -> bool:
    return "context" in model.lower()


def _headers() -> dict[str, str]:
    settings = get_settings()
    if not settings.voyage_api_key:
        raise VoyageError("VOYAGE_API_KEY is not set")
    return {
        "Authorization": f"Bearer {settings.voyage_api_key}",
        "Content-Type": "application/json",
    }


def _standard_embeddings(data: dict[str, Any]) -> list[list[float]]:
    items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
    return [item["embedding"] for item in items]


async def _contextualized_embeddings(
    texts: list[str],
    *,
    input_type: str,
    grouped: bool,
) -> list[list[float]]:
    """voyage-context-* only works on /contextualizedembeddings, not /embeddings."""
    settings = get_settings()
    # Documents: one inner list = one document's chunks.
    # Queries: flat list of query strings (Voyage also accepts [[q]]).
    inputs: list[Any] = [list(texts)] if grouped else list(texts)
    payload: dict[str, Any] = {
        "model": settings.voyage_embed_model,
        "inputs": inputs,
        "input_type": input_type,
        "output_dimension": settings.voyage_embed_dim,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{VOYAGE_BASE}/contextualizedembeddings",
            headers=_headers(),
            json=payload,
        )
    if resp.status_code >= 400:
        raise VoyageError(f"Voyage contextualized embed error {resp.status_code}: {resp.text}")
    parsed = _parse_contextualized(resp.json())
    if not parsed:
        raise VoyageError("Voyage contextualized embed returned no data")
    return parsed


async def embed_documents(texts: list[str], *, document_context: str | None = None) -> list[list[float]]:
    """
    Embed chunks at VOYAGE_EMBED_DIM.
    voyage-context-* uses the contextualized endpoint (required).
    Other models use the standard embeddings API.
    """
    if not texts:
        return []
    settings = get_settings()
    _ = document_context  # chunks in `texts` already form the document

    if _is_context_model(settings.voyage_embed_model):
        parsed = await _contextualized_embeddings(
            texts,
            input_type="document",
            grouped=True,
        )
        if len(parsed) != len(texts):
            raise VoyageError(
                f"Voyage contextualized embed returned {len(parsed)} vectors for {len(texts)} chunks"
            )
        return parsed

    payload = {
        "model": settings.voyage_embed_model,
        "input": texts,
        "input_type": "document",
        "output_dimension": settings.voyage_embed_dim,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{VOYAGE_BASE}/embeddings", headers=_headers(), json=payload)
        if resp.status_code >= 400:
            raise VoyageError(f"Voyage embed error {resp.status_code}: {resp.text}")
        return _standard_embeddings(resp.json())


async def embed_query(query: str) -> list[float]:
    settings = get_settings()
    if _is_context_model(settings.voyage_embed_model):
        parsed = await _contextualized_embeddings(
            [query],
            input_type="query",
            grouped=False,
        )
        return parsed[0]

    payload = {
        "model": settings.voyage_embed_model,
        "input": [query],
        "input_type": "query",
        "output_dimension": settings.voyage_embed_dim,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{VOYAGE_BASE}/embeddings", headers=_headers(), json=payload)
        if resp.status_code >= 400:
            raise VoyageError(f"Voyage query embed error {resp.status_code}: {resp.text}")
        items = _standard_embeddings(resp.json())
        if not items:
            raise VoyageError("Voyage query embed returned no data")
        return items[0]


def _parse_contextualized(data: dict[str, Any]) -> list[list[float]]:
    out: list[list[float]] = []
    for item in data.get("data", []):
        if isinstance(item, dict) and "embedding" in item:
            out.append(item["embedding"])
            continue
        nested = None
        if isinstance(item, dict):
            nested = item.get("data") or item.get("embeddings")
        if not isinstance(nested, list):
            continue
        for sub in nested:
            if isinstance(sub, dict) and "embedding" in sub:
                out.append(sub["embedding"])
            elif isinstance(sub, dict) and "embeddings" in sub:
                emb = sub["embeddings"]
                out.append(emb[0] if emb and isinstance(emb[0], list) else emb)
            elif isinstance(sub, list) and sub and isinstance(sub[0], (int, float)):
                out.append(sub)
    if not out:
        raise VoyageError(f"Unexpected Voyage contextualized response: {data}")
    return out


async def rerank(query: str, documents: list[str], *, top_k: int) -> list[dict[str, Any]]:
    """Voyage rerank-2.5. Returns list of {index, relevance_score} sorted by score desc."""
    settings = get_settings()
    if not documents:
        return []

    payload = {
        "model": settings.voyage_rerank_model,
        "query": query,
        "documents": documents,
        "top_k": min(top_k, len(documents)),
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{VOYAGE_BASE}/rerank", headers=_headers(), json=payload)
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
