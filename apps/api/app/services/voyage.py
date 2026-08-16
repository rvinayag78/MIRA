from __future__ import annotations

from typing import Any

import httpx

from app.config import get_settings

VOYAGE_BASE = "https://api.voyageai.com/v1"


class VoyageError(RuntimeError):
    pass


def _standard_embeddings(data: dict[str, Any]) -> list[list[float]]:
    items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
    return [item["embedding"] for item in items]


async def embed_documents(texts: list[str], *, document_context: str | None = None) -> list[list[float]]:
    """
    Embed with voyage-context-4 at 1024d.
    Uses contextualized embeddings when a document context is provided.
    Falls back to standard embeddings API if contextualized endpoint fails.
    """
    if not texts:
        return []
    settings = get_settings()
    if not settings.voyage_api_key:
        raise VoyageError("VOYAGE_API_KEY is not set")

    headers = {
        "Authorization": f"Bearer {settings.voyage_api_key}",
        "Content-Type": "application/json",
    }

    # Prefer contextualized embeddings: one document whose chunks are `texts`.
    if document_context is not None:
        payload: dict[str, Any] = {
            "model": settings.voyage_embed_model,
            "inputs": [list(texts)],
            "input_type": "document",
            "output_dimension": settings.voyage_embed_dim,
        }
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{VOYAGE_BASE}/contextualizedembeddings",
                    headers=headers,
                    json=payload,
                )
            if resp.status_code < 400:
                parsed = _parse_contextualized(resp.json())
                if len(parsed) == len(texts):
                    return parsed
        except (VoyageError, httpx.HTTPError, ValueError, KeyError, TypeError):
            pass

    payload = {
        "model": settings.voyage_embed_model,
        "input": texts,
        "input_type": "document",
        "output_dimension": settings.voyage_embed_dim,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{VOYAGE_BASE}/embeddings", headers=headers, json=payload)
        if resp.status_code >= 400:
            raise VoyageError(f"Voyage embed error {resp.status_code}: {resp.text}")
        return _standard_embeddings(resp.json())


async def embed_query(query: str) -> list[float]:
    settings = get_settings()
    if not settings.voyage_api_key:
        raise VoyageError("VOYAGE_API_KEY is not set")
    headers = {
        "Authorization": f"Bearer {settings.voyage_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.voyage_embed_model,
        "input": [query],
        "input_type": "query",
        "output_dimension": settings.voyage_embed_dim,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{VOYAGE_BASE}/embeddings", headers=headers, json=payload)
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
    if not settings.voyage_api_key:
        raise VoyageError("VOYAGE_API_KEY is not set")
    if not documents:
        return []

    headers = {
        "Authorization": f"Bearer {settings.voyage_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.voyage_rerank_model,
        "query": query,
        "documents": documents,
        "top_k": min(top_k, len(documents)),
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{VOYAGE_BASE}/rerank", headers=headers, json=payload)
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
