from __future__ import annotations

from typing import Any

import httpx

from app.config import get_settings

VOYAGE_BASE = "https://api.voyageai.com/v1"


class VoyageError(RuntimeError):
    pass


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

    # Prefer contextualized embeddings API
    if document_context is not None:
        payload: dict[str, Any] = {
            "model": settings.voyage_embed_model,
            "inputs": [[document_context, t] for t in texts],
            "input_type": "document",
            "output_dimension": settings.voyage_embed_dim,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{VOYAGE_BASE}/contextualizedembeddings",
                headers=headers,
                json=payload,
            )
            if resp.status_code < 400:
                data = resp.json()
                # Response shape: data[].data[].embedding or data[].embeddings
                return _parse_contextualized(data)

    # Standard embeddings fallback / query path
    payload = {
        "model": settings.voyage_embed_model,
        "input": texts,
        "input_type": "document" if document_context is not None else "query",
        "output_dimension": settings.voyage_embed_dim,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{VOYAGE_BASE}/embeddings", headers=headers, json=payload)
        if resp.status_code >= 400:
            raise VoyageError(f"Voyage embed error {resp.status_code}: {resp.text}")
        data = resp.json()
        items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
        return [item["embedding"] for item in items]


async def embed_query(query: str) -> list[float]:
    vectors = await embed_documents([query], document_context=None)
    # Re-call with query input_type
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
            # fall back to previous result if any
            if vectors:
                return vectors[0]
            raise VoyageError(f"Voyage query embed error {resp.status_code}: {resp.text}")
        data = resp.json()
        items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
        return items[0]["embedding"]


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
