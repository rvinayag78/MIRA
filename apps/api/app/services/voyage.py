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

    When document_context is provided, treat `texts` as pre-chunked pieces of one
    document and call the contextualized embeddings API with inputs=[texts] so each
    chunk is embedded in the context of its siblings. Falls back to the standard
    embeddings API if contextualized fails or returns the wrong count.

    `document_context` is a mode flag (truthy → document/contextualized path). The
    sibling chunks themselves supply document context; do not prepend a synthetic
    full-document chunk (that previously caused every chunk to store the wrong vector).
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

    # Prefer contextualized embeddings API for document chunks.
    # Correct shape: one inner list = one document's chunks (not [full_doc, chunk] pairs).
    if document_context is not None:
        payload: dict[str, Any] = {
            "model": settings.voyage_embed_model,
            "inputs": [texts],
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
                try:
                    parsed = _parse_contextualized(resp.json())
                    if len(parsed) == len(texts):
                        return parsed
                except VoyageError:
                    # Fall through to standard embeddings
                    pass

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
    """Flatten contextualized response into one embedding per input chunk, in order."""
    out: list[list[float]] = []
    for item in data.get("data", []):
        if "embedding" in item:
            out.append(item["embedding"])
            continue
        if "data" in item:
            # Nested: one document → list of chunk embeddings (must keep all, not just [0])
            nested = sorted(item["data"] or [], key=lambda x: x.get("index", 0))
            for nest in nested:
                if "embedding" in nest:
                    out.append(nest["embedding"])
                elif "embeddings" in nest:
                    emb = nest["embeddings"]
                    out.append(emb[0] if emb and isinstance(emb[0], list) else emb)
            continue
        if "embeddings" in item:
            emb = item["embeddings"]
            # SDK-style: embeddings is List[List[float]] for all chunks in the document
            if emb and isinstance(emb[0], list):
                out.extend(emb)
            else:
                out.append(emb)
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
