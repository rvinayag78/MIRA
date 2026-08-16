from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config import get_settings

VOYAGE_BASE = "https://api.voyageai.com/v1"
# voyage-context-* is rejected by POST /embeddings; use this sibling model instead.
_STANDARD_EMBED_MODEL = "voyage-4"

logger = logging.getLogger(__name__)


class VoyageError(RuntimeError):
    pass


def _is_context_model(model: str) -> bool:
    return "context" in model.lower()


def _standard_embed_model(model: str) -> str:
    if _is_context_model(model):
        return _STANDARD_EMBED_MODEL
    return model


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


async def _post_embeddings(texts: list[str], *, input_type: str) -> list[list[float]]:
    settings = get_settings()
    model = _standard_embed_model(settings.voyage_embed_model)
    payload = {
        "model": model,
        "input": texts,
        "input_type": input_type,
        "output_dimension": settings.voyage_embed_dim,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{VOYAGE_BASE}/embeddings", headers=_headers(), json=payload)
    if resp.status_code >= 400:
        raise VoyageError(f"Voyage embed error {resp.status_code}: {resp.text}")
    items = _standard_embeddings(resp.json())
    if len(items) != len(texts):
        raise VoyageError(f"Voyage embed returned {len(items)} vectors for {len(texts)} texts")
    return items


async def _contextualized_embeddings(
    texts: list[str],
    *,
    input_type: str,
    grouped: bool,
) -> list[list[float]]:
    """voyage-context-* only works on /contextualizedembeddings, not /embeddings."""
    settings = get_settings()
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
    """Embed chunks. Context models try /contextualizedembeddings, then voyage-4."""
    if not texts:
        return []
    settings = get_settings()
    _ = document_context

    if _is_context_model(settings.voyage_embed_model):
        try:
            parsed = await _contextualized_embeddings(
                texts,
                input_type="document",
                grouped=True,
            )
            if len(parsed) == len(texts):
                return parsed
            raise VoyageError(
                f"Voyage contextualized embed returned {len(parsed)} vectors for {len(texts)} chunks"
            )
        except (VoyageError, httpx.HTTPError) as exc:
            logger.warning("Contextualized document embed failed; using %s: %s", _STANDARD_EMBED_MODEL, exc)

    return await _post_embeddings(texts, input_type="document")


async def embed_query(query: str) -> list[float]:
    settings = get_settings()
    if _is_context_model(settings.voyage_embed_model):
        try:
            parsed = await _contextualized_embeddings(
                [query],
                input_type="query",
                grouped=False,
            )
            if parsed:
                return parsed[0]
        except (VoyageError, httpx.HTTPError) as exc:
            logger.warning("Contextualized query embed failed; using %s: %s", _STANDARD_EMBED_MODEL, exc)

    return (await _post_embeddings([query], input_type="query"))[0]


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
