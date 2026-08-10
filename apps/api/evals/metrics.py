from __future__ import annotations

from typing import Iterable, Sequence


def recall_at_k(relevant: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    if not relevant:
        # For refuse questions, success is retrieving nothing critical — treat as 1.0 if empty relevant
        return 1.0
    top = list(retrieved)[:k]
    hits = sum(1 for r in relevant if r in top)
    return hits / len(relevant)


def mean_recall_at_k(
    cases: Iterable[tuple[Sequence[str], Sequence[str]]], k: int
) -> float:
    scores = [recall_at_k(rel, ret, k) for rel, ret in cases]
    return sum(scores) / len(scores) if scores else 0.0


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    *,
    k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
